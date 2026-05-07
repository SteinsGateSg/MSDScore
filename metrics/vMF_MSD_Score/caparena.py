import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm



DEFAULT_JSON_PATH = os.environ.get("CAPARENA_JSON", "data/caparena/caparena_annots_eval.json")
DEFAULT_IMG_ROOT = os.environ.get("CAPARENA_IMG_ROOT", "data/caparena/images")
DEFAULT_LLAVA_REPO = os.environ.get("LLAVA_REPO", "")
DEFAULT_MODEL_PATH = os.environ.get("MSD_LLAVA_MODEL_PATH", "")
DEFAULT_MODEL_BASE = os.environ.get("MSD_LLAVA_MODEL_BASE", "")
DEFAULT_MODEL_NAME = os.environ.get("MSD_LLAVA_MODEL_NAME", "llava")
DEFAULT_DEVICE_MAP = os.environ.get("DEVICE_MAP", "cuda" if torch.cuda.is_available() else "cpu")

MAX_TEXT_LEN = 2048


KAPPA = 20.0
ALPHA = 0.1
TAU = 0.2
L0 = 20.0
TAU_L = 3.0
VMF_IMG_COMPONENTS = 5
VMF_TXT_COMPONENTS = 3

tokenizer = None
model = None
image_processor = None
context_len = None
TEXT_DEVICE = None
MM_DEVICE = None
VISION_DEVICE = None


def _get_param_device(module: torch.nn.Module) -> torch.device:
    for p in module.parameters():
        return p.device
    return torch.device("cpu")


def init_llava(args):
    global tokenizer, model, image_processor, context_len
    global TEXT_DEVICE, MM_DEVICE, VISION_DEVICE

    if not args.model_path:
        raise ValueError("Set --model_path or MSD_LLAVA_MODEL_PATH for CapArena evaluation.")

    if args.llava_repo and args.llava_repo not in sys.path:
        sys.path.insert(0, args.llava_repo)

    try:
        from llava.model.builder import load_pretrained_model
    except ImportError as exc:
        raise ImportError(
            "Cannot import llava. Install LLaVA or pass --llava_repo pointing to a local LLaVA checkout."
        ) from exc

    model_base = args.model_base or None
    print(f"[Init] Loading LLaVA from: {args.model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        model_base,
        args.model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device_map=args.device_map,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    TEXT_DEVICE = _get_param_device(model.get_input_embeddings())
    MM_DEVICE = _get_param_device(model.get_model().mm_projector)
    VISION_DEVICE = model.get_vision_tower().device


# ------------------------- Elo -------------------------
def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def elo_update(r_a, r_b, s_a, k=32.0):
    e_a = elo_expected(r_a, r_b)
    r_a_new = r_a + k * (s_a - e_a)
    r_b_new = r_b + k * ((1.0 - s_a) - (1.0 - e_a))
    return r_a_new, r_b_new


def winner_to_score(winner, source1, source2):
    if winner == source1:
        return 1.0
    if winner == source2:
        return 0.0
    if winner == "tie":
        return 0.5
    return None


def rank_corr(rank_a_names, rank_b_names):
    a = [m for m in rank_a_names if m != "human"]
    b = [m for m in rank_b_names if m != "human"]
    common = [m for m in a if m in b]
    if len(common) < 2:
        return {"common": common, "spearman": 0.0, "kendall": 0.0}

    a_pos = np.arange(1, len(common) + 1, dtype=np.float64)
    b_pos = np.array([b.index(m) + 1 for m in common], dtype=np.float64)
    rho, _ = spearmanr(b_pos, a_pos)
    tau, _ = kendalltau(b_pos, a_pos)
    rho = 0.0 if np.isnan(rho) else float(rho)
    tau = 0.0 if np.isnan(tau) else float(tau)
    return {"common": common, "spearman": rho, "kendall": tau}


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VMFMixture:
    def __init__(self, n_components=3, kappa=20.0, max_iter=20, eps=1e-9, reinit_thresh=1e-6):
        self.n_components = int(n_components)
        self.fixed_kappa = float(kappa)
        self.max_iter = int(max_iter)
        self.eps = float(eps)
        self.reinit_thresh = float(reinit_thresh)
        self.device = None
        self.mu = None
        self.weights = None

    def to(self, device):
        self.device = device
        if self.mu is not None:
            self.mu = self.mu.to(device)
        if self.weights is not None:
            self.weights = self.weights.to(device)
        return self

    @staticmethod
    def _l2norm(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def fit(self, x: torch.Tensor):
        if self.device is None:
            self.device = x.device
        x = x.to(self.device)
        x = self._l2norm(x)

        N, _ = x.shape
        K = self.n_components
        if N >= K:
            idx = torch.randperm(N, device=self.device)[:K]
        else:
            idx = torch.randint(0, N, (K,), device=self.device)

        self.mu = self._l2norm(x[idx].clone())
        self.weights = torch.full((K,), 1.0 / K, device=self.device)

        for _ in range(self.max_iter):
            dot = x @ self.mu.T
            log_w = torch.log(self.weights.clamp_min(self.eps))
            logits = self.fixed_kappa * dot + log_w.unsqueeze(0)
            posterior = F.softmax(logits, dim=1)

            N_k = posterior.sum(dim=0)
            self.weights = (N_k / N).clamp_min(self.eps)
            self.weights = self.weights / self.weights.sum()

            weighted_sum = posterior.T @ x

            dead = N_k < (self.reinit_thresh * N)
            if dead.any():
                n_dead = int(dead.sum().item())
                re_idx = torch.randint(0, N, (n_dead,), device=self.device)
                weighted_sum[dead] = x[re_idx]
                self.weights[dead] = 1.0 / N
                self.weights = self.weights / self.weights.sum()

            self.mu = self._l2norm(weighted_sum)

        return self

    def score_samples(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self._l2norm(x)
        dot = x @ self.mu.T
        log_w = torch.log(self.weights.clamp_min(self.eps))
        logits = self.fixed_kappa * dot + log_w.unsqueeze(0)
        return torch.logsumexp(logits, dim=1)


# ------------------------- Feature extraction (LLaVA) -------------------------
def get_image_features(image):
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=VISION_DEVICE, dtype=model.dtype)

    with torch.inference_mode():
        image_features = model.encode_images(pixel_values)

        prompt_len = int(getattr(model.config, "prompt_length", 0))
        if prompt_len > 0 and image_features.shape[1] > prompt_len:
            image_features = image_features[:, prompt_len:, :]

        img_patch = F.normalize(image_features, dim=-1)
        img_global = F.normalize(img_patch.mean(dim=1), dim=-1)

    return {
        "img_patch": img_patch.squeeze(0).cpu().numpy(),
        "img_global": img_global.cpu().numpy(),
    }


def _build_token_mask(input_ids, attention_mask):
    mask = attention_mask.bool()
    if tokenizer.bos_token_id is not None:
        mask &= input_ids != tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        mask &= input_ids != tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        mask &= input_ids != tokenizer.pad_token_id
    return mask


def get_text_features(text_str):
    inputs = tokenizer(
        text_str,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=min(MAX_TEXT_LEN, context_len),
    )
    input_ids = inputs["input_ids"].to(TEXT_DEVICE)
    attention_mask = inputs["attention_mask"].to(TEXT_DEVICE)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]

        token_mask = _build_token_mask(input_ids, attention_mask)
        if token_mask.sum() == 0:
            token_mask = attention_mask.bool()

        tokens = last_hidden[0][token_mask[0]]
        txt_token = F.normalize(tokens, dim=-1)

        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        txt_global = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / denom
        txt_global = F.normalize(txt_global, dim=-1)

    return {
        "txt_token": txt_token.cpu().numpy(),
        "txt_global": txt_global.cpu().numpy(),
    }


def calc_cosine(image_feats, text_feats):
    img_global = torch.tensor(image_feats["img_global"]).to(MM_DEVICE)
    txt_global = torch.tensor(text_feats["txt_global"]).to(MM_DEVICE)
    return (img_global @ txt_global.T).item()


def calc_bi_kl(image_feats, text_feats, kappa=KAPPA):
    img_patch = image_feats["img_patch"]
    txt_token = text_feats["txt_token"]

    img_data = torch.tensor(img_patch, device=MM_DEVICE, dtype=torch.float32)
    txt_data = torch.tensor(txt_token, device=MM_DEVICE, dtype=torch.float32)

    if txt_data.shape[0] == 0:
        return 10.0

    vmf_img = VMFMixture(n_components=VMF_IMG_COMPONENTS, kappa=kappa).to(MM_DEVICE)
    vmf_img.fit(img_data)
    vmf_txt = VMFMixture(n_components=VMF_TXT_COMPONENTS, kappa=kappa).to(MM_DEVICE)
    vmf_txt.fit(txt_data)

    log_p_img_x = vmf_img.score_samples(img_data)
    log_q_txt_x = vmf_txt.score_samples(img_data)
    kl_i2t = torch.mean(log_p_img_x - log_q_txt_x).item()

    log_q_txt_y = vmf_txt.score_samples(txt_data)
    log_p_img_y = vmf_img.score_samples(txt_data)
    kl_t2i = torch.mean(log_q_txt_y - log_p_img_y).item()

    L = txt_data.shape[0]
    exponent = (L - L0) / TAU_L
    beta = 1.0 / (1.0 + np.exp(exponent))

    bi_kl = beta * kl_i2t + (1.0 - beta) * kl_t2i
    return bi_kl


def soft_gated_msd_scores(cos_scores, div_scores, alpha=ALPHA, tau=TAU):
    M = cos_scores.numel()
    p = F.softmax(cos_scores / tau, dim=0)
    if M <= 1:
        u = cos_scores.new_tensor(1.0)
    else:
        u = (M / (M - 1.0)) * (1.0 - p.max())
        u = torch.clamp(u, 0.0, 1.0)
    final = cos_scores - alpha * u * div_scores
    return final


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON_PATH)
    parser.add_argument("--img_root", type=str, default=DEFAULT_IMG_ROOT)
    parser.add_argument("--llava_repo", type=str, default=DEFAULT_LLAVA_REPO, help="Path to a LLaVA repository, if it is not pip-installed.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="LLaVA or local-aligner checkpoint path.")
    parser.add_argument("--model_base", type=str, default=DEFAULT_MODEL_BASE)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device_map", type=str, default=DEFAULT_DEVICE_MAP)
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffle_rounds", type=int, default=1)
    parser.add_argument("--k", type=float, default=32.0, help="Elo K factor")
    parser.add_argument("--eps", type=float, default=1e-4, help="tie threshold for metric diff (abs(diff) < eps => tie)")
    parser.add_argument("--only_in_400", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    init_llava(args)

    device = MM_DEVICE

    print(f"Loading data from {args.json_path} ...")
    with open(args.json_path, "r", encoding="utf-8") as f:
        raw_data_all = json.load(f)

    if args.only_in_400:
        raw_data = [x for x in raw_data_all if bool(x.get("in-400", False))]
        print(f"Filtered in-400=true: {len(raw_data)} / {len(raw_data_all)}")
    else:
        raw_data = raw_data_all
        print(f"Loaded {len(raw_data)} samples.")

    if args.max_samples is not None:
        raw_data = raw_data[: args.max_samples]
        print(f"Truncated to {len(raw_data)} samples.")

    level_stats = defaultdict(lambda: {"correct": 0, "total": 0, "samples": 0})
    records = []
    skip_count = 0

    print("\nRunning evaluation...")
    for item in tqdm(raw_data, desc="Scoring"):
        winner = item.get("winner")
        source1 = item.get("source1")
        source2 = item.get("source2")
        cluster = item.get("cluster", "unknown")

        if winner == source1:
            human_label = 1
        elif winner == source2:
            human_label = -1
        elif winner == "tie":
            human_label = 0
        else:
            skip_count += 1
            continue

        if not source1 or not source2:
            skip_count += 1
            continue

        img_name = item.get("img", "")
        img_path = os.path.join(args.img_root, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skip_count += 1
            continue

        caption1 = item.get("caption1", "")
        caption2 = item.get("caption2", "")

        try:
            img_feats = get_image_features(image)
            txt1_feats = get_text_features(caption1)
            txt2_feats = get_text_features(caption2)

            s1_cos = calc_cosine(img_feats, txt1_feats)
            s2_cos = calc_cosine(img_feats, txt2_feats)
            s1_kl = calc_bi_kl(img_feats, txt1_feats)
            s2_kl = calc_bi_kl(img_feats, txt2_feats)

            cos_vec = torch.tensor([s1_cos, s2_cos], device=device)
            kl_vec = torch.tensor([s1_kl, s2_kl], device=device)
            final_scores = soft_gated_msd_scores(cos_vec, kl_vec, alpha=ALPHA, tau=TAU)

            score_diff = final_scores[0].item() - final_scores[1].item()
        except Exception:
            skip_count += 1
            continue

        level_stats[cluster]["samples"] += 1
        if human_label != 0:
            if score_diff > args.eps:
                pred_label = 1
            elif score_diff < -args.eps:
                pred_label = -1
            else:
                pred_label = 0
            if pred_label == human_label:
                level_stats[cluster]["correct"] += 1
            level_stats[cluster]["total"] += 1

        records.append(
            {
                "source1": source1,
                "source2": source2,
                "winner": winner,
                "score_diff": score_diff,
                "human_label": human_label,
            }
        )

    print("\n" + "=" * 70)
    print("CapArena Accuracy by Level (non-tie only)")
    print("-" * 70)
    total_correct = 0
    total_count = 0
    for level in sorted(level_stats.keys()):
        stat = level_stats[level]
        acc = stat["correct"] / stat["total"] if stat["total"] > 0 else 0.0
        print(f"{level:<12} | Acc {acc:.2%} ({stat['total']}) | Samples {stat['samples']}")
        total_correct += stat["correct"]
        total_count += stat["total"]
    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    print("-" * 70)
    print(f"{'OVERALL':<12} | Acc {overall_acc:.2%} ({total_count})")
    print(f"Skipped/Errors: {skip_count}")
    print("=" * 70)

    if not records:
        print("No valid records to compute Elo.")
        return

    ratings_metric_sum = defaultdict(float)
    ratings_human_sum = defaultdict(float)
    games_sum = defaultdict(int)

    for round_idx in range(args.shuffle_rounds):
        rng = random.Random(args.seed + round_idx)
        data_round = list(records)
        rng.shuffle(data_round)

        ratings_metric = defaultdict(lambda: 1000.0)
        ratings_human = defaultdict(lambda: 1000.0)
        games = defaultdict(int)

        for rec in tqdm(data_round, desc=f"Elo (round {round_idx+1}/{args.shuffle_rounds})"):
            source1 = rec["source1"]
            source2 = rec["source2"]
            winner = rec["winner"]
            score_diff = rec["score_diff"]

            if score_diff > args.eps:
                s_m = 1.0
            elif score_diff < -args.eps:
                s_m = 0.0
            else:
                s_m = 0.5

            ratings_metric[source1], ratings_metric[source2] = elo_update(
                ratings_metric[source1], ratings_metric[source2], s_m, k=args.k
            )

            s_h = winner_to_score(winner, source1, source2)
            if s_h is not None:
                ratings_human[source1], ratings_human[source2] = elo_update(
                    ratings_human[source1], ratings_human[source2], s_h, k=args.k
                )

            games[source1] += 1
            games[source2] += 1

        for m, r in ratings_metric.items():
            ratings_metric_sum[m] += r
        for m, r in ratings_human.items():
            ratings_human_sum[m] += r
        for m, g in games.items():
            games_sum[m] += g

    rounds = float(args.shuffle_rounds)
    ratings_metric_avg = {m: r / rounds for m, r in ratings_metric_sum.items()}
    ratings_human_avg = {m: r / rounds for m, r in ratings_human_sum.items()}

    metric_sorted = sorted(ratings_metric_avg.items(), key=lambda x: x[1], reverse=True)
    human_sorted = sorted(ratings_human_avg.items(), key=lambda x: x[1], reverse=True)

    metric_rank_names = [m for m, _ in metric_sorted if m != "human"]
    human_rank_names = [m for m, _ in human_sorted if m != "human"]

    corr = rank_corr(metric_rank_names, human_rank_names)

    print("\n" + "=" * 70)
    print("Model-level correlation (metric Elo vs human Elo)")
    print(f"Intersection: {len(corr['common'])}")
    print(f"Spearman r : {corr['spearman']:.4f}")
    print(f"Kendall tau: {corr['kendall']:.4f}")
    print("=" * 70)

    print("\nTop-20 Elo (metric):")
    for m, r in metric_sorted[:20]:
        print(f"  {m:<25}  Elo={r:8.2f}  games={games_sum.get(m,0)}")

    print("\nTop-20 Elo (human-from-json):")
    for m, r in human_sorted[:20]:
        print(f"  {m:<25}  Elo={r:8.2f}  games={games_sum.get(m,0)}")


if __name__ == "__main__":
    main()
