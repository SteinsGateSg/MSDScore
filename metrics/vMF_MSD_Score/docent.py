import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from scipy.stats import kendalltau, pearsonr, spearmanr

MAX_TEXT_LEN = 2048
DEFAULT_LLAVA_REPO = os.environ.get("LLAVA_REPO", "")
DEFAULT_MODEL_PATH = os.environ.get("MSD_LLAVA_MODEL_PATH", "")
DEFAULT_MODEL_BASE = os.environ.get("MSD_LLAVA_MODEL_BASE", "")
DEFAULT_MODEL_NAME = os.environ.get("MSD_LLAVA_MODEL_NAME", "llava")
DEFAULT_DEVICE_MAP = os.environ.get("DEVICE_MAP", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_COARSE_PATH = os.environ.get("DOCENT_COARSE_PATH", "data/docent/coarse")
DEFAULT_IMAGE_PATH = os.environ.get("DOCENT_IMAGE_PATH", "data/docent/images")
DEFAULT_COARSE_SPLIT = os.environ.get("DOCENT_COARSE_SPLIT", "train")
DEFAULT_IMAGE_SPLIT = os.environ.get("DOCENT_IMAGE_SPLIT", "train")


KAPPA = 20.0      
ALPHA = 0.1      
TAU = 0.2         
L0 = 20.0           
TAU_L = 3.0       
# ============================================

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
        raise ValueError("Set --model_path or MSD_LLAVA_MODEL_PATH for DocENT evaluation.")

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

        N, D = x.shape
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
    vmf_img = VMFMixture(n_components=5, kappa=kappa).to(MM_DEVICE)
    vmf_img.fit(img_data)
    
    vmf_txt = VMFMixture(n_components=3, kappa=kappa).to(MM_DEVICE)
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
    parser.add_argument("--coarse_path", type=str, default=DEFAULT_COARSE_PATH)
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--coarse_split", type=str, default=DEFAULT_COARSE_SPLIT)
    parser.add_argument("--image_split", type=str, default=DEFAULT_IMAGE_SPLIT)
    parser.add_argument("--llava_repo", type=str, default=DEFAULT_LLAVA_REPO, help="Path to a LLaVA repository, if it is not pip-installed.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="LLaVA or local-aligner checkpoint path.")
    parser.add_argument("--model_base", type=str, default=DEFAULT_MODEL_BASE)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device_map", type=str, default=DEFAULT_DEVICE_MAP)
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    init_llava(args)

    print("\n[Step 1] Loading coarse annotations...")
    try:
        ds_coarse = load_dataset("parquet", data_dir=args.coarse_path, split=args.coarse_split)
    except Exception as e:
        print(f"Failed to load coarse annotations: {e}")
        return

    print("\n[Step 2] Loading images and building index...")
    try:
        ds_images = load_dataset("parquet", data_dir=args.image_path, split=args.image_split)
    except Exception as e:
        print(f"Failed to load images: {e}")
        return

    uuid_to_image = {}
    print("Building in-memory index (UUID -> Image)...")
    for row in tqdm(ds_images, desc="Indexing"):
        uuid_to_image[str(row["uuid"]).strip()] = row["image"].convert("RGB")
    print(f"Index ready, {len(uuid_to_image)} images.")

    label_map = {
        "1_much_better": 2.0,
        "1_slightly_better": 1.0,
        "equal": 0.0,
        "2_slightly_better": -1.0,
        "2_much_better": -2.0,
    }

    all_human_labels = []
    all_pred_diffs = []
    skip_count = 0

    print("\n[Step 3] Running evaluation loop...")
    for item in tqdm(ds_coarse, desc="Evaluating"):
        uuid = str(item["uuid"]).strip()
        caption1 = item["model1_generation"]
        caption2 = item["model2_generation"]
        human_text_label = item["overall_quality"]

        if uuid not in uuid_to_image:
            skip_count += 1
            continue
        if human_text_label not in label_map:
            continue

        image = uuid_to_image[uuid]
        human_score = label_map[human_text_label]

        try:
            img_feats = get_image_features(image)
            txt1_feats = get_text_features(caption1)
            txt2_feats = get_text_features(caption2)

            s1_cos = calc_cosine(img_feats, txt1_feats)
            s2_cos = calc_cosine(img_feats, txt2_feats)
            
            s1_kl = calc_bi_kl(img_feats, txt1_feats)
            s2_kl = calc_bi_kl(img_feats, txt2_feats)

            cos_vec = torch.tensor([s1_cos, s2_cos], device=MM_DEVICE)
            kl_vec = torch.tensor([s1_kl, s2_kl], device=MM_DEVICE)
            
            final_scores = soft_gated_msd_scores(cos_vec, kl_vec, alpha=ALPHA, tau=TAU)

            pred_diff = final_scores[0].item() - final_scores[1].item()
            all_human_labels.append(human_score)
            all_pred_diffs.append(pred_diff)
        except Exception as e:
            print(f"Error processing {uuid}: {e}")
            skip_count += 1

    print("\n" + "=" * 60)
    print("DOCENT Benchmark Results (LLaVA + Soft-MSD)")
    print(f"Valid samples: {len(all_human_labels)} (skipped/errors: {skip_count})")
    print("-" * 60)

    if len(all_human_labels) < 2:
        print("Not enough samples to compute correlations.")
        return

    h = np.array(all_human_labels)
    p = np.array(all_pred_diffs)

    tau, _ = kendalltau(h, p)
    rho, _ = spearmanr(h, p)
    r, _ = pearsonr(h, p)

    non_tie_mask = (h != 0)
    if np.sum(non_tie_mask) > 0:
        h_clean = h[non_tie_mask]
        p_clean = p[non_tie_mask]
        correct = np.sum((h_clean > 0) == (p_clean > 0))
        acc = correct / len(h_clean)
    else:
        acc = 0.0

    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 35)
    print(f"{'Kendall Tau':<20} | {tau:.4f}")
    print(f"{'Spearman Rho':<20} | {rho:.4f}")
    print(f"{'Pearson R':<20} | {r:.4f}")
    print(f"{'Binary Accuracy':<20} | {acc:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
