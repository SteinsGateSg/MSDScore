import argparse
import json
import os
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


K_IMG = 3
K_TXT = 2
KAPPA = 20.0
VMF_MAX_ITER = 20
ALPHA_MSD = 0.1
ALPHA_SOFT = 0.1
TAU_SOFT = 0.2
L0 = 20.0
TAU_L = 3.0

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)



def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def soft_gate_u_from_cos_delta(delta_cos: np.ndarray, tau: float) -> np.ndarray:
    p = sigmoid_stable(delta_cos / max(tau, 1e-8))
    u = 1.0 - 2.0 * np.abs(p - 0.5)
    return np.clip(u, 0.0, 1.0)


class VMFMixtureFixedKappa:
    def __init__(self, n_components: int, kappa: float = 20.0, max_iter: int = 20, seed: int = 0, device: str = "cuda"):
        self.K = int(n_components)
        self.kappa = float(kappa)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.mu: Optional[torch.Tensor] = None     
        self.weights: Optional[torch.Tensor] = None  

    def fit(self, x: torch.Tensor):
        x = x.to(self.device)
        x = l2norm(x, dim=1)
        N, D = x.shape
        K = max(1, min(self.K, N))

        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)
        if N >= K:
            idx = torch.randperm(N, generator=g, device=self.device)[:K]
        else:
            idx = torch.randint(0, N, (K,), generator=g, device=self.device)

        mu = l2norm(x[idx].clone(), dim=1)
        w = torch.full((K,), 1.0 / K, device=self.device)

        for _ in range(self.max_iter):
            logits = self.kappa * (x @ mu.T) + torch.log(w.unsqueeze(0) + 1e-9)
            post = F.softmax(logits, dim=1)
            Nk = post.sum(dim=0) + 1e-8
            mu = l2norm(post.T @ x, dim=1)
            w = Nk / float(N)

        self.mu = mu.detach()
        self.weights = w.detach()
        return self


def vmf_logp(x: torch.Tensor, mu: torch.Tensor, weights: torch.Tensor, kappa: float) -> torch.Tensor:
    """
    x: [N, D] normalized
    mu: [K, D] normalized
    weights: [K]
    return log p(x) under mixture, shape [N]
    """
    logits = kappa * (x @ mu.T) + torch.log(weights.unsqueeze(0) + 1e-9)
    return torch.logsumexp(logits, dim=1)


@torch.no_grad()
def local_cost_bi_kl_from_imgpatch(
    img_patch: torch.Tensor,         
    img_mu: torch.Tensor,             
    img_w: torch.Tensor,            
    txt_tok: torch.Tensor,           
    kappa: float,
    vmf_max_iter: int,
    seed: int,
    device: str,
) -> float:
    if img_patch.numel() == 0 or txt_tok.numel() == 0:
        return 1e9

    vmf_txt = VMFMixtureFixedKappa(
        n_components=K_TXT, kappa=kappa, max_iter=vmf_max_iter, seed=seed + 123, device=device
    ).fit(txt_tok)

    log_p_img_x = vmf_logp(img_patch, img_mu, img_w, kappa=kappa)
    log_q_txt_x = vmf_logp(img_patch, vmf_txt.mu, vmf_txt.weights, kappa=kappa)
    kl_i2t = (log_p_img_x - log_q_txt_x).mean()

    log_q_txt_y = vmf_logp(txt_tok, vmf_txt.mu, vmf_txt.weights, kappa=kappa)
    log_p_img_y = vmf_logp(txt_tok, img_mu, img_w, kappa=kappa)
    kl_t2i = (log_q_txt_y - log_p_img_y).mean()

    L = txt_tok.shape[0]
    exponent = (L - L0) / TAU_L
    beta = 1.0 / (1.0 + np.exp(exponent))

    bi_kl = beta * kl_i2t + (1.0 - beta) * kl_t2i
    return float(bi_kl.item())

@torch.no_grad()
def clip_image_global_and_patches(model: CLIPModel, processor: CLIPProcessor, image: Image.Image, device: str):
    inputs = processor(images=image, return_tensors="pt").to(device)
    v_out = model.vision_model(inputs.pixel_values)
    patch_tokens = v_out.last_hidden_state[:, 1:, :]          # [1, Np, Hv]
    img_patch = model.visual_projection(patch_tokens)         # [1, Np, D]
    img_patch = l2norm(img_patch, dim=-1).squeeze(0)          # [Np, D]

    img_global = model.get_image_features(pixel_values=inputs.pixel_values)
    img_global = l2norm(img_global, dim=-1).squeeze(0)        # [D]
    return img_global, img_patch


@torch.no_grad()
def clip_text_global_and_tokens(model: CLIPModel, processor: CLIPProcessor, text: str, device: str):
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    t_out = model.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    tok = model.text_projection(t_out.last_hidden_state)     
    valid_len = int(inputs.attention_mask.sum().item())
    if valid_len > 2:
        tok = tok[:, 1:valid_len - 1, :]
    else:
        tok = tok[:, :valid_len, :]
    tok = l2norm(tok, dim=-1).squeeze(0)                    

    txt_global = model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    txt_global = l2norm(txt_global, dim=-1).squeeze(0)
    return txt_global, tok

def load_coco_cf_json(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    if isinstance(blob, dict) and "data" in blob:
        data = blob["data"]
        meta = blob.get("meta", {})
        src = meta.get("gen_captions", None)
        if isinstance(src, str) and src:
            source_name = os.path.splitext(os.path.basename(src))[0]
        else:
            source_name = os.path.splitext(os.path.basename(path))[0]
    else:
        data = blob
        source_name = os.path.splitext(os.path.basename(path))[0]

    samples = []
    for ex in data:
        image_id = ex.get("image_id", None)
        image_path = ex.get("image_path", None)
        cap = ex.get("caption", None)
        neg = ex.get("negative_caption", ex.get("neg_caption", None))
        if image_id is None or image_path is None or cap is None or neg is None:
            continue
        samples.append({
            "image_id": int(image_id),
            "image_path": str(image_path),
            "caption": str(cap),
            "negative_caption": str(neg),
        })
    return source_name, samples

@dataclass
class ImageCacheEntry:
    img_global: torch.Tensor   
    mu: torch.Tensor          
    weights: torch.Tensor     


@torch.no_grad()
def build_scores(
    data_by_source: Dict[str, List[Dict[str, Any]]],
    judge_model: str,
    device: str,
    seed: int,
    cache_npz: Optional[str] = None,
    max_samples_per_source: int = -1,
) -> Dict[str, Any]:

    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    kappa = KAPPA
    vmf_max_iter = VMF_MAX_ITER

    model = CLIPModel.from_pretrained(judge_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(judge_model)

    all_sources = sorted(list(data_by_source.keys()))
    src_to_id = {s: i for i, s in enumerate(all_sources)}

    groups: Dict[int, Dict[str, Any]] = {}
    total = 0
    for src in all_sources:
        xs = data_by_source[src]
        if max_samples_per_source > 0:
            xs = xs[:max_samples_per_source]
        for ex in xs:
            img_id = int(ex["image_id"])
            total += 1
            if img_id not in groups:
                groups[img_id] = {"image_path": ex["image_path"], "items": []}
            groups[img_id]["items"].append((src, ex))

    image_ids = []
    source_ids = []
    cos_pos = []
    cos_neg = []
    div_pos = []
    div_neg = []

    failed = 0
    done = 0


    img_id_list = sorted(groups.keys())

    for img_id in img_id_list:
        img_path = groups[img_id]["image_path"]
        items = groups[img_id]["items"] 

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            failed += len(items)
            continue

        try:
            img_g, img_patch = clip_image_global_and_patches(model, processor, image, device)
        except Exception:
            failed += len(items)
            continue

        if img_patch.numel() == 0:
            failed += len(items)
            continue

        K_img_eff = max(1, min(K_IMG, img_patch.shape[0]))
        seed_img = seed + (img_id % 1000003) + K_img_eff * 97
        vmf_img = VMFMixtureFixedKappa(
            n_components=K_img_eff, kappa=kappa, max_iter=vmf_max_iter, seed=seed_img, device=device
        ).fit(img_patch)
        img_mu_fixed = vmf_img.mu.detach()
        img_w_fixed = vmf_img.weights.detach()

        for src, ex in items:
            cap_p = ex["caption"]
            cap_n = ex["negative_caption"]

            try:
                tg_p, tok_p = clip_text_global_and_tokens(model, processor, cap_p, device)
                tg_n, tok_n = clip_text_global_and_tokens(model, processor, cap_n, device)
            except Exception:
                failed += 1
                continue

            if tok_p.numel() == 0 or tok_n.numel() == 0:
                failed += 1
                continue

            # cosine
            cpos = float((img_g @ tg_p).item())
            cneg = float((img_g @ tg_n).item())

            seed_pair = seed + (done * 3)
            dpos = local_cost_bi_kl_from_imgpatch(
                img_patch=img_patch,
                img_mu=img_mu_fixed,
                img_w=img_w_fixed,
                txt_tok=tok_p,
                kappa=kappa,
                vmf_max_iter=vmf_max_iter,
                seed=seed_pair,
                device=device,
            )
            dneg = local_cost_bi_kl_from_imgpatch(
                img_patch=img_patch,
                img_mu=img_mu_fixed,
                img_w=img_w_fixed,
                txt_tok=tok_n,
                kappa=kappa,
                vmf_max_iter=vmf_max_iter,
                seed=seed_pair,
                device=device,
            )

            image_ids.append(img_id)
            source_ids.append(src_to_id[src])
            cos_pos.append(cpos)
            cos_neg.append(cneg)
            div_pos.append(dpos)
            div_neg.append(dneg)

            done += 1
            if (done % 500) == 0:
                print(f"[extract] done {done} / {total} (failed={failed})")

    arr = {
        "image_ids": np.array(image_ids, dtype=np.int64),
        "source_ids": np.array(source_ids, dtype=np.int32),
        "source_names": np.array(all_sources),
        "cos_pos": np.array(cos_pos, dtype=np.float32),
        "cos_neg": np.array(cos_neg, dtype=np.float32),
        "div_pos": np.array(div_pos, dtype=np.float32),
        "div_neg": np.array(div_neg, dtype=np.float32),
        "meta": {
            "judge_model": judge_model,
            "device": device,
            "K_img": K_IMG,
            "K_txt": K_TXT,
            "kappa": kappa,
            "vmf_max_iter": vmf_max_iter,
            "alpha_msd": ALPHA_MSD,
            "alpha_soft": ALPHA_SOFT,
            "tau_soft": TAU_SOFT,
            "seed": seed,
            "valid": int(len(image_ids)),
            "failed": int(failed),
            "total": int(total),
        }
    }

    if cache_npz:
        meta_str = json.dumps(arr["meta"], ensure_ascii=False)
        np.savez_compressed(
            cache_npz,
            image_ids=arr["image_ids"],
            source_ids=arr["source_ids"],
            source_names=arr["source_names"],
            cos_pos=arr["cos_pos"],
            cos_neg=arr["cos_neg"],
            div_pos=arr["div_pos"],
            div_neg=arr["div_neg"],
            meta=meta_str
        )
        print(f"[cache] saved scores to {cache_npz} (valid={arr['meta']['valid']})")

    return arr



def load_scores_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return {
        "image_ids": z["image_ids"].astype(np.int64),
        "source_ids": z["source_ids"].astype(np.int32),
        "source_names": z["source_names"],
        "cos_pos": z["cos_pos"].astype(np.float32),
        "cos_neg": z["cos_neg"].astype(np.float32),
        "div_pos": z["div_pos"].astype(np.float32),
        "div_neg": z["div_neg"].astype(np.float32),
        "meta": meta,
    }


def accuracy_from_deltas(delta: np.ndarray) -> float:
    return float(np.mean(delta > 0.0))


def eval_metrics(
    mask: np.ndarray,
    d_cos: np.ndarray,
    d_div: np.ndarray,
    alpha_msd: float,
    alpha_soft: float,
    tau_soft: float,
) -> Dict[str, float]:
    out = {}
    d_cos_m = d_cos[mask]
    d_div_m = d_div[mask]

    out["cos"] = accuracy_from_deltas(d_cos_m)
    out["local"] = float(np.mean(d_div_m < 0.0))

    d_msd = d_cos_m - alpha_msd * d_div_m
    out["msd"] = accuracy_from_deltas(d_msd)

    u = soft_gate_u_from_cos_delta(d_cos_m, tau=tau_soft)
    d_soft = d_cos_m - alpha_soft * u * d_div_m
    out["soft"] = accuracy_from_deltas(d_soft)
    return out


def bucket_analysis(
    is_test: np.ndarray,
    source_ids: np.ndarray,
    source_names: List[str],
    cos_pos: np.ndarray,
    cos_neg: np.ndarray,
    div_pos: np.ndarray,
    div_neg: np.ndarray,
    alpha_msd: float,
    alpha_soft: float,
    tau_soft: float,
    n_bins: int = 5,
    by_source: bool = False,
    plot_path: Optional[str] = None,
):
    d_cos = (cos_pos - cos_neg)
    d_div = (div_pos - div_neg)

    def summarize(mask: np.ndarray, title: str):
        d_cos_m = d_cos[mask]
        d_div_m = d_div[mask]
        conf = np.abs(d_cos_m)

        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(conf, qs)

        edges = np.unique(edges)
        if len(edges) < 3:
            print(f"[bucket] {title}: not enough unique edges (all margins similar). skip.")
            return
        rows = []
        for i in range(len(edges) - 1):
            lo = edges[i]
            hi = edges[i + 1]
            if i < len(edges) - 2:
                bmask = (conf >= lo) & (conf < hi)
            else:
                bmask = (conf >= lo) & (conf <= hi)

            if bmask.sum() == 0:
                continue

            dc = d_cos_m[bmask]
            dd = d_div_m[bmask]
            u = soft_gate_u_from_cos_delta(dc, tau=tau_soft)

            acc_cos = float(np.mean(dc > 0))
            acc_local = float(np.mean(dd < 0))
            acc_msd = float(np.mean((dc - alpha_msd * dd) > 0))
            acc_soft = float(np.mean((dc - alpha_soft * u * dd) > 0))

            rows.append({
                "bin": i,
                "range": (float(lo), float(hi)),
                "count": int(bmask.sum()),
                "mean_abs_margin": float(conf[bmask].mean()),
                "mean_u": float(u.mean()),
                "cos": acc_cos,
                "local": acc_local,
                "msd": acc_msd,
                "soft": acc_soft,
                "soft_minus_cos": float(acc_soft - acc_cos)
            })

        print(f"\n=== Bucket analysis on TEST: {title} (quantile bins on |cos_pos-cos_neg|) ===")
        print("bin | count | mean|delta_cos| | mean_u | Cos | Local | MSD | Soft | Soft-Cos | range")
        for r in rows:
            lo, hi = r["range"]
            print(f"{r['bin']:>3d} | {r['count']:>5d} | {r['mean_abs_margin']:.6f} | {r['mean_u']:.3f} | "
                  f"{r['cos']*100:6.2f}% | {r['local']*100:6.2f}% | {r['msd']*100:6.2f}% | {r['soft']*100:6.2f}% | "
                  f"{r['soft_minus_cos']*100:7.2f}% | [{lo:.6f}, {hi:.6f}]")

        if plot_path:
            try:
                import matplotlib.pyplot as plt
                xs = [r["mean_abs_margin"] for r in rows]
                y_cos = [r["cos"] for r in rows]
                y_soft = [r["soft"] for r in rows]
                y_msd = [r["msd"] for r in rows]

                plt.figure()
                plt.plot(xs, y_cos, marker="o", label="Cosine")
                plt.plot(xs, y_msd, marker="o", label="MSD")
                plt.plot(xs, y_soft, marker="o", label="Soft-MSD")
                plt.xlabel("Mean |cos_pos - cos_neg| in bin")
                plt.ylabel("Accuracy")
                plt.title(f"Accuracy vs Cosine-Confidence (TEST) - {title}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_path, dpi=200)
                print(f"[plot] saved to {plot_path}")
            except Exception as e:
                print(f"[plot] failed: {e}")

    if not by_source:
        summarize(is_test, title="ALL")
    else:
        for sid, name in enumerate(source_names):
            m = is_test & (source_ids == sid)
            if m.sum() == 0:
                continue
            summarize(m, title=name)


# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", type=str, action="append", required=True,
                    help="repeatable: coco_cf_*.json (llava/qwen/internvl/gpt...)")
    ap.add_argument("--judge_model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    # tuning config
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_npz", type=str, default="coco_cf_scores.npz")
    ap.add_argument("--recompute", action="store_true", help="ignore cache_npz and recompute scores")

    # bucket config
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--bucket_by_source", action="store_true")
    ap.add_argument("--plot_path", type=str, default=None)

    ap.add_argument("--max_samples_per_source", type=int, default=-1, help="debug only")
    args = ap.parse_args()

    set_seed(args.seed)

    # load data
    data_by_source = {}
    for p in args.data_json:
        src, xs = load_coco_cf_json(p)
        data_by_source[src] = xs
        print(f"[load] {src}: {len(xs)} samples")

    # load or compute scores
    if (not args.recompute) and args.cache_npz and os.path.exists(args.cache_npz):
        scores = load_scores_npz(args.cache_npz)
        print(f"[cache] loaded from {args.cache_npz}, valid={len(scores['image_ids'])}")
        print(f"[cache-meta] {scores['meta']}")
    else:
        scores = build_scores(
            data_by_source=data_by_source,
            judge_model=args.judge_model,
            device=args.device,
            seed=args.seed,
            cache_npz=args.cache_npz,
            max_samples_per_source=args.max_samples_per_source,
        )

    source_names = [str(x) for x in scores["source_names"].tolist()]

    d_cos = scores["cos_pos"] - scores["cos_neg"]
    d_div = scores["div_pos"] - scores["div_neg"]
    is_test = np.ones_like(scores["image_ids"], dtype=bool)

    test_acc = eval_metrics(is_test, d_cos, d_div, ALPHA_MSD, ALPHA_SOFT, TAU_SOFT)

    print("\n================ Step A-1: Fixed hyperparams on ALL DATA ================")
    print(f"MSD alpha           : {ALPHA_MSD:g}")
    print(f"Soft-MSD alpha,tau  : alpha={ALPHA_SOFT:g}, tau={TAU_SOFT:g}")
    print(f"ALL  Acc: Cos={test_acc['cos']:.4%} | Local={test_acc['local']:.4%} | "
          f"MSD={test_acc['msd']:.4%} | Soft={test_acc['soft']:.4%}")

    print("\n[Per-source TEST breakdown]")
    for sid, name in enumerate(source_names):
        m = is_test & (scores["source_ids"] == sid)
        if m.sum() == 0:
            continue
        per = eval_metrics(m, d_cos, d_div, ALPHA_MSD, ALPHA_SOFT, TAU_SOFT)
        print(f"- {name:>12s} (n={int(m.sum())}): "
              f"Cos={per['cos']:.2%} | Local={per['local']:.2%} | MSD={per['msd']:.2%} | Soft={per['soft']:.2%}")

    print("\n================ Step A-2: Cosine-margin Bucket Analysis (ALL) ================")
    bucket_analysis(
        is_test=is_test,
        source_ids=scores["source_ids"],
        source_names=source_names,
        cos_pos=scores["cos_pos"],
        cos_neg=scores["cos_neg"],
        div_pos=scores["div_pos"],
        div_neg=scores["div_neg"],
        alpha_msd=ALPHA_MSD,
        alpha_soft=ALPHA_SOFT,
        tau_soft=TAU_SOFT,
        n_bins=args.n_bins,
        by_source=args.bucket_by_source,
        plot_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
