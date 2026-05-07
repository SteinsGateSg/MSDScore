# eval_coco_cf.py
import argparse
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

K_IMG = 3
K_TXT = 2
KAPPA = 20.0
VMF_MAX_ITER = 20
ALPHA_SOFT = 0.1
TAU_SOFT = 0.2
L0 = 20.0
TAU_L = 3.0

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

class VMFMixtureFixedKappa:
    def __init__(self, n_components: int, kappa: float = 20.0, max_iter: int = 20, seed: int = 0, device: str = "cuda"):
        self.n_components = int(n_components)
        self.kappa = float(kappa)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.mu = None       # [K,D]
        self.weights = None  # [K]

    def fit(self, x: torch.Tensor):
        x = x.to(self.device)
        x = l2norm(x, dim=1)
        N, D = x.shape
        K = min(self.n_components, N)

        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)
        idx = torch.randperm(N, generator=g, device=self.device)[:K] if N >= K else torch.randint(0, N, (K,), generator=g, device=self.device)
        mu = l2norm(x[idx].clone(), dim=1)
        w = torch.full((K,), 1.0 / K, device=self.device)

        for _ in range(self.max_iter):
            logits = self.kappa * (x @ mu.T) + torch.log(w.unsqueeze(0) + 1e-9)
            post = F.softmax(logits, dim=1)
            Nk = post.sum(dim=0) + 1e-8
            mu = l2norm(post.T @ x, dim=1)
            w = Nk / float(N)

        self.mu, self.weights = mu, w
        return self

    def score_samples(self, x: torch.Tensor) -> torch.Tensor:
        assert self.mu is not None and self.weights is not None
        x = x.to(self.device)
        x = l2norm(x, dim=1)
        logits = self.kappa * (x @ self.mu.T) + torch.log(self.weights.unsqueeze(0) + 1e-9)
        return torch.logsumexp(logits, dim=1)  # [N]

# ---------------- Bi-KL ----------------
@torch.no_grad()
def local_cost_bi_kl(img_patch: torch.Tensor, txt_tok: torch.Tensor,
                     vmf_img: VMFMixtureFixedKappa,
                     kappa: float, max_iter: int,
                     seed: int, device: str) -> float:
    vmf_txt = VMFMixtureFixedKappa(K_TXT, kappa=kappa, max_iter=max_iter, seed=seed + 123, device=device).fit(txt_tok)

    log_p_img_x = vmf_img.score_samples(img_patch)
    log_q_txt_x = vmf_txt.score_samples(img_patch)
    kl_i2t = (log_p_img_x - log_q_txt_x).mean()

    log_q_txt_y = vmf_txt.score_samples(txt_tok)
    log_p_img_y = vmf_img.score_samples(txt_tok)
    kl_t2i = (log_q_txt_y - log_p_img_y).mean()

    L = txt_tok.shape[0]
    exponent = (L - L0) / TAU_L
    beta = 1.0 / (1.0 + np.exp(exponent))
    bi_kl = beta * kl_i2t + (1.0 - beta) * kl_t2i
    return float(bi_kl.item())

def soft_msd_from_pair(cos_pos: float, cos_neg: float,
                       div_pos: float, div_neg: float,
                       alpha: float, tau: float) -> Tuple[float, float]:
    cos = torch.tensor([cos_pos, cos_neg], dtype=torch.float32)
    div = torch.tensor([div_pos, div_neg], dtype=torch.float32)
    p = F.softmax(cos / max(tau, 1e-6), dim=0)
    u = (2.0 / 1.0) * (1.0 - p.max())
    u = torch.clamp(u, 0.0, 1.0)
    final = cos - alpha * u * div
    return float(final[0].item()), float(final[1].item())

# ---------------- CLIP feature extraction ----------------
@torch.no_grad()
def get_image_feats(model: CLIPModel, processor: CLIPProcessor, image: Image.Image, device: str):
    inputs = processor(images=image, return_tensors="pt").to(device)
    v_out = model.vision_model(inputs.pixel_values)
    patch_tokens = v_out.last_hidden_state[:, 1:, :]             # [1, Np, D]
    img_patch = model.visual_projection(patch_tokens)             # [1, Np, D]
    img_patch = l2norm(img_patch, dim=-1).squeeze(0)              # [Np, D]
    img_global = model.get_image_features(pixel_values=inputs.pixel_values)
    img_global = l2norm(img_global, dim=-1).squeeze(0)            # [D]
    return img_patch, img_global

@torch.no_grad()
def get_text_feats(model: CLIPModel, processor: CLIPProcessor, text: str, device: str):
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    t_out = model.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    tok = model.text_projection(t_out.last_hidden_state)          # [1, L, D]
    valid_len = int(inputs.attention_mask.sum().item())
    if valid_len > 2:
        tok = tok[:, 1:valid_len - 1, :]
    else:
        tok = tok[:, :valid_len, :]
    tok = l2norm(tok, dim=-1).squeeze(0)                          # [Nt, D]
    txt_global = model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    txt_global = l2norm(txt_global, dim=-1).squeeze(0)            # [D]
    return tok, txt_global

def cosine(img_global: torch.Tensor, txt_global: torch.Tensor) -> float:
    return float((img_global @ txt_global).item())

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", type=str, required=True, help="COCO-CF json from build_coco_cf.py")
    ap.add_argument("--judge_model", type=str, required=True, help="CLIP model id or local path")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    with open(args.data_json, "r", encoding="utf-8") as f:
        blob = json.load(f)
    data = blob["data"] if isinstance(blob, dict) and "data" in blob else blob

    if args.max_samples > 0:
        data = data[: args.max_samples]

    print(f"[Load] {len(data)} samples from {args.data_json}")
    print(f"[Judge] {args.judge_model} on {device}")

    model = CLIPModel.from_pretrained(args.judge_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.judge_model)
# inputs
    correct_cos = 0
    correct_soft = 0
    n = 0

    for i, ex in enumerate(data):
        img_path = ex["image_path"]
        cap_pos = ex["caption"]
        cap_neg = ex["negative_caption"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        img_patch, img_global = get_image_feats(model, processor, image, device)

        pos_tok, pos_global = get_text_feats(model, processor, cap_pos, device)
        neg_tok, neg_global = get_text_feats(model, processor, cap_neg, device)

        K_img_eff = max(1, min(K_IMG, img_patch.shape[0]))
        vmf_img = VMFMixtureFixedKappa(K_img_eff, kappa=KAPPA, max_iter=VMF_MAX_ITER,
                                       seed=args.seed + i * 7, device=device).fit(img_patch)

        cos_pos = cosine(img_global, pos_global)
        cos_neg = cosine(img_global, neg_global)

        div_pos = local_cost_bi_kl(img_patch, pos_tok, vmf_img, KAPPA, VMF_MAX_ITER, seed=args.seed + i * 3, device=device)
        div_neg = local_cost_bi_kl(img_patch, neg_tok, vmf_img, KAPPA, VMF_MAX_ITER, seed=args.seed + i * 3, device=device)

        # Soft-MSD
        soft_pos, soft_neg = soft_msd_from_pair(cos_pos, cos_neg, div_pos, div_neg, alpha=ALPHA_SOFT, tau=TAU_SOFT)
        correct_soft += int(soft_pos > soft_neg)

        # cosine baseline
        correct_cos += int(cos_pos > cos_neg)

        n += 1
        if n % 200 == 0:
            print(f"[{n:5d}] Cos={correct_cos/n:.2%} | Soft-MSD={correct_soft/n:.2%}")

    print("-" * 60)
    print(f"Done. valid={n}")
    print(f"Cosine Acc : {correct_cos/n:.2%}")
    print(f"Soft-MSD   : {correct_soft/n:.2%}")

if __name__ == "__main__":
    main()
