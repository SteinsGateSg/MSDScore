import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, SiglipModel

device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
model_id = os.environ.get("SIGLIP_MODEL_ID", "google/siglip-so400m-patch14-384")
json_path = os.environ.get("SUGARCREPE_JSON", "data/SugarCrepe/replace_rel.json")
base_image_dir = os.environ.get("COCO_VAL2017_DIR", "data/images/val2017")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

parsed_data = []
for item in data.values():
    filename = item["filename"]
    full_image_path = os.path.join(base_image_dir, filename)
    caption = item["caption"]
    neg_caption = item["negative_caption"]
    parsed_data.append(
        {
            "image_path": full_image_path,
            "caption": caption,
            "neg_caption": neg_caption,
        }
    )

print(f"Loaded {len(parsed_data)} samples")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"Loading SigLIP model from {model_id} ...")
model = SiglipModel.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)


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
        if self.device is None: self.device = x.device
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

        kappa = self.fixed_kappa

        for _ in range(self.max_iter):
            dot = x @ self.mu.T
            log_w = torch.log(self.weights.clamp_min(self.eps))
            logits = kappa * dot + log_w.unsqueeze(0)
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
        logits = log_w.unsqueeze(0) + self.fixed_kappa * dot
        return torch.logsumexp(logits, dim=1)


def get_image_features(image):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        img_global = model.get_image_features(pixel_values=inputs.pixel_values)
        img_global = img_global / img_global.norm(dim=-1, keepdim=True)

        vision_outputs = model.vision_model(pixel_values=inputs.pixel_values)
        img_patch = vision_outputs.last_hidden_state
        img_patch = img_patch / img_patch.norm(dim=-1, keepdim=True)

    return {
        "img_patch": img_patch.squeeze(0).float().cpu().numpy(),
        "img_global": img_global.float().cpu().numpy(),
    }


def get_text_features(text):
    inputs = processor.tokenizer(
        text=text,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    with torch.no_grad():
        txt_global = model.get_text_features(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        txt_global = txt_global / txt_global.norm(dim=-1, keepdim=True)

        text_outputs = model.text_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        txt_token = text_outputs.last_hidden_state

        valid_len = inputs.attention_mask.sum().item()
        txt_token = txt_token[:, :valid_len, :]
        txt_token = txt_token / txt_token.norm(dim=-1, keepdim=True)

    return {
        "txt_token": txt_token.squeeze(0).float().cpu().numpy(),
        "txt_global": txt_global.float().cpu().numpy(),
    }


def calc_cosine(image_feats, text_feats):
    img_global = torch.tensor(image_feats["img_global"]).to(device)
    txt_global = torch.tensor(text_feats["txt_global"]).to(device)
    return (img_global @ txt_global.T).item()


def calc_bi_kl(image_feats, text_feats, device,
               vmf_img: VMFMixture = None,
               n_img_components=4, n_txt_components=2,
               kappa=20.0, max_iter=20,
               L0=20.0, tau_L=3.0):

    img_patch = image_feats["img_patch"]
    txt_token = text_feats["txt_token"]

    img_data = torch.tensor(img_patch).to(device=device, dtype=torch.float32)
    txt_data = torch.tensor(txt_token).to(device=device, dtype=torch.float32)

    if vmf_img is None:
        vmf_img = VMFMixture(n_components=n_img_components, kappa=kappa, max_iter=max_iter).to(device)
        vmf_img.fit(img_data)

    vmf_txt = VMFMixture(n_components=n_txt_components, kappa=kappa, max_iter=max_iter).to(device)
    vmf_txt.fit(txt_data)

    log_p_x = vmf_img.score_samples(img_data)
    log_q_x = vmf_txt.score_samples(img_data)
    kl_i2t = (log_p_x - log_q_x).mean()

    log_q_y = vmf_txt.score_samples(txt_data)
    log_p_y = vmf_img.score_samples(txt_data)
    kl_t2i = (log_q_y - log_p_y).mean()

    L = txt_data.shape[0]
    exponent = (L - L0) / tau_L
    beta = 1.0 / (1.0 + np.exp(exponent))

    bi_kl = beta * kl_i2t + (1.0 - beta) * kl_t2i

    return float(bi_kl.item()), vmf_img, float(kl_i2t.item()), float(kl_t2i.item())


def soft_gated_msd_scores(cos_scores, div_scores, alpha=0.1, tau=0.2):
    M = cos_scores.numel()
    p = F.softmax(cos_scores / tau, dim=0)

    if M <= 1:
        u = cos_scores.new_tensor(1.0)
    else:
        u = (M / (M - 1.0)) * (1.0 - p.max())
        u = torch.clamp(u, 0.0, 1.0)

    final = cos_scores - alpha * u * div_scores
    return final, u, p


def main():
    test_data = parsed_data
    print(f"\nProcessing {len(test_data)} samples with Bi-KL...")
    print("-" * 60)

    correct_cosine = 0
    correct_bikl = 0
    correct_msd = 0
    correct_softmsd = 0
    valid_count = 0

    for i, item in enumerate(test_data):
        img_path = item["image_path"]
        caption_pos = item["caption"]
        caption_neg = item["neg_caption"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        img_feats = get_image_features(image)
        pos_txt_feats = get_text_features(caption_pos)
        neg_txt_feats = get_text_features(caption_neg)

        cos_pos = calc_cosine(img_feats, pos_txt_feats)
        cos_neg = calc_cosine(img_feats, neg_txt_feats)

        bikl_pos, vmf_img, _, _ = calc_bi_kl(img_feats, pos_txt_feats, device=device, vmf_img=None)
        bikl_neg, _, _, _       = calc_bi_kl(img_feats, neg_txt_feats, device=device, vmf_img=vmf_img)

        alpha_fixed = 0.1
        msd_pos = cos_pos - alpha_fixed * bikl_pos
        msd_neg = cos_neg - alpha_fixed * bikl_neg

        cos_vec = torch.tensor([cos_pos, cos_neg], device=device)
        div_vec = torch.tensor([bikl_pos, bikl_neg], device=device)

        soft_vec, _, _ = soft_gated_msd_scores(cos_vec, div_vec, alpha=0.1, tau=0.2)
        softmsd_pos = float(soft_vec[0].item())
        softmsd_neg = float(soft_vec[1].item())

        if cos_pos > cos_neg: correct_cosine += 1
        if bikl_pos < bikl_neg: correct_bikl += 1
        if msd_pos > msd_neg: correct_msd += 1
        if softmsd_pos > softmsd_neg: correct_softmsd += 1

        valid_count += 1

        if (i + 1) % 10 == 0:
            print(f"Prog: {i + 1} | "
                  f"Cos: {correct_cosine/(i+1):.2%} | "
                  f"Bi-KL: {correct_bikl/(i+1):.2%} | "
                  f"MSD: {correct_msd/(i+1):.2%} | "
                  f"Soft-MSD: {correct_softmsd/(i+1):.2%}")

    print("-" * 60)
    if valid_count > 0:
        print(f"Final Results (N={valid_count}):")
        print(f"Global Cosine Accuracy : {correct_cosine / valid_count:.2%}")
        print(f"Bi-KL Divergence Acc   : {correct_bikl / valid_count:.2%}")
        print(f"MSD Score Accuracy     : {correct_msd / valid_count:.2%}")
        print(f"Soft-MSD Accuracy      : {correct_softmsd / valid_count:.2%}")

if __name__ == "__main__":
    main()
