# gmm_fixed_gamma.py
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = os.environ.get("CLIP_MODEL_ID", "openai/clip-vit-large-patch14")
json_path = os.environ.get("SUGARCREPE_JSON", "data/SugarCrepe/replace_rel.json")
base_image_dir = os.environ.get("COCO_VAL2017_DIR", "data/images/val2017")

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

parsed_data = []

for item in data.values():
    filename = item['filename']
    full_image_path = os.path.join(base_image_dir, filename)
    
    caption = item['caption']
    
    neg_caption = item['negative_caption']
    
    parsed_data.append({
        "image_path": full_image_path,
        "caption": caption,
        "neg_caption": neg_caption
    })

print(f"Loaded {len(parsed_data)} data")
print("the first:", parsed_data[0])                            


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


print("Loading model...")
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)


class GaussianFixedVarMixture:
    def __init__(self, n_components=3, n_features=768, sigma=0.1, max_iter=20):
        self.n_components = n_components
        self.n_features = n_features
        self.sigma = sigma
        self.gamma = 1.0 / (2 * sigma**2) 
        self.max_iter = max_iter
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.mu = None 
        self.weights = None

    def to(self, device):
        self.device = device
        if self.mu is not None:
            self.mu = self.mu.to(device)
        if self.weights is not None:
            self.weights = self.weights.to(device)
        return self

    def fit(self, x):
        x = x / x.norm(dim=1, keepdim=True)
        x = x.to(self.device)
        
        N, D = x.shape
        
        indices = torch.randperm(N)[:self.n_components]
        if N < self.n_components:
            indices = torch.randint(0, N, (self.n_components,))
        
        self.mu = x[indices].clone().to(self.device) 
        
        for _ in range(self.max_iter):
            x_sq = torch.sum(x**2, dim=1, keepdim=True) 
            mu_sq = torch.sum(self.mu**2, dim=1).unsqueeze(0)
            interaction = x @ self.mu.T
            
            dist_sq = x_sq + mu_sq - 2 * interaction
            
            logits = -self.gamma * dist_sq
            posterior = F.softmax(logits, dim=1) 
            
            numerator = posterior.T @ x 
            denominator = posterior.sum(dim=0, keepdim=True).T + 1e-6 
            
            self.mu = numerator / denominator
            
            self.weights = posterior.mean(dim=0)

    def score_samples(self, x):
        x = x / x.norm(dim=1, keepdim=True)
        x = x.to(self.device)
        
        x_sq = torch.sum(x**2, dim=1, keepdim=True)
        mu_sq = torch.sum(self.mu**2, dim=1).unsqueeze(0)
        interaction = x @ self.mu.T
        dist_sq = x_sq + mu_sq - 2 * interaction
        
        exp_term = -self.gamma * dist_sq
        
        if self.weights is None:
            log_weights = torch.log(torch.full((self.n_components,), 1.0/self.n_components, device=self.device))
        else:
            log_weights = torch.log(self.weights + 1e-9)
            
        log_prob = torch.logsumexp(log_weights + exp_term, dim=1)
        
        return log_prob

def get_image_features(image):
    img_input = image.resize((224, 224), Image.BICUBIC)
    inputs = processor(images=img_input, return_tensors="pt").to(device)
    with torch.no_grad():
        v_out = model.vision_model(inputs.pixel_values)
        patch_tokens = v_out.last_hidden_state[:, 1:, :]              
        img_patch = model.visual_projection(patch_tokens)             
        img_patch = img_patch / img_patch.norm(dim=-1, keepdim=True)

        img_global = model.get_image_features(pixel_values=inputs.pixel_values)
        img_global = img_global / img_global.norm(dim=-1, keepdim=True)

    return {
        "img_patch": img_patch.squeeze(0).cpu().numpy(),
        "img_global": img_global.cpu().numpy(),
    }


def get_text_features(text):
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        t_out = model.text_model(input_ids=inputs.input_ids,
                                 attention_mask=inputs.attention_mask)
        txt_token = model.text_projection(t_out.last_hidden_state)  
        valid_len = inputs.attention_mask.sum().item()
        if valid_len > 2:
            txt_token = txt_token[:, 1:valid_len-1, :]   
        else:
            txt_token = txt_token[:, :valid_len, :]
        txt_token = txt_token / txt_token.norm(dim=-1, keepdim=True)

        txt_global = model.get_text_features(input_ids=inputs.input_ids,
                                             attention_mask=inputs.attention_mask)
        txt_global = txt_global / txt_global.norm(dim=-1, keepdim=True)

    return {
        "txt_token": txt_token.squeeze(0).cpu().numpy(),
        "txt_global": txt_global.cpu().numpy(),
    }
def calc_cosine(image_feats, text_feats):
    img_global = torch.tensor(image_feats["img_global"]).to(device)
    txt_global = torch.tensor(text_feats["txt_global"]).to(device)
    return (img_global @ txt_global.T).item()


def calc_kl(image_feats, text_feats):
    img_patch = image_feats["img_patch"] 
    txt_token = text_feats["txt_token"]   
    img_data = torch.tensor(img_patch).to(device)
    txt_data = torch.tensor(txt_token).to(device)

    n_tokens = txt_data.shape[0]
    k = 1 if n_tokens <= 3 else (2 if n_tokens <= 6 else (3 if n_tokens <= 10 else 4))
    k = min(k, img_data.shape[0], txt_data.shape[0])

    vmf_img = GaussianFixedVarMixture(n_components=k).to(device)
    vmf_img.fit(img_data)
    vmf_txt = GaussianFixedVarMixture(n_components=k).to(device)
    vmf_txt.fit(txt_data)

    log_p_img = vmf_img.score_samples(img_data)
    log_q_txt = vmf_txt.score_samples(img_data)
    kl_i2t = torch.mean(log_p_img - log_q_txt)

    log_p_txt = vmf_txt.score_samples(txt_data)
    log_q_img = vmf_img.score_samples(txt_data)
    kl_t2i = torch.mean(log_p_txt - log_q_img)

    return ((kl_i2t)).item()


def soft_gated_msd_scores(cos_scores, div_scores, alpha=0.002, tau=0.02):
    assert cos_scores.ndim == 1 and div_scores.ndim == 1
    assert cos_scores.numel() == div_scores.numel()

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
    
    print("-" * 60)

    correct_cosine = 0
    correct_kl = 0
    correct_final = 0
    correct_rank = 0
    correct_softmsd = 0
    valid_count = 0  

  
    for i, item in enumerate(test_data):
        img_path = item['image_path']
        caption_pos = item['caption']
        caption_neg = item['neg_caption']

      
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Error] {img_path}: {e}")
            continue

        img_feats = get_image_features(image)
        
        pos_txt_feats = get_text_features(caption_pos)
        neg_txt_feats = get_text_features(caption_neg)

        cos_score_pos = calc_cosine(img_feats, pos_txt_feats)
        cos_score_neg = calc_cosine(img_feats, neg_txt_feats)

        kl_dist_pos, vmf_img = calc_kl(img_feats, pos_txt_feats, device=device, vmf_img=None)
        kl_dist_neg, _       = calc_kl(img_feats, neg_txt_feats, device=device, vmf_img=vmf_img)


        alpha = 0.01
        final_score_pos = cos_score_pos - alpha * kl_dist_pos
        final_score_neg = cos_score_neg - alpha * kl_dist_neg
        
        lim = 0.005
        if abs(cos_score_pos - cos_score_neg) > lim:
            correct_rank += int(cos_score_pos > cos_score_neg)
        else:
            correct_rank += int(kl_dist_pos < kl_dist_neg)

        alpha = 0.2
        tau = 0.2  
      
        cos_vec = torch.tensor([cos_score_pos, cos_score_neg], device=device, dtype=torch.float32)
        div_vec = torch.tensor([kl_dist_pos,  kl_dist_neg],  device=device, dtype=torch.float32)

        final_vec, u, p = soft_gated_msd_scores(cos_vec, div_vec, alpha=alpha, tau=tau)

        softmsd_pos = float(final_vec[0].item())
        softmsd_neg = float(final_vec[1].item())


        if cos_score_pos > cos_score_neg:
            correct_cosine += 1
            
        if kl_dist_pos < kl_dist_neg:
            correct_kl += 1

        if final_score_pos > final_score_neg:
            correct_final += 1

        if softmsd_pos > softmsd_neg : correct_softmsd += 1 

        valid_count += 1

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(test_data)} | "
                  f"Cos Acc: {correct_cosine/(i+1):.2%} | "
                  f"KL Acc: {correct_kl/(i+1):.2%} | "
                  f"Rank Agg Acc: {correct_rank/(i+1):.2%} | "
                  f"Soft-MSD Acc: {correct_softmsd/(i+1):.2%} | "
                  f"MSD Acc: {correct_final/(i+1):.2%}")

    print("-" * 60)
    if valid_count > 0:
        acc_cosine = correct_cosine / valid_count
        acc_kl = correct_kl / valid_count
        acc_msd = correct_final / valid_count
        acc_softmsd = correct_softmsd / valid_count
        acc_rank = correct_rank / valid_count
        print(f"Baseline (Global Cosine) Accuracy : {acc_cosine:.2%}")
        print(f"(vMF-KL Mixture) Accuracy    : {acc_kl:.2%}")
        print(f"Rank Aggreation    : {acc_rank:.2%}")
        print(f"(MSD) Accuracy    : {acc_msd:.2%}")
        print(f"(Soft-MSD) Accuracy    : {acc_softmsd:.2%}")
    else:
        print("ERROR")

if __name__ == "__main__":
    main()
