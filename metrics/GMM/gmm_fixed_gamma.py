# gmm_fixed_gamma.py
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import random
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-large-patch14"

print(f"Loading model: {model_id} ...")
clip_model = CLIPModel.from_pretrained(model_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_id)

class GaussianFixedVarMixture:
    def __init__(self, n_components=3, n_features=768, sigma=0.1, max_iter=10):
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
            denominator = posterior.sum(dim=0, keepdim=True).T + 1e-6 # [K, 1]
            
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

def get_features(image, text):
    img_input = image.resize((224, 224), Image.BICUBIC)
    inputs = clip_processor(images=img_input, return_tensors="pt").to(device)
    with torch.no_grad():
        v_out = clip_model.vision_model(inputs.pixel_values)
        patch_tokens = v_out.last_hidden_state[:, 1:, :]

        img_emb = clip_model.visual_projection(patch_tokens)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        
        img_global = clip_model.get_image_features(pixel_values=inputs.pixel_values)
        img_global = img_global / img_global.norm(dim=-1, keepdim=True)

    inputs = clip_processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        t_out = clip_model.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        txt_emb = clip_model.text_projection(t_out.last_hidden_state)
        valid_len = inputs.attention_mask.sum().item()
        if valid_len > 2:
            txt_emb = txt_emb[:, 1:valid_len-1, :]
        else:
            txt_emb = txt_emb[:, :valid_len, :]
            
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        
        txt_global = clip_model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        txt_global = txt_global / txt_global.norm(dim=-1, keepdim=True)

    return {
        "img_patch": img_emb.squeeze(0).cpu().numpy(),
        "txt_token": txt_emb.squeeze(0).cpu().numpy(),
    }


def calc_scores(feats):
    img_data = torch.tensor(feats["img_patch"]).to(device)
    txt_data = torch.tensor(feats["txt_token"]).to(device)
    
    n_tokens = txt_data.shape[0]
    k_dynamic = 1 if n_tokens <= 3 else (2 if n_tokens <= 6 else (3 if n_tokens <= 10 else 4))
    
    gaussian_img = GaussianFixedVarMixture(n_components=k_dynamic, sigma=0.3).to(device)
    gaussian_img.fit(img_data)
    
    gaussian_txt = GaussianFixedVarMixture(n_components=k_dynamic, sigma=0.3).to(device)
    gaussian_txt.fit(txt_data)
    
    log_p_img = gaussian_img.score_samples(img_data)
    log_q_txt = gaussian_txt.score_samples(img_data)
    kl_i2t = torch.mean(log_p_img - log_q_txt)
    
    log_p_txt = gaussian_txt.score_samples(txt_data)
    log_q_img = gaussian_img.score_samples(txt_data)
    kl_t2i = torch.mean(log_p_txt - log_q_img)
    
    kl_score = (kl_i2t + kl_t2i) / 2
    
    return float(kl_score.item())



def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    image = Image.open("cat_dog.png").convert("RGB")
    texts = "a black cat and a white dog"
    feats = get_features(image, texts)
    res = calc_scores(feats)
    print(res)
        

if __name__ == "__main__":
    setup_seed()
    main()
