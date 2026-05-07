# gmm_sklearn_diag.py
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.mixture import GaussianMixture
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-large-patch14"

print(f"Loading model: {model_id} ...")
clip_model = CLIPModel.from_pretrained(model_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_id)

def get_features(image, text):
    # Resize
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
        "img_global": img_global.cpu().numpy(),
        "txt_global": txt_global.cpu().numpy()
    }

def calc(feats):
    
    n_tokens = feats["txt_token"].shape[0]
    
    if n_tokens <= 3: k_dynamic = 1
    elif n_tokens <= 6:k_dynamic = 2
    elif n_tokens <= 10:k_dynamic = 3
    else:k_dynamic = 4
        
    
   
    gmm_img = GaussianMixture(n_components=k_dynamic, covariance_type='diag', random_state=42)
    gmm_img.fit(feats["img_patch"])
        
    gmm_txt = GaussianMixture(n_components=k_dynamic, covariance_type='diag', random_state=42)
    gmm_txt.fit(feats["txt_token"])
        
    def get_kl(p, q, n=2000):
        X, _ = p.sample(n)
        log_p = p.score_samples(X)
        log_q = q.score_samples(X)
        return np.mean(log_p - log_q)
            
    kl_score = (get_kl(gmm_img, gmm_txt) + get_kl(gmm_txt, gmm_img)) / 2
        
    
    return float(kl_score)


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
    score = calc(feats)
    print(score)



if __name__ == "__main__":
    setup_seed()
    main()