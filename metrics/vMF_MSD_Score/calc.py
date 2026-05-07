import random
import numpy as np
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


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

        # Initialization
        if N >= K:
            idx = torch.randperm(N, device=self.device)[:K]
        else:
            idx = torch.randint(0, N, (K,), device=self.device)
        self.mu = self._l2norm(x[idx].clone())
        self.weights = torch.full((K,), 1.0 / K, device=self.device)

        kappa = self.fixed_kappa

        for _ in range(self.max_iter):
            # E-step
            dot = x @ self.mu.T                                 
            log_w = torch.log(self.weights.clamp_min(self.eps)) 
            logits = kappa * dot + log_w.unsqueeze(0)           
            posterior = F.softmax(logits, dim=1)               

            # M-step
            N_k = posterior.sum(dim=0)
            self.weights = (N_k / N).clamp_min(self.eps)
            self.weights = self.weights / self.weights.sum()

            weighted_sum = posterior.T @ x                      
            
            # Re-init dead components
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

    L = txt_data.shape[0] # Caption Length
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
