# shrink_gmm.py
import math
import numpy as np
import torch


class ShrinkGMM:

    def __init__(
        self,
        n_components: int,
        alpha: float = 0.5,
        sigma2: float = None,
        max_iter: int = 20,
        device: str = None,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.sigma2 = sigma2
        self.max_iter = max_iter
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

      
        self.means_ = None       
        self.cov_diag_ = None    
        self.weights_ = None      

    def fit(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        N, D = X.shape
        K = self.n_components
        if self.sigma2 is None:
            self.sigma2 = X.var(dim=0, unbiased=False).mean().item()
        idx = torch.randperm(N, device=self.device)[:K]
        self.means_ = X[idx].clone()           
        self.cov_diag_ = torch.full(
            (K, D), self.sigma2, device=self.device
        )                                     
        self.weights_ = torch.full((K,), 1.0 / K, device=self.device) 

        eps = 1e-6
        for _ in range(self.max_iter):
            log_prob = self._estimate_log_prob(X)             
            log_prob = log_prob + torch.log(self.weights_ + eps)
            log_prob_norm = torch.logsumexp(log_prob, dim=1, keepdim=True)
            log_resp = log_prob - log_prob_norm
            resp = log_resp.exp()                            

            Nk = resp.sum(0) + eps                             
            self.weights_ = Nk / N                           

            self.means_ = (resp.T @ X) / Nk[:, None]            

           
            diff = X.unsqueeze(1) - self.means_.unsqueeze(0)    
            sq = (resp.unsqueeze(2) * diff ** 2).sum(0) / Nk[:, None]  

            self.cov_diag_ = self.alpha * sq + (1.0 - self.alpha) * self.sigma2
            self.cov_diag_ = torch.clamp(self.cov_diag_, min=1e-6)

        return self

    def _estimate_log_prob(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        N, D = X.shape
        diff = X.unsqueeze(1) - self.means_.unsqueeze(0)       
        var = self.cov_diag_.unsqueeze(0)                       
        mahal = (diff ** 2 / var).sum(dim=2)                   
        log_det = torch.log(var).sum(dim=2).squeeze(0)        
        log_prob = -0.5 * (D * math.log(2 * math.pi) + log_det + mahal)
        return log_prob                                       

    def score_samples(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        eps = 1e-6
        log_prob = self._estimate_log_prob(X)                 
        log_prob = log_prob + torch.log(self.weights_ + eps)
        result = torch.logsumexp(log_prob, dim=1)              
        return result.detach().cpu().numpy()

    def sample(self, n_samples: int):
        D = self.means_.shape[1]
        weights = self.weights_
        comp_ids = torch.multinomial(weights, n_samples, replacement=True)
        means = self.means_[comp_ids]                          
        var = self.cov_diag_[comp_ids]                         
        z = torch.randn(n_samples, D, device=self.device)
        samples = means + torch.sqrt(var) * z
        return samples.detach().cpu().numpy(), comp_ids.detach().cpu().numpy()
