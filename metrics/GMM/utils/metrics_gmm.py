# metrics_gmm.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment



def symmetric_kl_mc(p, q, n_samples: int = 2000) -> float:
    Xp, _ = p.sample(n_samples)
    Xq, _ = q.sample(n_samples)

    # KL(P||Q)
    log_p_xp = p.score_samples(Xp)
    log_q_xp = q.score_samples(Xp)
    kl_pq = float(np.mean(log_p_xp - log_q_xp))

    # KL(Q||P)
    log_q_xq = q.score_samples(Xq)
    log_p_xq = p.score_samples(Xq)
    kl_qp = float(np.mean(log_q_xq - log_p_xq))

    return 0.5 * (kl_pq + kl_qp)


def _clip_logs(log_p, log_q, clip_percentile: float = 0.5, clip_min: float = None):
    log_p = np.asarray(log_p)
    log_q = np.asarray(log_q)

    if clip_min is not None:
        log_p = np.maximum(log_p, clip_min)
        log_q = np.maximum(log_q, clip_min)

    if clip_percentile is not None and 0.0 < clip_percentile < 100.0:
        all_vals = np.concatenate([log_p, log_q], axis=0)
        low = np.percentile(all_vals, clip_percentile)
        log_p = np.maximum(log_p, low)
        log_q = np.maximum(log_q, low)

    return log_p, log_q


def symmetric_kl_mc_clipped(
    p,
    q,
    n_samples: int = 2000,
    clip_percentile: float = 0.5,
    clip_min: float = None,
) -> float:
    Xp, _ = p.sample(n_samples)
    Xq, _ = q.sample(n_samples)

    # KL(P||Q)
    lp_xp = p.score_samples(Xp)
    lq_xp = q.score_samples(Xp)
    lp_xp_c, lq_xp_c = _clip_logs(lp_xp, lq_xp,
                                  clip_percentile=clip_percentile,
                                  clip_min=clip_min)
    kl_pq = float(np.mean(lp_xp_c - lq_xp_c))

    # KL(Q||P)
    lq_xq = q.score_samples(Xq)
    lp_xq = p.score_samples(Xq)
    lq_xq_c, lp_xq_c = _clip_logs(lq_xq, lp_xq,
                                  clip_percentile=clip_percentile,
                                  clip_min=clip_min)
    kl_qp = float(np.mean(lq_xq_c - lp_xq_c))

    return 0.5 * (kl_pq + kl_qp)



def sliced_wasserstein2(
    p,
    q,
    n_samples: int = 2048,
    n_projections: int = 64,
    seed: int = 0,
) -> float:
    Xp, _ = p.sample(n_samples)
    Xq, _ = q.sample(n_samples)
    Xp = np.asarray(Xp)
    Xq = np.asarray(Xq)

    assert Xp.shape[1] == Xq.shape[1], "p and q dimensions do not match"
    d = Xp.shape[1]

    rng = np.random.default_rng(seed)
    sw2 = 0.0

    for _ in range(n_projections):
        u = rng.normal(size=(d,))
        u /= np.linalg.norm(u) + 1e-8

        proj_p = Xp @ u
        proj_q = Xq @ u

        proj_p.sort()
        proj_q.sort()

        w2_1d = np.mean((proj_p - proj_q) ** 2)
        sw2 += w2_1d

    sw2 /= float(n_projections)
    return float(sw2)


class _SimpleMLP(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)   # logits
        return x


def classifier_divergence(
    p,
    q,
    n_samples_per_dist: int = 2048,
    hidden: int = 64,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = None,
) -> float:
    Xp, _ = p.sample(n_samples_per_dist)
    Xq, _ = q.sample(n_samples_per_dist)

    Xp = np.asarray(Xp, dtype=np.float32)
    Xq = np.asarray(Xq, dtype=np.float32)

    X = np.concatenate([Xp, Xq], axis=0)
    y = np.concatenate([
        np.zeros(n_samples_per_dist, dtype=np.float32),
        np.ones(n_samples_per_dist, dtype=np.float32),
    ], axis=0)

    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]

    n_total = X.shape[0]
    n_train = int(0.8 * n_total)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    model = _SimpleMLP(dim_in=X.shape[1], hidden=hidden).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    n_batches = max(1, n_train // batch_size)
    for _ in range(epochs):
        perm = torch.randperm(n_train, device=device)
        for i in range(n_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]

            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)

            optim.zero_grad()
            loss.backward()
            optim.step()

    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t)
        loss_test = F.binary_cross_entropy_with_logits(logits_test, y_test_t)
    return float(loss_test.item())



def kl_gaussian_diag(mu0, cov0, mu1, cov1):
    mu0 = np.asarray(mu0)
    mu1 = np.asarray(mu1)
    cov0 = np.asarray(cov0)
    cov1 = np.asarray(cov1)

    D = mu0.shape[-1]
    cov0 = np.maximum(cov0, 1e-8)
    cov1 = np.maximum(cov1, 1e-8)

    log_det_ratio = np.log(cov1).sum() - np.log(cov0).sum()
    trace_term = (cov0 / cov1).sum()
    diff = mu1 - mu0
    quad_term = (diff ** 2 / cov1).sum()
    return 0.5 * (log_det_ratio - D + trace_term + quad_term)


def symmetric_kl_gaussian_diag(mu0, cov0, mu1, cov1):
    return 0.5 * (
        kl_gaussian_diag(mu0, cov0, mu1, cov1)
        + kl_gaussian_diag(mu1, cov1, mu0, cov0)
    )


def wasserstein2_gaussian_diag(mu0, cov0, mu1, cov1):
    mu0 = np.asarray(mu0)
    mu1 = np.asarray(mu1)
    cov0 = np.asarray(cov0)
    cov1 = np.asarray(cov1)

    diff_mu = mu0 - mu1
    diff_sigma = np.sqrt(np.maximum(cov0, 1e-8)) - np.sqrt(np.maximum(cov1, 1e-8))
    return float((diff_mu ** 2).sum() + (diff_sigma ** 2).sum())


def match_components_by_means(means_p, means_q, metric: str = "cosine"):
    means_p = np.asarray(means_p)  # [Kp, D]
    means_q = np.asarray(means_q)  # [Kq, D]
    assert means_p.shape[1] == means_q.shape[1]
    Kp, D = means_p.shape
    Kq, _ = means_q.shape

    if metric == "cosine":
        mp = means_p / (np.linalg.norm(means_p, axis=1, keepdims=True) + 1e-8)
        mq = means_q / (np.linalg.norm(means_q, axis=1, keepdims=True) + 1e-8)
        cost = 1.0 - mp @ mq.T   # [Kp, Kq]
    elif metric == "l2":
        diff = means_p[:, None, :] - means_q[None, :, :]
        cost = (diff ** 2).sum(axis=-1)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind, cost

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def local_component_divergence(
    gmm_p,
    gmm_q,
    mode: str = "skl",         
    match_metric: str = "cosine",
    weight_mode: str = "avg",  
):

    means_p = _to_numpy(gmm_p.means_)    
    means_q = _to_numpy(gmm_q.means_)     
    cov_p   = _to_numpy(gmm_p.cov_diag_)  
    cov_q   = _to_numpy(gmm_q.cov_diag_)   
    w_p     = _to_numpy(gmm_p.weights_)   
    w_q     = _to_numpy(gmm_q.weights_) 


    Kp, D = means_p.shape
    Kq, D2 = means_q.shape
    assert D == D2

    row_ind, col_ind, cost_mat = match_components_by_means(
        means_p, means_q, metric=match_metric
    )
    K = min(Kp, Kq)  

    total = 0.0
    weight_sum = 0.0

    for idx in range(K):
        i = row_ind[idx]
        j = col_ind[idx]

        if mode == "skl":
            div_ij = symmetric_kl_gaussian_diag(
                means_p[i], cov_p[i], means_q[j], cov_q[j]
            )
        elif mode == "w2":
            div_ij = wasserstein2_gaussian_diag(
                means_p[i], cov_p[i], means_q[j], cov_q[j]
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if weight_mode == "avg":
            w = 0.5 * (w_p[i] + w_q[j])
        elif weight_mode == "min":
            w = float(min(w_p[i], w_q[j]))
        elif weight_mode == "none":
            w = 1.0
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

        total += w * div_ij
        weight_sum += w

    if weight_sum < 1e-8:
        return float(total)
    return float(total / weight_sum)
