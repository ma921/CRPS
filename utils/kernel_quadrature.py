import torch
from utils._rchq import recombination
from fast_pytorch_kmeans import KMeans
from gpytorch.kernels import ScaleKernel, MaternKernel


def indicator_kernel(x,y):
    x = x.view(-1).unsqueeze(-1)
    y = y.view(-1).unsqueeze(0)

    indicator1 = (y < x).to(x.dtype)
    indicator2 = (x < y).to(x.dtype)
    return 0.5 * (x * indicator1 + y * indicator2)

def kq_compress(samples, n_low_rank, n_compressed):
    kmeans = KMeans(n_clusters=n_low_rank, mode='euclidean')
    kmeans.fit(samples.view(-1).unsqueeze(-1))
    samples_low_rank = kmeans.centroids
    
    idx, w_kq = recombination(
        samples,              # random samples for recombination
        samples_low_rank,     # number of samples used for approximating kernel with Nyström method
        n_compressed,         # number of samples finally returned
        indicator_kernel,     # kernel
        torch.device("cpu"),  # device
    )
    samples_kq = samples[idx]
    return w_kq, samples_kq

def error_terms_kq_non_comp(y_pred, y_true, decompose=False):
    M = len(y_pred)
    abs_err_est = (y_pred - y_true).abs().mean()
    smpl_mean_est = y_pred.mean()
    K = indicator_kernel(y_pred, y_pred)
    unbias_correction = M / (M - 1)
    cdf_term_est = unbias_correction * K.mean()
    if decompose:
        return abs_err_est, smpl_mean_est, cdf_term_est
    else:
        return abs_err_est + smpl_mean_est -2* cdf_term_est

def error_terms_kq_comp(y_pred, y_true, n_low_rank=200, n_compressed=100, decompose=False):
    w_kq, samples_kq = kq_compress(y_pred, n_low_rank, n_compressed)
    abs_err_est = w_kq @ (samples_kq - y_true).abs()
    smpl_mean_est = w_kq @ samples_kq
    unbias_correction = n_compressed / (n_compressed - 1)
    cdf_term_est = unbias_correction * w_kq @ indicator_kernel(samples_kq, samples_kq) @ w_kq
    if decompose:
        return abs_err_est, smpl_mean_est, cdf_term_est
    else:
        return abs_err_est + smpl_mean_est -2* cdf_term_est


