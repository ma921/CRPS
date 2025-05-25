import torch
import numpy as np
from utils.cdf import iECDF, ECDF
from utils.kernel_quadrature import error_terms_kq_non_comp, error_terms_kq_comp


def exact_crps(mu, sigma, y_true):
    """
    Closed-form CRPS per observation for Gaussian distribution
    pred: predictive posterior distribution of GP
    y_true: the ground truth y
    """
    # prepare samples
    z = (mu - y_true) / sigma
    norm = torch.distributions.Normal(0, 1)
    
    # first error term
    absolute_error_true = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.log_prob(z).exp())
    
    # third error term
    cdf_true = sigma / np.sqrt(np.pi)
    
    # total error
    crps = absolute_error_true - cdf_true
    return crps

def single_empirical_crps_function_sample(y_pred, y_true, quantiles):
    """
    Quantile-based CRPS estimation for function sample
    y_pred: function samples from GP
    y_true: the ground truth y
    quantiles: the grid of quantiles
    """
    quantile_losses = {}
    for kappa in quantiles:
        q = iECDF(y_pred, kappa)
        indicator = (y_true <= q).to(y_pred.dtype)
        quantile_losses[kappa] = (
            (kappa - indicator) * (y_true - q)
        ).abs().sum()
    approximated_crps = np.mean([
        (2 * Lambda).item() for Lambda in quantile_losses.values()
    ])
    return approximated_crps

def pwm_function_sample(y_pred, y_true):
    """
    Quantile-based CRPS estimation for function samples
    y_pred: function samples from GP
    y_true: the ground truth y
    """
    M = len(y_pred)
    absolute_error = (y_pred - y_true).abs().mean()
    
    # correction term
    y_pred = torch.sort(y_pred).values
    b0 = y_pred.mean()
    b1_values = y_pred * torch.arange(M) / M
    b1 = b1_values.mean()
    pwm = b0 - 2 * b1
    per_obs_crps = absolute_error + pwm
    return per_obs_crps

def crps_loop_function(pred, Y_test, M=100, quantiles=None, method="quantile", compress=False):
    function_samples = pred.sample(torch.Size([M]))
    if method == "quantile":
        approximated_crps = torch.tensor([
            single_empirical_crps_function_sample(
                function_samples[:,i,:].squeeze(), 
                Y_test[i], 
                quantiles, 
            ) for i in range(function_samples.shape[1])
        ])
    elif method == "pwm":
        approximated_crps = torch.tensor([
            pwm_function_sample(
                function_samples[:,i,:].squeeze(), 
                Y_test[i],
            ) for i in range(function_samples.shape[1])
        ])
    elif method == "kern_quad":
        if compress:
            approximated_crps = torch.tensor([
                error_terms_kq_comp(
                    function_samples[:,i,:].squeeze(),
                    Y_test[i],
                    n_low_rank=100, 
                    n_compressed=50,
                ) for i in range(function_samples.shape[1])
            ])
        else:
            approximated_crps = torch.tensor([
                error_terms_kq_non_comp(
                    function_samples[:,i,:].squeeze(),
                    Y_test[i],
                ) for i in range(function_samples.shape[1])
            ])
    elif method == "exact":
        approximated_crps = torch.tensor([
            exact_crps(
                pred.mean.detach()[i],
                pred.stddev.detach()[i],
                Y_test[i],
            ) for i in range(function_samples.shape[1])
        ])
    else:
        NotImplementedError(f"Not implemented: {method}")
    return approximated_crps
