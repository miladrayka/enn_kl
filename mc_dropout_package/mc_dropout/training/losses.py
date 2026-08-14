import torch


def heteroscedastic_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Gaussian Negative Log-Likelihood (NLL) for heteroscedastic regression.

    Parameters
    ----------
    mu : torch.Tensor
        Predicted mean of the Gaussian distribution.
    log_var : torch.Tensor
        Predicted log variance of the Gaussian distribution.
    y : torch.Tensor
        Ground truth target values.

    Returns
    -------
    torch.Tensor
        Mean negative log-likelihood over all samples.
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    
    precision = torch.exp(-log_var)
    # L = 1/2 * e^(-s) * (y - mu)^2 + 1/2 * s
    loss = 0.5 * precision * (y - mu) ** 2 + 0.5 * log_var
    
    return loss.mean()