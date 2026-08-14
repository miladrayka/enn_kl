import torch


def nig_nll(
    gamma: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the Negative Log-Likelihood (NLL) for a Normal-Inverse-Gamma distribution.

    Parameters
    ----------
    gamma : torch.Tensor
        Predicted mean of the Normal distribution.
    v : torch.Tensor
        Predicted variance scaling parameter.
    alpha : torch.Tensor
        Predicted shape parameter of the inverse gamma.
    beta : torch.Tensor
        Predicted scale parameter of the inverse gamma.
    y : torch.Tensor
        Ground truth target values.
    eps : float, optional
        Small constant for numerical stability, by default 1e-6.

    Returns
    -------
    torch.Tensor
        Mean negative log-likelihood over all samples.
    """
    v = v.clamp(min=eps)
    alpha = alpha.clamp(min=1.0 + eps)
    beta = beta.clamp(min=eps)
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    two_beta_lambda = 2 * beta * (1 + v)
    sq_error = v * (y - gamma) ** 2
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (sq_error + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()


def nig_reg(
    gamma: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    _beta: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the regularization term for evidential regression.

    Parameters
    ----------
    gamma : torch.Tensor
        Predicted mean of the Normal distribution.
    v : torch.Tensor
        Predicted variance scaling parameter.
    alpha : torch.Tensor
        Predicted shape parameter of the inverse gamma.
    _beta : torch.Tensor
        Predicted scale parameter of the inverse gamma (unused in this function).
    y : torch.Tensor
        Ground truth target values.

    Returns
    -------
    torch.Tensor
        Mean regularization term over all samples.
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()


def evidential_regression(
    dist_params, y: torch.Tensor, lamb: float = 0.2
) -> torch.Tensor:
    """
    Compute the evidential regression loss combining NLL and regularization.

    Parameters
    ----------
    dist_params : tuple of torch.Tensor
        Tuple containing (gamma, v, alpha, beta) predicted by the model.
    y : torch.Tensor
        Ground truth target values.
    lamb : float, optional
        Regularization coefficient, by default 0.2.

    Returns
    -------
    torch.Tensor
        Combined evidential regression loss.
    """
    return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)
