import torch.nn as nn
import torch


def init_weights(m: nn.Module, nonlinearity: str) -> None:
    """
    Initialize the weights of a given module.

    Parameters
    ----------
    m : nn.Module
        The PyTorch module whose weights need to be initialized.
    nonlinearity : str
        The type of nonlinearity used in the module. Supported values are
        'relu', 'elu', 'leaky_relu', and others. Determines the initialization
        strategy.
    """
    if isinstance(m, nn.Linear):
        if nonlinearity in ["relu", "elu"]:
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "leaky_relu":
            torch.nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.01
            )
        else:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
