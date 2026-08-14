from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

activation_functions = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


class NormalInvGamma(nn.Module):
    """
    Neural network layer that outputs parameters of a Normal-Inverse-Gamma distribution.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_units : int
        Number of output units.
    """

    def __init__(self, in_features: int, out_units: int):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def evidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute positive evidence from input using softplus function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Positive evidence.
        """
        return torch.clamp(F.softplus(x), min=1e-6, max=1e3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass producing the parameters of the Normal-Inverse-Gamma distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Tuple containing (mu, v, alpha, beta) parameters.
        """
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta


class FFNN(nn.Module):
    """
    Fully connected feedforward neural network with optional batch normalization, dropout,
    and Normal-Inverse-Gamma output layer.

    Parameters
    ----------
    in_features : int
        Number of input features.
    n_layers : int
        Number of hidden layers.
    act_fn_name : str
        Name of activation function to use. Must be one of 'relu', 'elu', 'leaky_relu'.
    num_neu_list : List[int]
        List specifying the number of neurons in each hidden layer.
    p : float
        Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        n_layers: int,
        act_fn_name: str,
        num_neu_list: List[int],
        p: float,
    ) -> None:
        super(FFNN, self).__init__()
        if act_fn_name not in activation_functions:
            raise ValueError(f"Activation function '{act_fn_name}' not supported.")
        self.in_features = in_features
        self.n_layers = n_layers
        self.act_fn_class = activation_functions[act_fn_name]
        self.act_fn_name = act_fn_name
        self.num_neu_list = num_neu_list
        self.p = p
        if len(num_neu_list) != n_layers:
            raise ValueError("Length of num_neu_list must equal n_layers.")
        layers = []
        current_in = in_features
        for i in range(n_layers):
            outf = num_neu_list[i]
            layers.append(nn.Linear(current_in, outf))
            layers.append(nn.BatchNorm1d(outf))
            layers.append(self.act_fn_class())
            layers.append(nn.Dropout(p))
            current_in = outf
        layers.append(NormalInvGamma(current_in, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Output of the Normal-Inverse-Gamma layer.
        """
        return self.model(x)

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns
        -------
        int
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
