from typing import List, Tuple
import torch
import torch.nn as nn

activation_functions = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


class GaussianOutput(nn.Module):
    """
    Neural network layer that outputs mean and log-variance of a Gaussian distribution
    for Heteroscedastic Aleatoric Uncertainty.

    Parameters
    ----------
    in_features : int
        Number of input features.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.dense = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing the mean and log-variance parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing (mu, log_var).
        """
        out = self.dense(x)
        mu, log_var = torch.split(out, 1, dim=-1)
        return mu, log_var


class FFNN(nn.Module):
    """
    Fully connected feedforward neural network with dropout and a Gaussian output layer.

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
        layers.append(GaussianOutput(current_in))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output containing (mu, log_var).
        """
        return self.model(x)

    def enable_dropout(self):
        """
        Enable dropout layers during evaluation to execute Monte-Carlo Dropout.
        """
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns
        -------
        int
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)