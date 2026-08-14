from typing import List, Tuple
import torch
import torch.nn as nn

activation_functions = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


class FFNN(nn.Module):
    """
    Fully connected feedforward neural network with dropout for MC-Dropout
    uncertainty quantification.

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
        Must have length equal to `n_layers`.
    p : float
        Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        in_features: int,
        n_layers: int,
        act_fn_name: str,
        num_neu_list: List[int],
        p: float,
    ) -> None:
        super().__init__()
        if act_fn_name not in activation_functions:
            raise ValueError(f"Activation function '{act_fn_name}' not supported.")
        if len(num_neu_list) != n_layers:
            raise ValueError("Length of num_neu_list must equal n_layers.")

        self.in_features = in_features
        self.n_layers = n_layers
        self.act_fn_name = act_fn_name
        self.act_fn_class = activation_functions[act_fn_name]
        self.num_neu_list = num_neu_list
        self.p = p

        layers = []
        current_in = in_features
        for out_features in num_neu_list:
            layers.append(nn.Linear(current_in, out_features))
            layers.append(self.act_fn_class())
            layers.append(nn.Dropout(p))
            current_in = out_features
        layers.append(nn.Linear(current_in, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, 1).
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

    def enable_dropout(self) -> None:
        """
        Set only Dropout layers to train mode.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def mc_predict(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Run MC-Dropout inference by collecting stochastic forward passes.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_features).
        n_samples : int, optional
            Number of stochastic forward passes to perform. Default is 100.

        Returns
        -------
        torch.Tensor
            Stacked predictions of shape (n_samples, N, 1).
        """
        self.eval()
        self.enable_dropout()
        return torch.stack([self.model(x) for _ in range(n_samples)], dim=0)
