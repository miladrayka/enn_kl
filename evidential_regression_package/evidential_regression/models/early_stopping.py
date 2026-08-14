import torch.nn as nn


class EarlyStopper:
    """
    Early stopping utility to prevent overfitting during model training.

    Parameters
    ----------
    patience : int, optional
        Number of epochs with no improvement after which training is stopped (default is 15).
    delta : float, optional
        Minimum change in the monitored metric to qualify as an improvement (default is 0.0).
    minimize : bool, optional
        Whether to minimize (True) or maximize (False) the monitored metric (default is True).

    Attributes
    ----------
    patience : int
        Number of epochs to wait for improvement.
    delta : float
        Minimum improvement threshold.
    minimize : bool
        Direction of optimization (True for minimizing, False for maximizing).
    counter : int
        Counter tracking epochs without improvement.
    best_metric : float
        Best metric value observed so far.
    best_epoch : int
        Epoch number with the best metric.
    best_model_state : dict
        State dictionary of the model at the best epoch.
    early_stop_triggered : bool
        Indicates whether early stopping has been triggered.
    """

    def __init__(
        self,
        patience: int = 15,
        delta: float = 0.0,
        minimize: bool = True,
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.minimize = minimize
        self.counter = 0
        self.best_metric = float("inf") if minimize else float("-inf")
        self.best_epoch = -1
        self.best_model_state = None
        self.early_stop_triggered = False

    def early_stop(self, current_metric: float, model: nn.Module, epoch: int) -> bool:
        """
        Determine whether training should stop early based on performance metric.

        Parameters
        ----------
        current_metric : float
            The current value of the monitored metric.
        model : nn.Module
            The model being trained.
        epoch : int
            The current training epoch.

        Returns
        -------
        bool
            True if early stopping is triggered, otherwise False.
        """
        if self.minimize:
            if current_metric < self.best_metric - self.delta:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.best_model_state = model.state_dict()
                self.counter = 0
            else:
                self.counter += 1
        else:
            if current_metric > self.best_metric + self.delta:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.best_model_state = model.state_dict()
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop_triggered = True
            return True
        return False
