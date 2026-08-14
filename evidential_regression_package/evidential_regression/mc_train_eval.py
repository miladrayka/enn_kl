from typing import Dict, Callable, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from evidential_regression.training.metrics import cal_metrics
from evidential_regression.models.early_stopping import EarlyStopper


def train_and_val_proc(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: Callable,
    epochs: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader_dict: Dict[str, DataLoader],
    early_stopper: EarlyStopper,
    device: torch.device,
    path_to_save: Optional[Path],
    target_inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[float, Optional[Path]]:
    """
    Train and validate a PyTorch MC-Dropout model with early stopping and metric tracking.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained and validated.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    criterion : Callable
        Loss function used for training (e.g. nn.MSELoss()).
    epochs : int
        Number of training epochs.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    dataloader_dict : Dict[str, DataLoader]
        Dictionary containing DataLoaders for 'train' and 'val' datasets.
    early_stopper : EarlyStopper
        Early stopping utility to halt training if validation metric does not improve.
    device : torch.device
        Device on which computations are performed (CPU or GPU).
    path_to_save : Path or None
        Path to save models and training results. If None, results are not saved.
    target_inverse_transform : callable, optional
        Function to inverse-transform target values and predictions, by default None.

    Returns
    -------
    best_metric : float
        Best validation RMSE achieved during training.
    best_model_save_path : Path or None
        Path where the best model was saved, or None if not saved.
    """
    results_dict = defaultdict(list)
    model_save_dir = None
    best_model_save_path = None
    if path_to_save:
        model_save_dir = path_to_save / "models"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        best_model_save_path = model_save_dir / "ffnn_model_best.pth"

    for epoch in range(epochs):

        model.train()
        train_losses = []
        y_train_all = []
        y_train_pred_all = []

        for X_train, y_train in dataloader_dict["train"]:
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train)
            train_loss = criterion(y_train_pred, y_train)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            mu = y_train_pred.squeeze().detach().cpu().numpy()
            train_losses.append(train_loss.item())
            y_train_all.append(y_train.detach().cpu().numpy())
            y_train_pred_all.append(mu)

        epoch_train_loss = np.mean(train_losses) if train_losses else 0.0
        y_train_all_np = np.concatenate(y_train_all).flatten()
        y_train_pred_all_np = np.concatenate(y_train_pred_all).flatten()

        if target_inverse_transform:
            y_train_all_np = target_inverse_transform(y_train_all_np)
            y_train_pred_all_np = target_inverse_transform(y_train_pred_all_np)

        train_r2, train_mse, train_rmse, train_mae, train_rp, train_mape = cal_metrics(
            y_train_all_np, y_train_pred_all_np
        )
        results_dict["Train_Loss"].append(epoch_train_loss)
        results_dict["Train_R2"].append(train_r2)
        results_dict["Train_MSE"].append(train_mse)
        results_dict["Train_RMSE"].append(train_rmse)
        results_dict["Train_MAE"].append(train_mae)
        results_dict["Train_RP"].append(train_rp)
        results_dict["Train_MAPE"].append(train_mape)

        model.eval()
        val_losses = []
        y_val_all = []
        y_val_pred_all = []

        with torch.no_grad():
            for X_val, y_val in dataloader_dict["val"]:
                X_val, y_val = X_val.to(device), y_val.to(device)

                y_val_pred = model(X_val)
                val_loss = criterion(y_val_pred, y_val)
                mu = y_val_pred.squeeze().detach().cpu().numpy()

                val_losses.append(val_loss.item())
                y_val_all.append(y_val.detach().cpu().numpy())
                y_val_pred_all.append(mu)

        epoch_val_loss = np.mean(val_losses) if val_losses else 0.0
        y_val_all_np = np.concatenate(y_val_all).flatten()
        y_val_pred_all_np = np.concatenate(y_val_pred_all).flatten()

        if target_inverse_transform:
            y_val_all_np = target_inverse_transform(y_val_all_np)
            y_val_pred_all_np = target_inverse_transform(y_val_pred_all_np)

        val_r2, val_mse, val_rmse, val_mae, val_rp, val_mape = cal_metrics(
            y_val_all_np, y_val_pred_all_np
        )
        results_dict["Val_Loss"].append(epoch_val_loss)
        results_dict["Val_R2"].append(val_r2)
        results_dict["Val_MSE"].append(val_mse)
        results_dict["Val_RMSE"].append(val_rmse)
        results_dict["Val_MAE"].append(val_mae)
        results_dict["Val_RP"].append(val_rp)
        results_dict["Val_MAPE"].append(val_mape)

        print(f"\n Epoch {epoch + 1}/{epochs}")
        print(
            f"Train Metrics-> R2: {train_r2:.3f}, RP: {train_rp:.3f}, MAPE: {train_mape:.3f}, "
            f"RMSE: {train_rmse:.3f}, MSE: {train_mse:.3f}, Loss: {epoch_train_loss:.3f}"
        )
        print(
            f"Val Metrics-> R2: {val_r2:.3f}, RP: {val_rp:.3f}, MAPE: {val_mape:.3f}, "
            f"RMSE: {val_rmse:.3f}, MSE: {val_mse:.3f}, Loss: {epoch_val_loss:.3f}"
        )
        print("-----------------------------------------------------")

        if early_stopper.early_stop(val_rmse, model, epoch):
            print("Training is stopped because of early stopping.")
            if best_model_save_path and early_stopper.best_model_state:
                torch.save(early_stopper.best_model_state, best_model_save_path)
                print(
                    f"Best model saved at epoch {early_stopper.best_epoch + 1} "
                    f"with RMSE: {early_stopper.best_metric:.3f}"
                )
            else:
                print(
                    f"Best model state recorded at epoch {early_stopper.best_epoch + 1} "
                    f"with RMSE: {early_stopper.best_metric:.3f} "
                    f"(no model saved as path_to_save was None)."
                )
            break

        scheduler.step(val_rmse)

    results_df = pd.DataFrame.from_dict(results_dict)
    if path_to_save:
        results_df_path = path_to_save / "results_ffnn_model_train_and_val.csv"
        results_df.to_csv(results_df_path, index=False)
    else:
        print("Training results DataFrame not saved as path_to_save was None.")

    if not early_stopper.early_stop_triggered:
        if best_model_save_path and early_stopper.best_model_state:
            torch.save(early_stopper.best_model_state, best_model_save_path)
            print(
                f"Training finished. Best model saved at epoch {early_stopper.best_epoch + 1} "
                f"with RMSE: {early_stopper.best_metric:.3f}"
            )
        else:
            print(
                f"Training finished. Best validation RMSE: {early_stopper.best_metric:.3f} "
                f"at epoch {early_stopper.best_epoch + 1} "
                f"(no model saved as path_to_save was None)."
            )

    return early_stopper.best_metric, best_model_save_path


def _mc_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run MC-Dropout inference over a full dataloader and compute uncertainty estimates.

    Epistemic uncertainty is estimated as the variance of the sample means across
    MC forward passes. Aleatoric uncertainty is approximated as the difference
    between total variance and epistemic variance. Total uncertainty is their sum.

    Parameters
    ----------
    model : nn.Module
        Trained MC-Dropout model exposing ``enable_dropout()`` and ``mc_predict()``.
    dataloader : DataLoader
        DataLoader yielding ``(X, y)`` batches.
    device : torch.device
        Device on which computations are performed (CPU or GPU).
    n_samples : int
        Number of stochastic forward passes per batch.

    Returns
    -------
    y_true : np.ndarray
        Ground-truth targets, shape (N,).
    y_pred : np.ndarray
        Predictive mean across MC samples, shape (N,).
    aleatoric_unc : np.ndarray
        Aleatoric uncertainty (total - epistemic), shape (N,).
    epistemic_unc : np.ndarray
        Epistemic uncertainty (variance of sample means), shape (N,).
    total_unc : np.ndarray
        Total predictive variance, shape (N,).
    """
    model.eval()
    model.enable_dropout()
    y_true_all = []
    y_pred_all = []
    epistemic_unc_all = []
    total_unc_all = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)

            preds = model.mc_predict(X_batch, n_samples=n_samples).cpu()

            batch_mean = preds.mean(dim=0).squeeze(dim=-1)
            batch_var = preds.var(dim=0).squeeze(dim=-1)

            batch_epistemic = preds.squeeze(-1).var(dim=0)

            y_pred_all.append(batch_mean.numpy())
            total_unc_all.append(batch_var.numpy())
            epistemic_unc_all.append(batch_epistemic.numpy())
            y_true_all.append(y_batch.numpy())

    y_true = np.concatenate(y_true_all).flatten()
    y_pred = np.concatenate(y_pred_all).flatten()
    total_unc = np.concatenate(total_unc_all).flatten()
    epistemic_unc = np.concatenate(epistemic_unc_all).flatten()
    aleatoric_unc = np.clip(total_unc - epistemic_unc, a_min=0.0, a_max=None)

    return y_true, y_pred, aleatoric_unc, epistemic_unc, total_unc


def test_proc(
    model: nn.Module,
    criterion: Callable,
    dataloader_dict: Dict[str, DataLoader],
    device: torch.device,
    path_to_save: Path,
    n_samples: int = 100,
    target_inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    key: str = "test",
) -> None:
    """
    Evaluate a trained MC-Dropout model on a test dataset and save metrics to CSV.

    Uncertainty is decomposed into epistemic (model/reducible) and aleatoric
    (data/irreducible) components via MC-Dropout stochastic forward passes.

    Parameters
    ----------
    model : nn.Module
        Trained MC-Dropout model exposing ``enable_dropout()`` and ``mc_predict()``.
    criterion : Callable
        Loss function used to evaluate test loss.
    dataloader_dict : Dict[str, DataLoader]
        Dictionary containing DataLoader for the test dataset, keyed by ``key``.
    device : torch.device
        Device on which computations are performed (CPU or GPU).
    path_to_save : Path
        Directory path to save the test results CSV files.
    n_samples : int, optional
        Number of stochastic MC-Dropout forward passes. Default is 100.
    target_inverse_transform : callable, optional
        Function to inverse-transform target values and predictions, by default None.
    key : str, optional
        Key to select the correct DataLoader from ``dataloader_dict``.
        Use ``'test'`` for the held-out test set or ``'ext_test'`` for an
        external test set. Default is ``'test'``.

    Returns
    -------
    None
    """
    (
        y_test_all_np,
        y_test_pred_all_np,
        aleatoric_unc_all_np,
        epistemic_unc_all_np,
        total_unc_all_np,
    ) = _mc_inference(model, dataloader_dict[key], device, n_samples)

    model.eval()
    test_losses = []
    with torch.no_grad():
        for X_test, y_test in dataloader_dict[key]:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_test_pred = model(X_test)
            test_loss = criterion(y_test_pred, y_test)
            test_losses.append(test_loss.item())
    epoch_test_loss = np.mean(test_losses) if test_losses else 0.0

    if target_inverse_transform:
        y_test_all_np = target_inverse_transform(y_test_all_np)
        y_test_pred_all_np = target_inverse_transform(y_test_pred_all_np)
        scale = y_test_pred_all_np**2
        aleatoric_unc_all_np = scale * aleatoric_unc_all_np
        epistemic_unc_all_np = scale * epistemic_unc_all_np
        total_unc_all_np = scale * total_unc_all_np

    test_r2, test_mse, test_rmse, test_mae, test_rp, test_mape = cal_metrics(
        y_test_all_np, y_test_pred_all_np
    )

    print(
        f"\nTest Metrics-> R2: {test_r2:.3f}, RP: {test_rp:.3f}, MAPE: {test_mape:.3f}, "
        f"RMSE: {test_rmse:.3f}, MSE: {test_mse:.3f}, Loss: {epoch_test_loss:.3f}"
    )
    print("-----------------------------------------------------")

    results_dict = defaultdict(list)
    results_dict["Test_Loss"].append(epoch_test_loss)
    results_dict["Test_R2"].append(test_r2)
    results_dict["Test_MSE"].append(test_mse)
    results_dict["Test_RMSE"].append(test_rmse)
    results_dict["Test_MAE"].append(test_mae)
    results_dict["Test_RP"].append(test_rp)
    results_dict["Test_MAPE"].append(test_mape)

    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.to_csv(path_to_save / f"results_ffnn_model_{key}_set.csv", index=False)

    un_results_df = pd.DataFrame(
        {
            "y_test": y_test_all_np,
            "y_pred": y_test_pred_all_np,
            "aleatoric_unc": aleatoric_unc_all_np,
            "epistemic_unc": epistemic_unc_all_np,
            "total_unc": total_unc_all_np,
        }
    )
    un_results_df.to_csv(path_to_save / f"results_mcd_{key}_set.csv", index=False)


def val_proc(
    model: nn.Module,
    criterion: Callable,
    dataloader_dict: Dict[str, DataLoader],
    device: torch.device,
    path_to_save: Path,
    n_samples: int = 100,
    target_inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> None:
    """
    Evaluate a trained MC-Dropout model on a validation dataset and save metrics to CSV.

    Uncertainty is decomposed into epistemic (model/reducible) and aleatoric
    (data/irreducible) components via MC-Dropout stochastic forward passes.

    Parameters
    ----------
    model : nn.Module
        Trained MC-Dropout model exposing ``enable_dropout()`` and ``mc_predict()``.
    criterion : Callable
        Loss function used to evaluate validation loss.
    dataloader_dict : Dict[str, DataLoader]
        Dictionary containing DataLoader for the 'val' dataset.
    device : torch.device
        Device on which computations are performed (CPU or GPU).
    path_to_save : Path
        Directory path to save the validation results CSV files.
    n_samples : int, optional
        Number of stochastic MC-Dropout forward passes. Default is 100.
    target_inverse_transform : callable, optional
        Function to inverse-transform target values and predictions, by default None.

    Returns
    -------
    None
    """
    (
        y_val_all_np,
        y_val_pred_all_np,
        aleatoric_unc_all_np,
        epistemic_unc_all_np,
        total_unc_all_np,
    ) = _mc_inference(model, dataloader_dict["val"], device, n_samples)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_val, y_val in dataloader_dict["val"]:
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)
            val_losses.append(val_loss.item())
    epoch_val_loss = np.mean(val_losses) if val_losses else 0.0

    if target_inverse_transform:
        y_val_all_np = target_inverse_transform(y_val_all_np)
        y_val_pred_all_np = target_inverse_transform(y_val_pred_all_np)

    val_r2, val_mse, val_rmse, val_mae, val_rp, val_mape = cal_metrics(
        y_val_all_np, y_val_pred_all_np
    )

    print(
        f"\nValidation Metrics -> R2: {val_r2:.3f}, RP: {val_rp:.3f}, MAPE: {val_mape:.3f}, "
        f"RMSE: {val_rmse:.3f}, MSE: {val_mse:.3f}, Loss: {epoch_val_loss:.3f}"
    )
    print("-----------------------------------------------------")

    results_dict = defaultdict(list)
    results_dict["Val_Loss"].append(epoch_val_loss)
    results_dict["Val_R2"].append(val_r2)
    results_dict["Val_MSE"].append(val_mse)
    results_dict["Val_RMSE"].append(val_rmse)
    results_dict["Val_MAE"].append(val_mae)
    results_dict["Val_RP"].append(val_rp)
    results_dict["Val_MAPE"].append(val_mape)

    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.to_csv(path_to_save / "results_ffnn_model_val_set.csv", index=False)

    un_results_df = pd.DataFrame(
        {
            "y_val": y_val_all_np,
            "y_pred": y_val_pred_all_np,
            "aleatoric_unc": aleatoric_unc_all_np,
            "epistemic_unc": epistemic_unc_all_np,
            "total_unc": total_unc_all_np,
        }
    )
    un_results_df.to_csv(path_to_save / "results_mcd_val_set.csv", index=False)
