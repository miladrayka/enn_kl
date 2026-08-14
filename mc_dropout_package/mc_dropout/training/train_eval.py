from typing import Dict, Callable, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mc_dropout.training.metrics import cal_metrics
from mc_dropout.training.losses import heteroscedastic_nll
from mc_dropout.models.early_stopping import EarlyStopper
from mc_dropout.config.config import CONFIG


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
    Train and validate a PyTorch model with early stopping and metric tracking.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained and validated.
    optimizer : optim.Optimizer
        Optimizer for updating model parameters.
    criterion : Callable
        Loss function used for training (e.g., heteroscedastic_nll).
    epochs : int
        Number of training epochs.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    dataloader_dict : dict
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
        best_model_save_path = model_save_dir / f"ffnn_model_best.pth"
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        y_train_all = []
        y_train_pred_all = []
        for X_train, y_train in dataloader_dict["train"]:
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            y_train_pred_dist_params = model(X_train)
            train_loss = criterion(*y_train_pred_dist_params, y_train)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            mu, _ = [
                d.squeeze().detach().cpu().numpy() for d in y_train_pred_dist_params
            ]
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
                y_val_pred_dist_params = model(X_val)
                val_loss = criterion(*y_val_pred_dist_params, y_val)
                mu, _ = [
                    d.squeeze().detach().cpu().numpy() for d in y_val_pred_dist_params
                ]
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

        print(f"\n Epoch {epoch+1}/{epochs}")
        print(
            f"Train Metrics-> R2: {train_r2:.3f}, RP: {train_rp:.3f}, MAPE: {train_mape:.3f}, RMSE: {train_rmse:.3f}, MSE: {train_mse:.3f}, Loss: {epoch_train_loss:.3f}"
        )
        print(
            f"Val Metrics-> R2: {val_r2:.3f}, RP: {val_rp:.3f}, MAPE: {val_mape:.3f}, RMSE: {val_rmse:.3f}, MSE: {val_mse:.3f}, Loss: {epoch_val_loss:.3f}"
        )
        print("-----------------------------------------------------")

        if early_stopper.early_stop(val_rmse, model, epoch):
            print("Training is stopped because of early stopping.")
            if best_model_save_path and early_stopper.best_model_state:
                torch.save(early_stopper.best_model_state, best_model_save_path)
                print(
                    f"Best model saved at epoch {early_stopper.best_epoch+1} with RMSE: {early_stopper.best_metric:.3f}"
                )
            break
        scheduler.step(val_rmse)

    results_df = pd.DataFrame.from_dict(results_dict)
    if path_to_save:
        results_df_path = path_to_save / "results_ffnn_model_train_and_val.csv"
        results_df.to_csv(results_df_path, index=False)

    if not early_stopper.early_stop_triggered:
        if best_model_save_path and early_stopper.best_model_state:
            torch.save(early_stopper.best_model_state, best_model_save_path)
            print(
                f"Training finished. Best model saved at epoch {early_stopper.best_epoch+1} with RMSE: {early_stopper.best_metric:.3f}"
            )

    return early_stopper.best_metric, best_model_save_path


def test_proc(
    model: nn.Module,
    criterion: Callable,
    dataloader_dict: Dict[str, DataLoader],
    device: torch.device,
    path_to_save: Path,
    target_inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    key: str = "test",
    mc_forward_passes: int = CONFIG["training_params"]["mc_forward_passes"]
) -> None:
    """
    Evaluate a trained PyTorch model on a test dataset using MC Dropout and save metrics to CSV.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    criterion : Callable
        Loss function used to evaluate test loss (e.g., heteroscedastic_nll).
    dataloader_dict : dict
        Dictionary containing DataLoaders.
    device : torch.device
        Device on which computations are performed (CPU or GPU).
    path_to_save : Path
        Path to save the test results CSV.
    target_inverse_transform : callable, optional
        Function to inverse-transform target values and predictions, by default None.
    key : str, optional
        Key specifying which dataloader to evaluate on, by default "test".
    mc_forward_passes : int, optional
        Number of forward passes to run for MC Dropout uncertainty estimation,
        derived from CONFIG by default.
    """
    model.eval()
    model.enable_dropout()  # Enable dropout explicitly for MC approximation
    
    test_losses = []
    y_test_all = []
    y_test_pred_all = []
    aleatoric_unc_all = []
    epistemic_unc_all = []
    total_unc_all = []
    
    with torch.no_grad():
        for X_test, y_test in dataloader_dict[key]:
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            mus = []
            log_vars = []
            for _ in range(mc_forward_passes):
                mu, log_var = model(X_test)
                mus.append(mu)
                log_vars.append(log_var)
            
            mus = torch.stack(mus)
            log_vars = torch.stack(log_vars)
            
            # Mean and Variance across MC samples
            pred_mu = mus.mean(dim=0)
            pred_var = torch.exp(log_vars).mean(dim=0)
            epistemic_var = mus.var(dim=0, unbiased=False)
            
            test_loss = criterion(pred_mu, torch.log(pred_var), y_test)
            
            aleatoric_unc = pred_var.squeeze().detach().cpu().numpy()
            epistemic_unc = epistemic_var.squeeze().detach().cpu().numpy()
            total_unc = aleatoric_unc + epistemic_unc

            test_losses.append(test_loss.item())
            y_test_all.append(y_test.detach().cpu().numpy())
            y_test_pred_all.append(pred_mu.squeeze().detach().cpu().numpy())
            aleatoric_unc_all.append(aleatoric_unc)
            epistemic_unc_all.append(epistemic_unc)
            total_unc_all.append(total_unc)

    epoch_test_loss = np.mean(test_losses) if test_losses else 0.0
    y_test_all_np = np.concatenate(y_test_all).flatten()
    y_test_pred_all_np = np.concatenate(y_test_pred_all).flatten()
    aleatoric_unc_all_np = np.concatenate(aleatoric_unc_all).flatten()
    epistemic_unc_all_np = np.concatenate(epistemic_unc_all).flatten()
    total_unc_all_np = np.concatenate(total_unc_all).flatten()

    if target_inverse_transform:
        y_test_all_np = target_inverse_transform(y_test_all_np)
        kL_pred_mean_orig = target_inverse_transform(y_test_pred_all_np)
        y_test_pred_all_np = kL_pred_mean_orig
        kL_mean_squared = kL_pred_mean_orig ** 2
        
        # Scaling uncertainties under transformation
        aleatoric_unc_all_np = kL_mean_squared * aleatoric_unc_all_np
        epistemic_unc_all_np = kL_mean_squared * epistemic_unc_all_np
        total_unc_all_np = kL_mean_squared * total_unc_all_np

    test_r2, test_mse, test_rmse, test_mae, test_rp, test_mape = cal_metrics(
        y_test_all_np, y_test_pred_all_np
    )

    results_dict = defaultdict(list)
    results_dict["Test_Loss"].append(epoch_test_loss)
    results_dict["Test_R2"].append(test_r2)
    results_dict["Test_MSE"].append(test_mse)
    results_dict["Test_RMSE"].append(test_rmse)
    results_dict["Test_MAE"].append(test_mae)
    results_dict["Test_RP"].append(test_rp)
    results_dict["Test_MAPE"].append(test_mape)

    print(
        f"\nTest Metrics-> R2: {test_r2:.3f}, RP: {test_rp:.3f}, MAPE: {test_mape:.3f}, RMSE: {test_rmse:.3f}, MSE: {test_mse:.3f}, Loss: {epoch_test_loss:.3f}"
    )
    print("-----------------------------------------------------")

    results_df = pd.DataFrame.from_dict(results_dict)
    results_df_path = path_to_save / f"results_ffnn_model_{key}_set.csv"
    results_df.to_csv(results_df_path, index=False)

    un_results_df = pd.DataFrame({
        "y_test": y_test_all_np,
        "y_pred": y_test_pred_all_np,
        "aleatoric_unc": aleatoric_unc_all_np,
        "epistemic_unc": epistemic_unc_all_np,
        "total_unc": total_unc_all_np
    })

    un_results_path = path_to_save / f"results_mcdropout_{key}_set.csv"
    un_results_df.to_csv(un_results_path, index=False)

def val_proc(
    model: nn.Module,
    criterion: Callable,
    dataloader_dict: Dict[str, DataLoader],
    device: torch.device,
    path_to_save: Path,
    target_inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    mc_forward_passes: int = CONFIG["training_params"]["mc_forward_passes"]
) -> None:
    """
    Evaluate a trained PyTorch model on a validation dataset using MC Dropout and save metrics to CSV.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    criterion : Callable
        Loss function used to evaluate validation loss (e.g., heteroscedastic_nll).
    dataloader_dict : dict
        Dictionary containing DataLoader for the 'val' dataset.
    device : torch.device
        Device on which computations are performed (CPU or GPU).
    path_to_save : Path
        Path to save the validation results CSV.
    target_inverse_transform : callable, optional
        Function to inverse-transform target values and predictions, by default None.
    mc_forward_passes : int, optional
        Number of forward passes to run for MC Dropout uncertainty estimation,
        derived from CONFIG by default.
    """
    model.eval()
    model.enable_dropout()
    
    val_losses = []
    y_val_all = []
    y_val_pred_all = []
    aleatoric_unc_all = []
    epistemic_unc_all = []
    total_unc_all = []

    with torch.no_grad():
        for X_val, y_val in dataloader_dict["val"]:
            X_val, y_val = X_val.to(device), y_val.to(device)
            
            mus = []
            log_vars = []
            for _ in range(mc_forward_passes):
                mu, log_var = model(X_val)
                mus.append(mu)
                log_vars.append(log_var)
            
            mus = torch.stack(mus)
            log_vars = torch.stack(log_vars)
            
            pred_mu = mus.mean(dim=0)
            pred_var = torch.exp(log_vars).mean(dim=0)
            epistemic_var = mus.var(dim=0, unbiased=False)
            
            val_loss = criterion(pred_mu, torch.log(pred_var), y_val)
            
            aleatoric_unc = pred_var.squeeze().detach().cpu().numpy()
            epistemic_unc = epistemic_var.squeeze().detach().cpu().numpy()
            total_unc = aleatoric_unc + epistemic_unc

            val_losses.append(val_loss.item())
            y_val_all.append(y_val.detach().cpu().numpy())
            y_val_pred_all.append(pred_mu.squeeze().detach().cpu().numpy())
            
            aleatoric_unc_all.append(aleatoric_unc)
            epistemic_unc_all.append(epistemic_unc)
            total_unc_all.append(total_unc)

    epoch_val_loss = np.mean(val_losses) if val_losses else 0.0
    y_val_all_np = np.concatenate(y_val_all).flatten()
    y_val_pred_all_np = np.concatenate(y_val_pred_all).flatten()
    aleatoric_unc_all_np = np.concatenate(aleatoric_unc_all).flatten()
    epistemic_unc_all_np = np.concatenate(epistemic_unc_all).flatten()
    total_unc_all_np = np.concatenate(total_unc_all).flatten()

    if target_inverse_transform:
        y_val_all_np = target_inverse_transform(y_val_all_np)
        kL_pred_mean_orig = target_inverse_transform(y_val_pred_all_np)
        y_val_pred_all_np = kL_pred_mean_orig
        kL_mean_squared = kL_pred_mean_orig ** 2
        
        aleatoric_unc_all_np = kL_mean_squared * aleatoric_unc_all_np
        epistemic_unc_all_np = kL_mean_squared * epistemic_unc_all_np
        total_unc_all_np = kL_mean_squared * total_unc_all_np

    val_r2, val_mse, val_rmse, val_mae, val_rp, val_mape = cal_metrics(
        y_val_all_np, y_val_pred_all_np
    )

    results_dict = defaultdict(list)
    results_dict["Val_Loss"].append(epoch_val_loss)
    results_dict["Val_R2"].append(val_r2)
    results_dict["Val_MSE"].append(val_mse)
    results_dict["Val_RMSE"].append(val_rmse)
    results_dict["Val_MAE"].append(val_mae)
    results_dict["Val_RP"].append(val_rp)
    results_dict["Val_MAPE"].append(val_mape)

    print(
        f"\nValidation Metrics -> R2: {val_r2:.3f}, RP: {val_rp:.3f}, MAPE: {val_mape:.3f}, RMSE: {val_rmse:.3f}, MSE: {val_mse:.3f}, Loss: {epoch_val_loss:.3f}"
    )
    print("-----------------------------------------------------")

    results_df = pd.DataFrame.from_dict(results_dict)
    results_df_path = path_to_save / "results_ffnn_model_val_set.csv"
    results_df.to_csv(results_df_path, index=False)

    un_results_df = pd.DataFrame({
        "y_val": y_val_all_np,
        "y_pred": y_val_pred_all_np,
        "aleatoric_unc": aleatoric_unc_all_np,
        "epistemic_unc": epistemic_unc_all_np,
        "total_unc": total_unc_all_np
    })

    un_results_path = path_to_save / "results_mcdropout_val_set.csv"
    un_results_df.to_csv(un_results_path, index=False)