# ---------------- Standard Library ----------------
import os
import re
import pickle
from collections import defaultdict
from pathlib import Path
import random

# ---------------- Type Hints ----------------
from typing import Dict, List, Tuple, Union

# ---------------- Visualization ----------------
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

# ---------------- Data & ML ----------------
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import torch
from uncertainty_toolbox.metrics_calibration import get_proportion_lists

# ---------------- Materials / Domain-specific ----------------
from pymatgen.core import Composition
from pymatgen.core.composition import Composition
from smact.metallicity import metallicity_score

from evidential_regression.data import data_utils
from evidential_regression.config.config import CONFIG
from evidential_regression.data.data_utils import create_dataloaders
from evidential_regression.models.network import FFNN
from evidential_regression.models.init_utils import init_weights
from evidential_regression.models.early_stopping import EarlyStopper
from evidential_regression.training.train_eval import (
    train_and_val_proc,
    test_proc,
    val_proc,
)
from evidential_regression.training.losses import nig_nll, evidential_regression
from evidential_regression.training import metrics
from evidential_regression.utils.seed import set_seed


def run_experiment(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    test_set: pd.DataFrame,
    test_set_name: str,
    output_dir: Union[str, Path],
) -> None:
    """
    Run experiments across multiple random seeds and model instances.

    Parameters
    ----------
    train_set : pandas.DataFrame
        Train set.
    val_set: pandas.DataFrame
        Val set.
    test_set : pandas.DataFrame
        External test set.
    test_set_name : str
        Name of external test set.
    output_dir : str or pathlib.Path
        Directory path for saving outputs, model checkpoints, and split files.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(output_dir)
    run_dir = output_dir / f"run_1"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Train shape:", train_set.shape)

    X_val = val_set.iloc[:, 1:-2].to_numpy()
    X_train = train_set.iloc[:, 1:-2].to_numpy()
    X_test = test_set.iloc[:, 1:-2].to_numpy()
    y_val = val_set.iloc[:, -2].to_numpy()
    y_train = train_set.iloc[:, -2].to_numpy()
    y_test = test_set.iloc[:, -2].to_numpy()

    split_dict = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    with open(run_dir / "dataset_split.pkl", "wb") as f:
        pickle.dump(split_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    in_features = X_train.shape[1]
    print(f"Input features for model: {in_features}")

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        batch_size=CONFIG["batch_size"],
        shuffle_train=CONFIG["shuffle_train"],
    )

    dataloaders = {"train": train_loader, "val": val_loader, test_set_name: test_loader}

    for model_idx in range(1, 6):
        print(f"\n--- Training model {model_idx} on Run 1 ---")
        set_seed(42 + model_idx)

        model_dir = run_dir / f"model_{model_idx}"
        model_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = model_dir / f"models/ffnn_model_best.pth"

        if best_model_path.exists():
            print("Best model found — skipping training.")

        else:

            model = FFNN(
                in_features,
                CONFIG["model_params"]["n_layers"],
                CONFIG["model_params"]["act_fn"],
                CONFIG["model_params"]["num_neu_list"],
                CONFIG["model_params"]["p"],
            )
            model.apply(lambda m: init_weights(m, nonlinearity=model.act_fn_name))
            model = model.to(device)

            optimizer = getattr(
                torch.optim, CONFIG["training_params"]["optimizer_name"]
            )(
                model.parameters(),
                lr=CONFIG["training_params"]["learning_rate"],
                weight_decay=CONFIG["training_params"]["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=CONFIG["training_params"]["scheduler_patience"],
            )

            early_stopper = EarlyStopper(
                patience=CONFIG["training_params"]["early_stopping_patience"],
                delta=CONFIG["training_params"]["early_stopping_delta"],
                minimize=True,
            )

            best_val_rmse, best_model_path = train_and_val_proc(
                model,
                optimizer,
                evidential_regression,
                CONFIG["training_params"]["epochs"],
                scheduler,
                dataloaders,
                early_stopper,
                device,
                model_dir,
                lamb_loss=CONFIG["training_params"]["loss_lamb"],
                target_inverse_transform=(
                    CONFIG["target_transform"]["inverse_function_for_metrics"]
                    if CONFIG["target_transform"]["enabled"]
                    else None
                ),
            )

            print(f"Model {model_idx} best validation RMSE: {best_val_rmse:.3f}")

        final_model = FFNN(
            in_features,
            CONFIG["model_params"]["n_layers"],
            CONFIG["model_params"]["act_fn"],
            CONFIG["model_params"]["num_neu_list"],
            CONFIG["model_params"]["p"],
        )
        final_model.load_state_dict(torch.load(best_model_path, map_location=device))
        final_model.to(device)

        test_proc(
            final_model,
            nig_nll,
            dataloaders,
            device,
            model_dir,
            target_inverse_transform=(
                CONFIG["target_transform"]["inverse_function_for_metrics"]
                if CONFIG["target_transform"]["enabled"]
                else None
            ),
            key=test_set_name,
        )

        val_proc(
            final_model,
            nig_nll,
            dataloaders,
            device,
            model_dir,
            target_inverse_transform=(
                CONFIG["target_transform"]["inverse_function_for_metrics"]
                if CONFIG["target_transform"]["enabled"]
                else None
            ),
        )


def create_strata(df, n_bins=5):
    """Creates a combined stratification column based on kL and T quantiles."""

    df[f"kL_bins"] = pd.qcut(df["kL"], q=n_bins, labels=False, duplicates="drop")
    df[f"T_bins"] = pd.qcut(df["T"], q=n_bins, labels=False, duplicates="drop")

    df["Strata"] = df["T_bins"].astype(str) + "_" + df["kL_bins"].astype(str)
    return df


total_fv_df = (
    pd.read_parquet("../feature_vectors/total_dataset.parquet")
    .sample(frac=1)
    .reset_index(drop=True)
)
telab_fv_df = (
    pd.read_parquet("../feature_vectors/unique_telab_dataset.parquet")
    .sample(frac=1)
    .reset_index(drop=True)
)
starry_fv_subdf = pd.read_csv("subset_starrydataset.csv")

total_fv_df = create_strata(total_fv_df)
df_train, df_val = train_test_split(
    total_fv_df,
    test_size=400,
    stratify=total_fv_df["Strata"],
    random_state=42,
)

df_train = pd.concat([df_train, starry_fv_subdf])
train_set = (
    df_train.drop(columns=["T_bins", "kL_bins", "Strata"])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

val_set = (
    df_val.drop(columns=["T_bins", "kL_bins", "Strata"])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

run_experiment(
    train_set,
    val_set,
    telab_fv_df,
    "telab",
    f"./aggregated_substarrydataset_results",
)
