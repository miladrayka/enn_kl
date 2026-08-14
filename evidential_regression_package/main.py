"""Main module for Evidential Regression FFNN training and evaluation."""

import pickle
import argparse
import random

import torch
import pandas as pd

from evidential_regression.config.config import CONFIG
from evidential_regression.data.data_utils import random_split, create_dataloaders
from evidential_regression.data.feature_selection import feature_discard
from evidential_regression.models.network import FFNN
from evidential_regression.models.init_utils import init_weights
from evidential_regression.models.early_stopping import EarlyStopper
from evidential_regression.training.train_eval import train_and_val_proc, test_proc, val_proc
from evidential_regression.training.losses import nig_nll, evidential_regression
from evidential_regression.utils.seed import set_seed
from evidential_regression.hpo.optuna_tuning import run_optuna


def train_pipeline():
    """
    Execute the full training and evaluation pipeline for FFNN models.

    The pipeline includes:
    - Loading the dataset and applying optional target transformation.
    - Performing multiple train/validation/test splits.
    - Feature selection for each split based on variance and correlation thresholds.
    - Creating DataLoaders for train, validation, and test sets.
    - Training multiple FFNN models per split with early stopping.
    - Saving the best model and metrics for each split.
    - Evaluating the best model on the test set.

    Notes
    -----
    Uses configuration parameters from `CONFIG` including model architecture,
    training parameters, batch size, and target transformation settings.
    """
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    DEVICE = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    )
    print(f"Using device: {DEVICE}")

    df_all = pd.read_csv(CONFIG["data_path"])
    y_original = df_all.loc[:, "kL"]
    y_transformed = y_original
    if CONFIG["target_transform"]["enabled"]:
        y_transformed = CONFIG["target_transform"]["function"](y_original)
        print("Target variable transformed.")

    for split_idx in range(1, 11):
        print(f"\n=== Split {split_idx} ===")
        split_seed = random.randint(1000, 100000)
        split_output_dir = CONFIG["output_dir"] / f"run_{split_idx}"
        split_output_dir.mkdir(parents=True, exist_ok=True)

        (X_train_np, X_val_np, X_test_np, y_train_np, y_val_np, y_test_np) = (
            random_split(
                df_all.iloc[:, 1:-5],
                y_transformed,
                CONFIG["train_ratio"],
                CONFIG["val_ratio"],
                CONFIG["test_ratio"],
                split_seed,
            )
        )

        df_train = pd.DataFrame(X_train_np, columns=df_all.columns[1:-5])
        df_train["kL"] = y_train_np
        df_val = pd.DataFrame(X_val_np, columns=df_all.columns[1:-5])
        df_val["kL"] = y_val_np
        df_test = pd.DataFrame(X_test_np, columns=df_all.columns[1:-5])
        df_test["kL"] = y_test_np

        selected_features = feature_discard(
            df_train, var_threshold=0.01, corr_threshold=0.95
        )
        print(f"Split {split_idx}: {len(selected_features)} features selected")

        split_feature_file = split_output_dir / "selected_features.txt"
        with open(split_feature_file, "w") as f:
            for feat in selected_features:
                f.write(feat + "\n")

        X_train = df_train[selected_features].to_numpy()
        X_val = df_val[selected_features].to_numpy()
        X_test = df_test[selected_features].to_numpy()
        y_train = df_train["kL"].to_numpy()
        y_val = df_val["kL"].to_numpy()
        y_test = df_test["kL"].to_numpy()

        split_dict = {"X_train": X_train, "X_val": X_val, "X_test": X_test, "y_train": y_train, "y_val": y_val, "y_test": y_test}
        split_dataset_file = split_output_dir / "dataset_split.pkl"
        with open(split_dataset_file, 'wb') as handle:
            pickle.dump(split_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        CONFIG["model_params"]["in_features"] = X_train.shape[1]
        print(f"Input features for model: {CONFIG['model_params']['in_features']}")

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
        dataloader_dict = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        for model_idx in range(1, 6):
            print(f"\n--- Training model {model_idx} on split {split_idx} ---")
            set_seed(random.randint(1000, 100000))
            model = FFNN(
                CONFIG["model_params"]["in_features"],
                CONFIG["model_params"]["n_layers"],
                CONFIG["model_params"]["act_fn"],
                CONFIG["model_params"]["num_neu_list"],
                CONFIG["model_params"]["p"],
            )
            model.apply(lambda m: init_weights(m, nonlinearity=model.act_fn_name))
            model = model.to(DEVICE)

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

            model_save_dir = split_output_dir / f"model_{model_idx}"
            model_save_dir.mkdir(parents=True, exist_ok=True)

            best_val_rmse, best_model_path = train_and_val_proc(
                model,
                optimizer,
                evidential_regression,
                CONFIG["training_params"]["epochs"],
                scheduler,
                dataloader_dict,
                early_stopper,
                DEVICE,
                model_save_dir,
                lamb_loss=CONFIG["training_params"]["loss_lamb"],
                target_inverse_transform=(
                    CONFIG["target_transform"]["inverse_function_for_metrics"]
                    if CONFIG["target_transform"]["enabled"]
                    else None
                ),
            )
            print(f"Model {model_idx} best validation RMSE: {best_val_rmse:.3f}")

            final_model = FFNN(
                CONFIG["model_params"]["in_features"],
                CONFIG["model_params"]["n_layers"],
                CONFIG["model_params"]["act_fn"],
                CONFIG["model_params"]["num_neu_list"],
                CONFIG["model_params"]["p"],
            )
            final_model.load_state_dict(
                torch.load(best_model_path, map_location=DEVICE)
            )
            final_model.to(DEVICE)

            test_proc(
                final_model,
                nig_nll,
                dataloader_dict,
                DEVICE,
                model_save_dir,
                target_inverse_transform=(
                    CONFIG["target_transform"]["inverse_function_for_metrics"]
                    if CONFIG["target_transform"]["enabled"]
                    else None
                ),
            )

            val_proc(
                final_model,
                nig_nll,
                dataloader_dict,
                DEVICE,
                model_save_dir,
                target_inverse_transform=(
                    CONFIG["target_transform"]["inverse_function_for_metrics"]
                    if CONFIG["target_transform"]["enabled"]
                    else None
                ),
            )


def main():
    """
    Entry point for training or hyperparameter optimization.

    Command-line Arguments
    ---------------------
    --hpo : bool
        If provided, runs Optuna hyperparameter optimization instead of
        the full training pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Evidential Regression FFNN Training / HPO"
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Enable Optuna hyperparameter optimization instead of full training pipeline.",
    )
    args = parser.parse_args()

    if args.hpo:
        print("Running Optuna hyperparameter optimization...")
        run_optuna()
    else:
        train_pipeline()


if __name__ == "__main__":
    main()
