import json
import optuna
import torch
import torch.optim as optim
import pandas as pd
from pathlib import Path

from mc_dropout.config.config import CONFIG
from mc_dropout.models.network import FFNN
from mc_dropout.models.init_utils import init_weights
from mc_dropout.data.data_utils import random_split, create_dataloaders
from mc_dropout.training.losses import heteroscedastic_nll
from mc_dropout.models.early_stopping import EarlyStopper
from mc_dropout.training.train_eval import train_and_val_proc
from mc_dropout.data.feature_selection import feature_discard
from mc_dropout.utils.seed import set_seed


try:
    df_all_global = pd.read_csv(CONFIG["data_path"])
    y_original_global = df_all_global.loc[:, "kL"]
except FileNotFoundError:
    print(f"Error: Data file not found at {CONFIG['data_path']}")
    raise


def ffnn_objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization of the FFNN model.

    Parameters
    ----------
    trial : optuna.Trial
        A single Optuna trial object that suggests hyperparameters to be tested.

    Returns
    -------
    float
        The validation RMSE obtained from training with the sampled hyperparameters.
    """
    set_seed(CONFIG["random_seed"] + trial.number)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = CONFIG["training_params"]["epochs"]

    n_layers = trial.suggest_int("n_layers", 1, 5)
    num_neu_list = [
        trial.suggest_int(f"n_units_l_{i}", 50, 500, step=50) for i in range(n_layers)
    ]

    act_fn_name = trial.suggest_categorical(
        "activation_fn", ["relu", "leaky_relu", "elu"]
    )
    p = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "AdamW", "RMSprop"]
    )
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True)

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    y_transformed_trial = y_original_global
    if CONFIG["target_transform"]["enabled"]:
        y_transformed_trial = CONFIG["target_transform"]["function"](y_original_global)

    (X_train, X_val, X_test, y_train, y_val, y_test) = random_split(
        df_all_global.iloc[:, 1:-5],
        y_transformed_trial,
        CONFIG["train_ratio"],
        CONFIG["val_ratio"],
        CONFIG["test_ratio"],
        CONFIG["random_seed"] + trial.number,
    )

    df_train = pd.DataFrame(X_train, columns=df_all_global.columns[1:-5])
    df_train["kL"] = y_train
    df_val = pd.DataFrame(X_val, columns=df_all_global.columns[1:-5])
    df_val["kL"] = y_val
    df_test = pd.DataFrame(X_test, columns=df_all_global.columns[1:-5])
    df_test["kL"] = y_test

    selected_features = feature_discard(
        df_train, var_threshold=0.01, corr_threshold=0.95
    )
    X_train = df_train[selected_features].to_numpy()
    X_val = df_val[selected_features].to_numpy()
    X_test = df_test[selected_features].to_numpy()
    in_features = X_train.shape[1]

    model = FFNN(
        in_features=in_features,
        n_layers=n_layers,
        act_fn_name=act_fn_name,
        num_neu_list=num_neu_list,
        p=p,
    )
    model.apply(lambda m: init_weights(m, nonlinearity=act_fn_name))
    model.to(device)

    optimizer = getattr(optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=CONFIG["training_params"]["scheduler_patience"],
    )

    train_loader, val_loader, _ = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=batch_size
    )
    dataloader_dict = {"train": train_loader, "val": val_loader}

    criterion = heteroscedastic_nll
    early_stopper = EarlyStopper(
        patience=CONFIG["training_params"]["early_stopping_patience"],
        delta=CONFIG["training_params"]["early_stopping_delta"],
        minimize=True,
    )

    try:
        val_rmse, _ = train_and_val_proc(
            model,
            optimizer,
            criterion,
            epochs,
            scheduler,
            dataloader_dict,
            early_stopper,
            device,
            path_to_save=None,
            target_inverse_transform=(
                CONFIG["target_transform"]["inverse_function_for_metrics"]
                if CONFIG["target_transform"]["enabled"]
                else None
            ),
        )
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()

    trial.report(val_rmse, step=epochs)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_rmse


def run_optuna(n_trials: int = 100, output_dir: Path = None):
    """
    Run Optuna hyperparameter optimization for the FFNN model.

    Parameters
    ----------
    n_trials : int, optional
        Number of trials to run in the study (default is 20).
    output_dir : pathlib.Path, optional
        Directory path where study results and best parameters will be saved.

    Returns
    -------
    None
        The function saves the best parameters and full study results to disk.
    """
    if output_dir is None:
        output_dir = CONFIG["output_dir"] / "optuna_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="ffnn_mcdropout_tuning",
        sampler=optuna.samplers.TPESampler(seed=CONFIG["random_seed"]),
        direction="minimize",
    )
    study.optimize(ffnn_objective, n_trials=n_trials, timeout=None, gc_after_trial=True)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Validation RMSE): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params_path = output_dir / "best_trial_params.json"
    with open(best_params_path, "w") as fp:
        json.dump(trial.params, fp, indent=4)

    df_study = study.trials_dataframe()
    df_study.to_csv(output_dir / "optuna_study_history.csv", index=False)