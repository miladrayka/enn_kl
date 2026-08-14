from pathlib import Path
import numpy as np

CONFIG = {
    "random_seed": 42,
    "data_path": "all_data.csv",
    "output_dir": Path("./experiment_results"),
    "train_ratio": 0.8,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
    "batch_size": 128,
    "shuffle_train": True,
    "model_params": {
        "in_features": 140,
        "n_layers": 4,
        "act_fn": "relu",
        "num_neu_list": [250, 150, 450, 500],
        "p": 0.1,
    },
    "training_params": {
        "epochs": 200,
        "learning_rate": 0.005940791968170906,
        "weight_decay": 1.530543598678328e-05,
        "optimizer_name": "Adam",
        "mc_forward_passes": 30,
        "scheduler_patience": 5,
        "early_stopping_patience": 20,
        "early_stopping_delta": 0.001,
        "gradient_clip_norm": 3.0,
    },
    "target_transform": {
        "enabled": False,
        "function": np.log,
        "inverse_function_for_metrics": lambda x: np.exp(x),
    },
}
