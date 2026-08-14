from typing import Optional, Tuple, Union
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch


class RegressionDataset(Dataset):
    """
    Custom PyTorch Dataset for regression tasks.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).

    Attributes
    ----------
    X : torch.Tensor
        Tensor containing input features.
    y : torch.Tensor
        Tensor containing target values with an additional dimension for compatibility.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            A tuple containing the feature tensor and target tensor.
        """
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing sets.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Feature arrays for training, validation, and testing.
    y_train, y_val, y_test : np.ndarray
        Target arrays for training, validation, and testing.
    batch_size : int, optional
        Number of samples per batch (default is 32).
    shuffle_train : bool, optional
        Whether to shuffle the training set (default is True).

    Returns
    -------
    tuple of (DataLoader, DataLoader, DataLoader)
        Train, validation, and test DataLoaders.
    """
    train_dataset = RegressionDataset(X_train, y_train)
    val_dataset = RegressionDataset(X_val, y_val)
    test_dataset = RegressionDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def random_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset randomly into training, validation, and test sets.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Input features.
    y : Union[pd.Series, np.ndarray]
        Target values.
    train_ratio : float, optional
        Proportion of training samples (default is 0.8).
    val_ratio : float, optional
        Proportion of validation samples (default is 0.1).
    test_ratio : float, optional
        Proportion of test samples (default is 0.1).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple of np.ndarray
        (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0.")

    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y_np = y.to_numpy() if isinstance(y, pd.Series) else y

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_np, y_np, test_size=(val_ratio + test_ratio), random_state=random_state
    )
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


def stratified_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    y_clusters: Union[pd.Series, np.ndarray],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Split dataset into stratified train, validation, and test sets.

    Ensures each split maintains proportional representation
    of the clusters provided in `y_clusters`.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Input features.
    y : Union[pd.Series, np.ndarray]
        Target values.
    y_clusters : Union[pd.Series, np.ndarray]
        Cluster or stratum labels used for stratification.
    train_ratio : float, optional
        Proportion of training samples (default is 0.8).
    val_ratio : float, optional
        Proportion of validation samples (default is 0.1).
    test_ratio : float, optional
        Proportion of test samples (default is 0.1).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple of np.ndarray
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_train, y_val, y_test,
         y_clusters_train, y_clusters_val, y_clusters_test)

    Raises
    ------
    ValueError
        If train_ratio + val_ratio + test_ratio != 1.0.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0.")

    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y_np = y.to_numpy() if isinstance(y, pd.Series) else y
    y_clusters_np = (
        y_clusters.to_numpy() if isinstance(y_clusters, pd.Series) else y_clusters
    )

    cluster_counts = Counter(y_clusters_np)
    stratify_param = y_clusters_np if min(cluster_counts.values()) >= 2 else None

    X_train, X_temp, y_train, y_temp, y_clusters_train, y_clusters_temp = (
        train_test_split(
            X_np,
            y_np,
            y_clusters_np,
            test_size=(val_ratio + test_ratio),
            stratify=stratify_param,
            random_state=random_state,
        )
    )

    relative_test_size = (
        test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.0
    )
    temp_cluster_counts = Counter(y_clusters_temp)
    stratify_temp = y_clusters_temp if min(temp_cluster_counts.values()) >= 2 else None

    X_val, X_test, y_val, y_test, y_clusters_val, y_clusters_test = train_test_split(
        X_temp,
        y_temp,
        y_clusters_temp,
        test_size=relative_test_size,
        stratify=stratify_temp,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        y_clusters_train,
        y_clusters_val,
        y_clusters_test,
    )
