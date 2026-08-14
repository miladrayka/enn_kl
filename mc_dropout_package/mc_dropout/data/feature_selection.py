import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


def feature_discard(
    df_train: pd.DataFrame,
    var_threshold: float = 0.01,
    corr_threshold: float = 0.95,
) -> list:
    """
    Perform sequential feature selection on a training DataFrame.

    This function applies three stages of feature selection:
    1. **Variance filtering**: removes features with low variance.
    2. **Correlation filtering**: removes highly correlated features.
    3. **LASSO regression**: removes features with zero coefficients after fitting.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training dataset containing feature columns and target columns.
        The target column is expected to be near the end (e.g., 'kL'),
        and the first 86 columns are always retained (domain-specific constraint).
    var_threshold : float, optional
        Minimum variance required to keep a feature (default is 0.01).
    corr_threshold : float, optional
        Maximum allowed Pearson correlation between two features.
        Features exceeding this threshold are dropped (default is 0.95).

    Returns
    -------
    list of str
        List of column names of the selected features (excluding the target).
    """
    mask_var = (
        [True] * 86
        + list(df_train.iloc[:, 86:-2].var(axis=0) > var_threshold)
        + [True] * 2
    )
    df_var = df_train.loc[:, mask_var]
    print(f"Variance filter: {df_var.shape[1]-1} features retained (plus target)")

    corr_matrix = df_var.iloc[:, 86:-2].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [
        col for col in upper_tri.columns if any(upper_tri[col] >= corr_threshold)
    ]
    df_corr = df_var.drop(columns=to_drop)
    print(f"Correlation filter: {df_corr.shape[1]-1} features retained (plus target)")

    scaler = StandardScaler()
    X = scaler.fit_transform(df_corr.iloc[:, 86:-2])
    y = df_corr.iloc[:, -2]
    reg = Lasso(alpha=0.01)
    reg.fit(X, y)

    mask_lasso = [True] * 86 + list(reg.coef_ != 0) + [True] * 2
    df_final = df_corr.loc[:, mask_lasso]
    print(f"LASSO filter: {df_final.shape[1]-1} features retained (plus target)")

    selected_features = df_final.columns[:-1]
    return selected_features.tolist()
