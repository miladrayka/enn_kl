import numpy as np
from typing import List, Tuple, Union
from sklearn import metrics
from scipy import stats


def cal_metrics(
    y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate common regression metrics between true and predicted values.

    Parameters
    ----------
    y_true : list of float or np.ndarray
        Ground truth target values.
    y_pred : list of float or np.ndarray
        Predicted target values.

    Returns
    -------
    tuple of float
        A tuple containing the following metrics:
        r2 : float
            Coefficient of determination (R^2 score).
        mse : float
            Mean squared error.
        rmse : float
            Root mean squared error.
        mae : float
            Mean absolute error.
        rp : float
            Pearson correlation coefficient.
        mape : float
            Mean absolute percentage error.
    """
    y_true_np = np.asarray(y_true).flatten()
    y_pred_np = np.asarray(y_pred).flatten()
    r2 = metrics.r2_score(y_true_np, y_pred_np)
    mse = metrics.mean_squared_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true_np, y_pred_np)
    try:
        rp = stats.pearsonr(y_true_np, y_pred_np)[0]
    except (ValueError, IndexError):
        rp = np.nan
    mape = metrics.mean_absolute_percentage_error(y_true_np, y_pred_np)
    return r2, mse, rmse, mae, rp, mape
