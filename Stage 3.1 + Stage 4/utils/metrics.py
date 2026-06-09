import numpy as np

def metrics_multioutput(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred: shape (N, 2). Returns a dict with:
      - rmse_overall: scalar RMSE over both components
      - rmse_per_dim: array(2,) RMSE per component
      - r2_overall: scalar R^2 over both components
      - r2_per_dim: array(2,) R^2 per component
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    assert y_true.shape == y_pred.shape and y_true.ndim == 2 and y_true.shape[1] == 2

    resid = y_true - y_pred
    # RMSE
    rmse_per_dim = np.sqrt(np.mean(resid**2, axis=0))
    rmse_overall = np.sqrt(np.mean(resid**2))

    # R^2 (overall)
    y_mean = np.mean(y_true, axis=0)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_true - y_mean)**2)
    if ss_tot == 0:
        r2_overall = 1.0 if ss_res == 0 else 0.0
    else:
        r2_overall = 1.0 - ss_res / ss_tot

    # R^2 (per component)
    r2_per_dim = np.empty(2)
    for j in range(2):
        ss_res_j = np.sum((y_true[:, j] - y_pred[:, j])**2)
        ss_tot_j = np.sum((y_true[:, j] - y_true[:, j].mean())**2)
        r2_per_dim[j] = 1.0 - ss_res_j / ss_tot_j if ss_tot_j != 0 else (1.0 if ss_res_j == 0 else 0.0)

    return {
        "rmse_overall": float(rmse_overall),
        "rmse_per_dim": rmse_per_dim,
        "r2_overall": float(r2_overall),
        "r2_per_dim": r2_per_dim,
    }
