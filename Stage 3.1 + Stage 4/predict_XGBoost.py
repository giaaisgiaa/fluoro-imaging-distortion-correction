from __future__ import annotations
import numpy as np
from typing import Tuple, Sequence
import matplotlib.pyplot as plt
from non_stationary_models.forward_xgboost import XGBDisplacementModel


def load_dataset(path: str = "data/dot_trajectories.json") -> np.ndarray:
    """
    Load dot trajectories data and convert to numpy array with shape (T, N, 2).
    """
    from utils.dataset_read import load_frame_datasets
    
    # Load datasets
    datasets = load_frame_datasets(path)
    frames = sorted(datasets.keys(), key=lambda k: int(k.split()[-1]))
    D = np.stack([datasets[frame] for frame in frames], axis=0)
    
    assert D.ndim == 3 and D.shape[2] == 2, "D must have shape (T, N, 2)"
    return D.astype(np.float64, copy=False)


def split_indices(n_points: int, test_frac: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Split point indices into train/val (20% validation by default)."""
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_points)
    n_test = max(1, int(round(test_frac * n_points)))
    test_idx = np.sort(rng.choice(all_idx, size=n_test, replace=False))
    train_idx = np.setdiff1d(all_idx, test_idx)
    return train_idx, test_idx


def build_supervised_pairs(D: np.ndarray, point_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X, Y for regression:
      X = [x_t, y_t, t],   Y = [dx_t, dy_t]  where dx_t = x_{t+1}-x_t, same for y
    """
    T, _, _ = D.shape
    X_list, Y_list = [], []
    t_col = np.arange(T - 1, dtype=np.float64)[:, None]  # (T-1, 1)
    for i in point_indices:
        traj = D[:, i, :]                       # (T, 2)
        xyt = np.hstack([traj[:-1, :], t_col])  # (T-1, 3)
        dxy = traj[1:, :] - traj[:-1, :]        # (T-1, 2)
        X_list.append(xyt)
        Y_list.append(dxy)
    return np.vstack(X_list), np.vstack(Y_list)


def evaluate_rollouts(
    D: np.ndarray,
    val_indices: Sequence[int],
    model: XGBDisplacementModel,
    verbose: bool = False,
) -> Tuple[float, np.ndarray]:
    """Evaluate model with autoregressive rollout."""
    T, _, _ = D.shape
    preds, gts = [], []
    for i in val_indices:
        gt = D[:, i, :]
        pred = model.rollout(gt[0], T)
        preds.append(pred)
        gts.append(gt)
    P = np.stack(preds, axis=1)  # (T, N_val, 2)
    G = np.stack(gts, axis=1)

    sqerr = (P - G) ** 2
    per_t_rmse = np.sqrt(sqerr.mean(axis=(1, 2)))   # (T,)
    per_t_std = np.sqrt(sqerr).std(axis=1).mean(axis=1)  # Standard deviation across dots
    overall_rmse = np.sqrt(sqerr[1:, :, :].mean())  # exclude t=0
    avg_rmse = per_t_rmse[1:].mean()  # Average RMSE excluding t=0

    if verbose:
        plt.figure(figsize=(12, 6))
        time = np.arange(T)
        plt.plot(time, per_t_rmse, 'b-', label='Per-timestep RMSE', linewidth=2)
        plt.fill_between(time, 
                        per_t_rmse - per_t_std, 
                        per_t_rmse + per_t_std, 
                        color='b', alpha=0.2, 
                        label='+/- std dev')
        plt.axhline(y=avg_rmse, color='r', linestyle='--', 
                   label=f'Average RMSE = {avg_rmse:.3f} +/- {per_t_std[1:].mean():.3f}', 
                   linewidth=2)
        plt.xlabel("Timestep")
        plt.ylabel("RMSE (pixels)")
        plt.title("Validation RMSE over time (XGBoost)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Save the plot
        plt.savefig('visualizations/xgboost/rmse_over_time.png', dpi=300, bbox_inches='tight')
        plt.show()

    return float(overall_rmse), per_t_rmse, avg_rmse, per_t_std


def main():
    # --- Load data ---
    D = load_dataset("data/dot_trajectories.json")
    T, N, two = D.shape
    print(f"Loaded D with shape {D.shape}")

    # --- Split train/val by indices ---
    train_idx, val_idx = split_indices(N, test_frac=0.20, seed=42)
    print(f"Train points: {len(train_idx)} | Val points: {len(val_idx)}")

    # --- Build supervised regression dataset ---
    X_train, Y_train = build_supervised_pairs(D, train_idx)
    print(f"Training samples: {X_train.shape[0]}")

    # --- Fit XGBoost model ---
    model = XGBDisplacementModel().fit(X_train, Y_train)

    # --- Evaluate ---
    overall_rmse, per_t_rmse, avg_rmse, per_t_std = evaluate_rollouts(D, val_idx, model, verbose=True)

    print("\n=== Validation metrics (XGBoost) ===")
    print(f"Overall RMSE: {overall_rmse:.3f} pixels")
    print(f"Average RMSE: {avg_rmse:.3f} +/- {per_t_std[1:].mean():.3f} pixels")
    print(f"Per-timestep RMSE range: [{per_t_rmse.min():.3f}, {per_t_rmse.max():.3f}] pixels")
    print(f"Per-timestep RMSE mean ± std: {per_t_rmse.mean():.3f} ± {per_t_rmse.std():.3f} pixels")


if __name__ == "__main__":
    main()