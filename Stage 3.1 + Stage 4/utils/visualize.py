import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple

# ---------- helpers (optional but handy) ----------

def idw_predict(X_train: np.ndarray, V_train: np.ndarray, X_query: np.ndarray,
                k: int = 8, eps: float = 1e-8) -> np.ndarray:
    """
    Inverse-distance weighted (IDW) interpolation of vectors.
    Not a 'model'—just a quick baseline for visualization.
    """
    # pairwise distances (QxN)
    d2 = ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)
    d = np.sqrt(d2 + eps)
    # take k nearest neighbors
    idx = np.argpartition(d, kth=min(k, len(X_train)-1), axis=1)[:, :k]
    d_knn = np.take_along_axis(d, idx, axis=1)
    w = 1.0 / (d_knn + eps)
    w /= w.sum(axis=1, keepdims=True)
    V_knn = V_train[idx]  # (Q,k,2)
    Vq = (w[..., None] * V_knn).sum(axis=1)
    return Vq

def make_grid_from_points(X: np.ndarray, nx: int = 20, ny: Optional[int] = None,
                          margin: float = 0.05) -> np.ndarray:
    """
    Create a uniform grid covering the bounding box of X.
    """
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    xmin -= margin * dx; xmax += margin * dx
    ymin -= margin * dy; ymax += margin * dy
    if ny is None:
        ny = int(round(nx * (dy / dx))) if dx > 0 else nx
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    Gx, Gy = np.meshgrid(gx, gy)
    return np.c_[Gx.ravel(), Gy.ravel()]

# ---------- 1) main vector-field visualizer ----------

def visualize_vectors(
    X: np.ndarray,
    V: np.ndarray,
    *,
    extra_points: Optional[np.ndarray] = None,
    extra_vectors: Optional[np.ndarray] = None,
    predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
    point_size: float = 12,
    base_color: str = "C0",
    extra_color: str = "C3",
    true_color: str = "C2",
    vector_scale: float = 1.0,     # scales vectors for display only
    show: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot points X with arrows V. Optionally plot 'extra_points' that weren't in X,
    using a different color. Their vectors can be provided via 'extra_vectors' or
    computed on the fly with 'predict_fn'.
    """
    assert X.shape[1] == 2 and V.shape[1] == 2, "X and V must be (N,2)."
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Main points
    ax.scatter(X[:, 0], X[:, 1], s=point_size, c=base_color, label="data points")
    ax.quiver(
        X[:, 0], X[:, 1],
        V[:, 0] * vector_scale, V[:, 1] * vector_scale,
        angles="xy", scale_units="xy", scale=1, width=0.003, color=base_color,
        label="vectors"
    )

    # Extra points not in X (e.g., grid or novel locations)
    if extra_points is not None and len(extra_points) > 0:
        if extra_vectors is None and predict_fn is not None:
            extra_vectors = predict_fn(extra_points)
        if extra_vectors is None:
            raise ValueError("Provide extra_vectors or a predict_fn to compute them.")
        assert extra_vectors.shape == (len(extra_points), 2), "extra_vectors must match extra_points."

        # Filter out any accidental duplicates (so 'not in coordinates' stays true)
        existing = set(map(tuple, np.round(X, 9)))
        mask_new = np.array([tuple(np.round(p, 9)) not in existing for p in extra_points])
        XP = extra_points[mask_new]
        VP = extra_vectors[mask_new]

        if len(XP):
            ax.scatter(XP[:, 0], XP[:, 1], s=point_size, c=extra_color, label="extra points")
            ax.quiver(
                XP[:, 0], XP[:, 1],
                VP[:, 0] * vector_scale, VP[:, 1] * vector_scale,
                angles="xy", scale_units="xy", scale=1, width=0.003, color=extra_color,
                label="extra vectors"
            )

    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", framealpha=0.8)
    if show:
        plt.show()
    return ax

def compare_arrow_fields(
    X: np.ndarray,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    *,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
    vector_scale: float = 1.0,   # display-only scale
    point_size: float = 18,
    color_true: str = "C2",
    color_pred: str = "C1",
    error_color: str = "0.4",
    error_alpha: float = 0.5,
    draw_error_lines: bool = True,
    max_points: Optional[int] = None,  # subsample for cluttered plots
    seed: Optional[int] = 42,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
):
    """
    Visualize two vector fields (true vs predicted) attached to the SAME base points X.

    Parameters
    ----------
    X : (N,2)
    Y_true : (N,2)
    Y_pred : (N,2)
    vector_scale : multiply vectors for display (does not change data)
    max_points : if set and N > max_points, randomly subsample to this many for plotting
    draw_error_lines : draw a line between predicted tip and true tip at each point
    """
    X = np.asarray(X, float); Y_true = np.asarray(Y_true, float); Y_pred = np.asarray(Y_pred, float)
    assert X.shape == Y_true.shape == Y_pred.shape and X.shape[1] == 2, "All arrays must be (N,2)."
    N = len(X)

    # Optional subsampling to avoid clutter
    if max_points is not None and N > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=max_points, replace=False)
        X, Y_true, Y_pred = X[idx], Y_true[idx], Y_pred[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Base points
    ax.scatter(X[:, 0], X[:, 1], s=point_size*0.7, c="k", alpha=0.35, label="points")

    # True arrows
    ax.quiver(
        X[:, 0], X[:, 1],
        Y_true[:, 0]*vector_scale, Y_true[:, 1]*vector_scale,
        angles="xy", scale_units="xy", scale=1,
        width=0.0035, color=color_true, alpha=0.95, label="true"
    )

    # Predicted arrows
    ax.quiver(
        X[:, 0], X[:, 1],
        Y_pred[:, 0]*vector_scale, Y_pred[:, 1]*vector_scale,
        angles="xy", scale_units="xy", scale=1,
        width=0.0035, color=color_pred, alpha=0.95, label="predicted"
    )

    # Error lines (from predicted tip to true tip)
    if draw_error_lines:
        tips_pred = X + Y_pred * vector_scale
        tips_true = X + Y_true * vector_scale
        for a, b in zip(tips_pred, tips_true):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=error_color, alpha=error_alpha, linewidth=1)

    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", framealpha=0.85)
    if show:
        plt.show()
    return ax

