from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Sequence
from sklearn.preprocessing import StandardScaler


@dataclass
class ForwardLinear:
    """
    Predicts displacement (dx, dy) from features [x, y, t] via:
        d = X_std @ W + b
    We default to no intercept so each output head has exactly 3 parameters.
    """
    scaler: StandardScaler
    W: np.ndarray  # shape (3, 2)
    b: np.ndarray  # shape (2,)

    @classmethod
    def fit(
        cls,
        X_train: np.ndarray,  # (M, 3) raw features [x, y, t]
        Y_train: np.ndarray,  # (M, 2) displacements [dx, dy]
        fit_intercept: bool = False,  # keep False to have exactly 3 params per output
    ) -> "ForwardLinear":
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X_train)
        Xs = scaler.transform(X_train)  # standardize using train only
        if fit_intercept:
            Xa = np.hstack([Xs, np.ones((Xs.shape[0], 1))])  # allow a constant drift
            W_aug, *_ = np.linalg.lstsq(Xa, Y_train, rcond=None)  # (4,2)
            W, b = W_aug[:3, :], W_aug[3, :]
        else:
            W, *_ = np.linalg.lstsq(Xs, Y_train, rcond=None)  # (3,2)
            b = np.zeros(2, dtype=np.float64)
        return cls(scaler=scaler, W=W, b=b)

    def predict_delta(self, X_raw: np.ndarray) -> np.ndarray:
        """
        X_raw: (K, 3) in raw coordinates [x, y, t]; returns (K, 2) deltas
        """
        Xs = self.scaler.transform(X_raw)
        return Xs @ self.W + self.b

    def rollout(self, x0y0: np.ndarray, T: int) -> np.ndarray:
        """
        Autoregressive rollout from t=0 to t=T-1.
        Only x0,y0 are given — each next step uses the model's own previous prediction.
        Returns trajectory of shape (T, 2).
        """
        traj = np.zeros((T, 2), dtype=np.float64)
        traj[0] = np.asarray(x0y0, dtype=np.float64)
        for t in range(T - 1):
            feats = np.array([[traj[t, 0], traj[t, 1], float(t)]], dtype=np.float64)  # (1,3)
            dxy = self.predict_delta(feats)[0]  # (2,)
            traj[t + 1] = traj[t] + dxy
        return traj
