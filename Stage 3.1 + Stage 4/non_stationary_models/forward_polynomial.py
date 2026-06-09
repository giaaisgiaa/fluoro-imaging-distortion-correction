from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


class PolyDisplacementModel:
    """
    Predicts displacement (dx, dy) from features [x, y, t] using polynomial regression.
    """
    def __init__(self, degree: int = 3):
        self.degree = degree
        self.models = [
            Pipeline([
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                ("linreg", LinearRegression())
            ]),
            Pipeline([
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                ("linreg", LinearRegression())
            ])
        ]

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.models[0].fit(X, Y[:, 0])
        self.models[1].fit(X, Y[:, 1])
        return self

    def predict_delta(self, X_raw: np.ndarray) -> np.ndarray:
        dx = self.models[0].predict(X_raw)
        dy = self.models[1].predict(X_raw)
        return np.column_stack([dx, dy])

    def rollout(self, x0y0: np.ndarray, T: int) -> np.ndarray:
        traj = np.zeros((T, 2), dtype=np.float64)
        traj[0] = np.asarray(x0y0, dtype=np.float64)
        for t in range(T - 1):
            feats = np.array([[traj[t, 0], traj[t, 1], float(t)]], dtype=np.float64)
            dxy = self.predict_delta(feats)[0]
            traj[t + 1] = traj[t] + dxy
        return traj
