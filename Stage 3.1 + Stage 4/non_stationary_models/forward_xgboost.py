from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class XGBDisplacementModel:
    """
    Predicts displacement (dx, dy) from features [x, y, t] using two XGBoost regressors.
    """
    def __init__(self, **xgb_params):
        # Sensible default parameters for regression
        default_params = dict(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0,
        )
        default_params.update(xgb_params)
        self.xgb_params = default_params

        self.scaler = StandardScaler()
        self.models = [
            XGBRegressor(**self.xgb_params),
            XGBRegressor(**self.xgb_params)
        ]

    def fit(self, X: np.ndarray, Y: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.models[0].fit(Xs, Y[:, 0])
        self.models[1].fit(Xs, Y[:, 1])
        return self

    def predict_delta(self, X_raw: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X_raw)
        dx = self.models[0].predict(Xs)
        dy = self.models[1].predict(Xs)
        return np.column_stack([dx, dy])

    def rollout(self, x0y0: np.ndarray, T: int) -> np.ndarray:
        traj = np.zeros((T, 2), dtype=np.float64)
        traj[0] = np.asarray(x0y0, dtype=np.float64)
        for t in range(T - 1):
            feats = np.array([[traj[t, 0], traj[t, 1], float(t)]], dtype=np.float64)
            dxy = self.predict_delta(feats)[0]
            traj[t + 1] = traj[t] + dxy
        return traj
