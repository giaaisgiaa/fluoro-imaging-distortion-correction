import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import RBFInterpolator


class SequentialRBF:
    """
    RBF model for sequential prediction: (x_t, y_t, t) -> (dx, dy)
    
    Predicts the displacement from t to t+1 given the current position and time.
    Uses local RBF interpolation with thin plate spline kernel.
    """
    
    def __init__(self,
                 kernel="thin_plate_spline",
                 neighbors=80,
                 smoothing=1e-3,
                 time_weight=1.0,
                 standardize=False):
        self.kernel = kernel
        self.neighbors = int(neighbors)
        self.smoothing = float(smoothing)
        self.time_weight = float(time_weight)
        self.standardize = bool(standardize)
        
        self.scaler_X = StandardScaler() if standardize else None
        self.scaler_t = StandardScaler() if standardize else None
        self.scaler_Y = StandardScaler() if standardize else None
        
        self._rbf = None
        self._fitted = False
        
    def fit(self, data, t0=0.0, dt=1.0):
        """
        Fit model from a (T, N, 2) tensor of positions.
        Each data[t,i] is the position of dot i at time t. 
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 3 or data.shape[2] != 2:
            raise ValueError(f"data must be (T, N, 2), got {data.shape}")
        T, N = data.shape[:2]
        if T < 2:
            raise ValueError("Need T >= 2 to compute forward positions")
            
        # Create training pairs (x_t, y_t, t) -> (dx, dy)
        X_t = data[:-1]  # Current positions (T-1, N, 2)
        X_t1 = data[1:]  # Next positions (T-1, N, 2)
        Y_forw = X_t1 - X_t  # Displacements (T-1, N, 2)
        
        # Flatten to 2D arrays
        X_curr = X_t.reshape(-1, 2)  # (T-1)*N, 2)
        Y_disp = Y_forw.reshape(-1, 2)  # (T-1)*N, 2)
        
        # Add time dimension
        times = t0 + dt * np.arange(T-1, dtype=np.float32)  # (T-1,)
        t_all = np.repeat(times, N).reshape(-1, 1)  # ((T-1)*N, 1)
        
        # Stack features
        if self.standardize:
            X_curr = self.scaler_X.fit_transform(X_curr)
            t_all = self.scaler_t.fit_transform(t_all)
            Y_disp = self.scaler_Y.fit_transform(Y_disp)
            
        X_features = np.column_stack([X_curr, t_all * self.time_weight])
        
        # Fit RBF interpolator
        self._rbf = RBFInterpolator(
            X_features, Y_disp,  # Now predicting displacements
            kernel=self.kernel,
            neighbors=self.neighbors,
            smoothing=self.smoothing
        )
        self._fitted = True
        return self
        
    def predict(self, X_curr, t):
        """
        Predict next positions given current positions X_curr at time t.
        
        Parameters
        ----------
        X_curr : array (N, 2)
            Current positions
        t : float or array (N,)
            Current time(s)
        
        Returns
        -------
        X_next : array (N, 2)
            Predicted positions at t+1
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")
            
        X_curr = np.asarray(X_curr, dtype=np.float32)
        if X_curr.ndim != 2 or X_curr.shape[1] != 2:
            raise ValueError(f"X_curr must be (N, 2), got {X_curr.shape}")
            
        # Handle scalar time input
        t = np.asarray(t, dtype=np.float32).reshape(-1)
        if t.size == 1:
            t = np.full(len(X_curr), t.item(), dtype=np.float32)
        t = t.reshape(-1, 1)
        
        # Standardize if needed
        X_curr_std = X_curr
        t_std = t
        if self.standardize:
            X_curr_std = self.scaler_X.transform(X_curr)
            t_std = self.scaler_t.transform(t)
            
        # Make prediction
        X_features = np.column_stack([X_curr_std, t_std * self.time_weight])
        Y_disp = self._rbf(X_features)  # Predicted displacements
        
        # Scaling back the predicted displacements
        if self.standardize:
            Y_disp = self.scaler_Y.inverse_transform(Y_disp)
            
        # Return next position by adding scaled back displacement 
        # to current unscaledposition. This ensures physical consistency :)
        return X_curr + Y_disp
