import numpy as np
import cv2  
from non_stationary_models.forward_rbf import SequentialRBF


class ForwardRBFEnsemble:
    """
    Spatio-temporal forward motion field using an ENSEMBLE of LOCAL RBF interpolators.

    Trains from a single tensor of positions:
        data: (T, N, 2)  where data[t, i] and data[t+1, i] are the same track
    Learns the forward displacement:
        Y_forw(t, i) = data[t+1, i] - data[t, i]

    After fitting, you can predict forward flow u_fwd(x, y, t),
    take forward steps, roll out to a future time, or build cv2.remap maps to correct the image.

    Parameters
    ----------
    n_models : int
        Number of RBF models in the ensemble.
    kernel : str or list
        RBF kernel(s) to use. Can be a single kernel or list of kernels.
    neighbors : int or list
        Number of neighbors for each model. Can be a single value or list.
    smoothing : float or list
        Smoothing parameter for each model. Can be a single value or list.
    time_weight : float or list
        Time weight for each model. Can be a single value or list.
    epsilon : float or None or list
        RBF shape parameter for kernels that need it. Can be a single value or list.
    knn_eps : int or list
        k for the epsilon heuristic. Can be a single value or list.
    standardize : bool
        If True, standardizes [x, y, t] before building the RBF (recommended).
    dtype : np.dtype
        Working dtype for arrays.
    """

    def __init__(self,
                 n_models=5,
                 kernel="thin_plate_spline",
                 neighbors=80,
                 smoothing=1e-3,
                 time_weight=1.0,
                 standardize=False):
        
        self.n_models = n_models
        self.kernel = kernel
        self.neighbors = neighbors
        self.smoothing = smoothing
        self.time_weight = time_weight
        self.standardize = bool(standardize)

        # Create ensemble of RBF models
        self.models = []
        self._create_ensemble()

        # For reference/inspection
        self.T_ = None
        self.N_ = None
        self.t0_ = None
        self.dt_ = None

    def _create_ensemble(self):
        """Create diverse ensemble of RBF models."""
        # Default configurations for diversity - using only kernels that don't need epsilon
        kernels = ["thin_plate_spline", "linear", "cubic", "quintic"]
        neighbor_counts = [60, 80, 100, 120]
        smoothing_values = [1e-3, 1e-2, 1e-1]
        
        # Convert single values to lists
        if isinstance(self.kernel, str):
            kernels = [self.kernel] + [k for k in kernels if k != self.kernel]
        if isinstance(self.neighbors, int):
            neighbor_counts = [self.neighbors] + [n for n in neighbor_counts if n != self.neighbors]
        if isinstance(self.smoothing, (int, float)):
            smoothing_values = [self.smoothing] + [s for s in smoothing_values if s != self.smoothing]
        if isinstance(self.time_weight, (int, float)):
            time_weights = [self.time_weight]
        else:
            time_weights = [self.time_weight]
        
        # Generate all possible configurations
        model_configs = []
        for kernel in kernels[:3]:  # Use top 3 kernels
            for neighbors in neighbor_counts[:3]:  # Use top 3 neighbor counts
                for smoothing in smoothing_values[:3]:  # Use top 3 smoothing values
                    for time_weight in time_weights:
                        model_configs.append({
                            'kernel': kernel,
                            'neighbors': neighbors,
                            'smoothing': smoothing,
                            'time_weight': time_weight,
                            'standardize': self.standardize
                        })
        
        # Randomly select n_models configurations
        if len(model_configs) > self.n_models:
            selected_indices = np.random.choice(len(model_configs), self.n_models, replace=False)
            selected_configs = [model_configs[i] for i in selected_indices]
        else:
            selected_configs = model_configs
        
        # Create the models
        for i, config in enumerate(selected_configs):
            rbf = SequentialRBF(**config)
            self.models.append(rbf)
            print(f"Created ensemble model {i+1}/{self.n_models}: {config['kernel']}, neighbors={config['neighbors']}, smoothing={config['smoothing']}")

    def fit_tensor(self, data, t0=0.0, dt=1.0, mask=None):
        """Fit all models in the ensemble."""
        print(f"Training ensemble of {len(self.models)} RBF models...")
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}...")
            model.fit(data, t0=t0, dt=dt)  # Using new fit method
        
        # Store metadata
        self.T_ = data.shape[0]
        self.N_ = data.shape[1]
        self.t0_ = float(t0)
        self.dt_ = float(dt)
        
        return self

    def predict(self, Xq, tq):
        """Predict forward displacement u_fwd at query points (Xq, tq) using ensemble average."""
        predictions = []
        valid_models = 0
        
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(Xq, tq)
                predictions.append(pred)
                valid_models += 1
            except Exception as e:
                print(f"Warning: Model {i+1} failed in predict: {e}")
                continue
        
        if valid_models == 0:
            print("Error: All models failed in predict! Returning zeros.")
            return np.zeros_like(Xq)
        
        # Average predictions from valid models
        return np.mean(predictions, axis=0)

    def forward_step(self, X, t):
        """One forward step using ensemble average."""
        return self.predict(X, t)  # Using predict method directly

    def rollout(self, X0, t_end, dt=1.0, batch=200_000):    
        """Roll positions from t=0 forward to time t_end using ensemble."""
        X = np.asarray(X0, dtype=np.float32)  # Use float32 for consistency
        t_start = getattr(self, 't0_', 0.0)
        steps = int(np.round((t_end - t_start) / dt))
        
        print(f"Ensemble rollout: {steps} steps from t={t_start} to t={t_end}")
        
        for s in range(steps):
            t_now = t_start + s * dt
            
            # Use ensemble forward step
            chunks = []
            for i in range(0, len(X), batch):
                Xi = X[i:i+batch]
                Xi_next = self.forward_step(Xi, t_now)
                chunks.append(Xi_next)
            
            X = np.vstack(chunks)
            
            # Progress indicator
            if s % 10 == 0:
                print(f"  Step {s}/{steps} (t={t_now:.1f})")
        
        return X

    def build_forward_remap(self, width, height, t_end, bounds=None, dt=1.0, batch=200_000):
        """
        Build cv2.remap maps that send frame-0 pixels to their source coords
        in the image at time t_end using FORWARD rollout with ensemble.
        """
        if bounds is None:
            xmin, xmax, ymin, ymax = 0.0, width - 1.0, 0.0, height - 1.0
        else:
            xmin, xmax, ymin, ymax = bounds

        xs = np.linspace(xmin, xmax, width)
        ys = np.linspace(ymin, ymax, height)
        XX, YY = np.meshgrid(xs, ys)
        X0 = np.column_stack([XX.ravel(), YY.ravel()])

        Xt = self.rollout(X0, t_end=t_end, dt=dt, batch=batch)
        map_x = Xt[:, 0].reshape(height, width).astype(np.float32)
        map_y = Xt[:, 1].reshape(height, width).astype(np.float32)
        return map_x, map_y

    @staticmethod
    def warp_image_to_frame0(I_t, map_x, map_y, border_value=0):
        """Pull colors from I_t at (map_x, map_y) into frame-0 grid."""
        return cv2.remap(I_t, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=border_value)