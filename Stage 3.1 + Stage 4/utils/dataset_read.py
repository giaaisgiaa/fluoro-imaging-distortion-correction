import json
from pathlib import Path
from typing import Dict, Union
import numpy as np

def load_frame_datasets(json_source: Union[str, Path, dict]) -> Dict[str, np.ndarray]:
    """
    Read your JSON and return a dict:
        { "frame 1": X, ... }
    where X is a numpy array with shape (N, 2) and dtype float32 containing coordinates.
    """
    # Load the JSON
    if isinstance(json_source, (str, Path)):
        with open(json_source, "r") as f:
            data = json.load(f)
    elif isinstance(json_source, dict):
        data = json_source
    else:
        raise TypeError("json_source must be a filepath or a dict already loaded from JSON.")

    datasets: Dict[str, np.ndarray] = {}

    for frame_name, items in data.items():
        # items: {"0": {...}, "1": {...}, ...}
        idx_keys = sorted(items.keys(), key=lambda s: int(s) if s.isdigit() else s)

        coords_list = []

        for k in idx_keys:
            entry = items[k]
            try:
                coords = entry["coordinates"][:2]
            except KeyError as e:
                raise KeyError(f"Missing key {e} in frame '{frame_name}', index '{k}'")

            coords_list.append(coords)

        X = np.asarray(coords_list, dtype=np.float32)

        if X.shape[1] != 2:
            raise ValueError(f"Coordinates must be length 2; got shape X={X.shape} in '{frame_name}'")

        datasets[frame_name] = X

    return datasets


# (Optional) helper to concatenate all frames into one big dataset
def stack_all_frames(datasets: Dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack(list(datasets.values()))
