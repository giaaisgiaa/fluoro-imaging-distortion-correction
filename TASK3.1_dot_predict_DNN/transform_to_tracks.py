
"""
Transform nested frame->object->fields data into Option B (track-centric) format.

Input schema (example):
{
  "frame 1": {
    "0": {"coordinates": [x, y], "correction": [...], "movement_back": [...]},
    "1": {"coordinates": [],     "correction": [...], "movement_back": [...]}
  },
  "frame 2": {
    "0": {"coordinates": [x, y], ...},
    "2": {"coordinates": [x, y], ...}
  }
}

Output schema:
{
  "version": "1.0",
  "schema": "tracks",
  "object_ids": ["0", "1", "2"],
  "tracks": {
    "0": [{"t": 0, "xy": [x, y]}, {"t": 1, "xy": [x, y]}],
    "1": [{"t": 2, "xy": [x, y]}],
    "2": [...]
  }
}
"""

from __future__ import annotations
import json
import math
import re
from typing import Any, Dict, List, Tuple, Iterable
from collections import defaultdict
import os

try:
    import numpy as np
except ImportError:
    np = None  # Only needed if --npz is requested


def parse_frame_index(frame_key: str) -> int:
    """
    Turn keys like 'frame 1' into 0-based ints (i.e., 0).
    If multiple integers appear, use the last one. If none appear, try casting the whole key.
    """
    m = re.findall(r"\d+", frame_key)
    if m:
        return int(m[-1]) - 1
    return int(frame_key) - 1


def transform_to_tracks(raw: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Transform raw nested dict into Option B track-centric JSON object.

    - Frames become 0-based integers t
    - Only coordinates are kept, under 'xy'
    - Missing/empty coordinates are skipped
    - Per-object lists are sorted by t
    """
    # 1) Normalize frames to 0-based and sort
    frames: List[Tuple[int, Dict[str, Dict[str, Any]]]] = []
    for fkey, content in raw.items():
        t = parse_frame_index(fkey)
        frames.append((t, content or {}))
    frames.sort(key=lambda kv: kv[0])

    # 2) Collect tracks
    tracks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_obj_ids = set()

    for t, objects in frames:
        for obj_id, payload in (objects or {}).items():
            all_obj_ids.add(str(obj_id))
            coords = payload.get("coordinates", None)
            if not coords or len(coords) < 2:
                # skip empty or invalid coordinates
                continue
            x, y = coords[0], coords[1]
            # basic numeric sanity
            if x is None or y is None:
                continue
            # keep only finite numbers
            try:
                xf, yf = float(x), float(y)
            except Exception:
                continue
            if not (math.isfinite(xf) and math.isfinite(yf)):
                continue

            tracks[str(obj_id)].append({"t": int(t), "xy": [xf, yf]})

    # 3) Sort each object's list by t
    for obj_id in tracks:
        tracks[obj_id].sort(key=lambda e: e["t"])

    # 4) Build object_ids stable order (sorted by string id)
    object_ids = sorted(all_obj_ids, key=lambda s: (len(s), s))  # stable, human-ish order

    # 5) Final JSON object
    out = {
        "version": "1.0",
        "schema": "tracks",
        "object_ids": object_ids,
        "tracks": tracks,  # dict of lists
    }
    return out


def export_npz_from_tracks(tracks_json: Dict[str, Any], out_path: str) -> None:
    """
    Optional: write NPZ with fast-to-load arrays for ML.
    Produces:
      - frame_index: (T,) int
      - object_ids: (N,) str (saved via numpy object array)
      - XY: (T, N, 2) float with NaN for missing
    """
    if np is None:
        raise RuntimeError("NumPy is required for --npz export. Please `pip install numpy`.")

    tracks: Dict[str, List[Dict[str, Any]]] = tracks_json["tracks"]
    object_ids: List[str] = tracks_json["object_ids"]

    # Collect all frame indices present across all objects
    frame_set = set()
    for obj_id in object_ids:
        for e in tracks.get(obj_id, []):
            frame_set.add(e["t"])
    if not frame_set:
        # Empty dataset
        np.savez_compressed(out_path, frame_index=np.array([], dtype=int),
                            object_ids=np.array(object_ids, dtype=object),
                            XY=np.empty((0, len(object_ids), 2), dtype=float))
        return

    frame_index = np.array(sorted(frame_set), dtype=int)
    t_to_pos = {t: i for i, t in enumerate(frame_index)}

    N = len(object_ids)
    T = len(frame_index)
    XY = np.full((T, N, 2), np.nan, dtype=float)

    # Fill matrix
    for j, obj_id in enumerate(object_ids):
        for e in tracks.get(obj_id, []):
            i = t_to_pos[e["t"]]
            x, y = e["xy"]
            XY[i, j, 0] = x
            XY[i, j, 1] = y

    np.savez_compressed(out_path, frame_index=frame_index,
                        object_ids=np.array(object_ids, dtype=object),
                        XY=XY)


def main() -> None:
   
    input_json = "data/dot_trajectories.json"
    output_json = "data/tracks.json"
    npz_path = "data/dataset.npz"

    # Load input
    with open(input_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Transform
    out = transform_to_tracks(raw)

    # Write JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Always generate NPZ for ML
    export_npz_from_tracks(out, npz_path)

    # Summary
    num_objs = len(out["object_ids"])
    num_points = sum(len(v) for v in out["tracks"].values())
    print(f"Wrote: {output_json}")
    print(f"Objects: {num_objs} | Observations: {num_points}")
    print(f"NPZ written: {npz_path}")


if __name__ == "__main__":
    main()
