#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize LSTM rollout on the validation set from a saved run folder.

Input:
  --run_dir runs/<your_saved_run>    # must contain: model.pt, config.json, splits.json, (norm_mean.npy, norm_std.npy if used)

Output:
  - Creates runs/<run>/viz/rollout_val.mp4 (or .gif fallback) unless --out is provided.
  - Creates runs/<run>/viz/rmse_plot.png with RMSE over time plot.

Plotting:
  - Green dots  : ground-truth validation trajectories
  - Orange dots : rollout (autoregressive) predictions (start from true x0,y0 at t0)
  - Red lines   : per-point error (connect orange -> green)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


# --------------------------- Model (must match training) ---------------------------

class LSTMSeq2Seq(nn.Module):
    """LSTM that outputs a 2D prediction at every timestep."""
    def __init__(self, input_size=6, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)   # (B, T, H)
        return self.head(out)   # (B, T, 2)


# --------------------------- Data utilities (match training prep) ---------------------------

def build_seq_pairs_with_anchors(XY: np.ndarray, frame_index: np.ndarray, object_ids) -> dict:
    """
    For each object, produce arrays of consecutive input/target pairs:
      input[k]  = [t_k, elapsed_k, x_k, y_k, x0, y0]
      target[k] = [x_{k+1}, y_{k+1}]
      times[k]  = t_{k+1}
    Also returns x0,y0,t0 for convenience.
    """
    obj_ids_str = np.array([str(x) for x in object_ids], dtype=object)
    out = {}
    for j, obj_id in enumerate(obj_ids_str):
        track = XY[:, j, :]  # (T, 2)
        valid_rows = np.where(~np.isnan(track[:, 0]))[0]
        if len(valid_rows) < 2:
            continue
        r0 = valid_rows[0]
        t0 = int(frame_index[r0])
        x0, y0 = track[r0]

        in_list, tgt_list, time_list = [], [], []
        for idx in range(len(valid_rows) - 1):
            i_curr = valid_rows[idx]
            i_next = valid_rows[idx + 1]

            t_curr = int(frame_index[i_curr])
            elapsed = t_curr - t0
            x_curr, y_curr = track[i_curr]

            x_next, y_next = track[i_next]
            t_next = int(frame_index[i_next])

            in_list.append([t_curr, elapsed, x_curr, y_curr, x0, y0])
            tgt_list.append([x_next, y_next])
            time_list.append(t_next)

        out[obj_id] = {
            "inputs": np.asarray(in_list, dtype=np.float32),    # (S, 6)
            "targets": np.asarray(tgt_list, dtype=np.float32),  # (S, 2)
            "times": np.asarray(time_list, dtype=np.int32),     # (S,)
            "x0y0t0": (float(x0), float(y0), int(t0)),
        }
    return out


# --------------------------- Rollout (eval-only) ---------------------------

def rollout_sequence(inputs_obj: np.ndarray, window: int, model: nn.Module, device: torch.device,
                     norm_mean: np.ndarray | None, norm_std: np.ndarray | None) -> np.ndarray:
    """
    Autoregressive rollout for a single object.

    inputs_obj: (S,6) rows [t_k, elapsed_k, x_k, y_k, x0, y0]
    Returns predictions array of shape (S,2) aligned with inputs_obj's targets/times.

    Note: We PAD sequences on the LEFT with the first row to reach 'window' length,
    so normalization (if saved as (W,6)) and model input shape always match training.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        buf = []
        # start from true x0,y0 (row 0's x,y)
        current_xy = inputs_obj[0, 2:4].astype(np.float32).copy()

        for k in range(inputs_obj.shape[0]):
            row = inputs_obj[k].copy()
            row[2:4] = current_xy                     # replace x_k,y_k with current (true for k=0, then predicted)
            buf.append(row)

            seq = np.stack(buf, axis=0)               # (L,6), L grows
            # Left-pad to 'window' with the first row (to match training shape)
            if window > 0:
                if seq.shape[0] < window:
                    pad = np.repeat(seq[0:1, :], window - seq.shape[0], axis=0)
                    seq_win = np.vstack([pad, seq])
                else:
                    seq_win = seq[-window:, :]
            else:
                seq_win = seq

            # Normalization if available (mean/std were saved as (W,6))
            if norm_mean is not None and norm_std is not None:
                seq_in = (seq_win - norm_mean) / norm_std
            else:
                seq_in = seq_win

            x_tensor = torch.from_numpy(seq_in).unsqueeze(0).float().to(device)  # (1, W, 6)
            pred_seq = model(x_tensor).cpu().numpy()                              # (1, W, 2)
            pred = pred_seq[0, -1, :]                                             # last step prediction
            preds.append(pred)
            current_xy = pred.astype(np.float32)                                   # feed forward

    return np.asarray(preds, dtype=np.float32)  # (S,2)


# --------------------------- Visualization helpers ---------------------------

def compute_bounds(val_true_by_t: dict[int, list[tuple[float, float]]],
                   val_pred_by_t: dict[int, list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for d in (val_true_by_t, val_pred_by_t):
        for pts in d.values():
            if pts:
                a = np.asarray(pts, dtype=np.float32)
                xs.extend(a[:, 0].tolist())
                ys.extend(a[:, 1].tolist())
    if not xs:  # fallback
        return -1.0, 1.0, -1.0, 1.0
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # add margins
    mx = 0.05 * (x_max - x_min + 1e-6)
    my = 0.05 * (y_max - y_min + 1e-6)
    return x_min - mx, x_max + mx, y_min - my, y_max + my


def plot_rmse_over_time(val_true_by_t: dict[int, list[tuple[float, float]]],
                       val_pred_by_t: dict[int, list[tuple[float, float]]],
                       save_path: Path):
    """Create and save a plot of RMSE over time."""
    times = []
    rmse_values = []
    
    # Calculate RMSE for each timestep
    all_t = sorted(set(val_true_by_t.keys()) | set(val_pred_by_t.keys()))
    for t in all_t:
        true_pts = val_true_by_t.get(t, [])
        pred_pts = val_pred_by_t.get(t, [])
        
        # Only calculate RMSE if we have matching pairs
        if true_pts and pred_pts:
            n = min(len(true_pts), len(pred_pts))
            true_array = np.array(true_pts[:n])
            pred_array = np.array(pred_pts[:n])
            
            # Calculate RMSE
            mse = np.mean(np.sum((true_array - pred_array) ** 2, axis=1))
            rmse = np.sqrt(mse)
            
            times.append(t)
            rmse_values.append(rmse)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, rmse_values, 'b-', label='Validation RMSE')
    plt.fill_between(times, rmse_values, alpha=0.2)
    
    # Add labels and title
    plt.xlabel('Frame Number')
    plt.ylabel('RMSE')
    plt.title('LSTM Validation RMSE Over Time (Autoregressive)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics
    mean_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    min_rmse = np.min(rmse_values)
    max_rmse = np.max(rmse_values)
    
    stats_text = f'Statistics:\nMean RMSE: {mean_rmse:.2f}\nStd Dev: {std_rmse:.2f}\nMin RMSE: {min_rmse:.2f}\nMax RMSE: {max_rmse:.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save and close
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved RMSE plot: {save_path}")

def render_video(run_dir: Path, out_path: Path, fps: int, dpi: int,
                 XY: np.ndarray, frame_index: np.ndarray, object_ids: np.ndarray,
                 val_ids: set[str], model: nn.Module, device: torch.device,
                 window: int, norm_mean: np.ndarray | None, norm_std: np.ndarray | None,
                 limit_objects: int | None = None, t_from: int | None = None, t_to: int | None = None):
    """
    Build per-time dictionaries of ground-truth and rollout predictions, then animate.
    Also creates an RMSE over time plot.
    """
    # Build pairs and rollout per-object
    pairs = build_seq_pairs_with_anchors(XY, frame_index, object_ids)

    # Optionally limit number of validation objects (for quick previews)
    val_list = [oid for oid in pairs.keys() if oid in val_ids]
    if limit_objects is not None:
        val_list = val_list[:limit_objects]

    # Maps: time t -> list of (x,y)
    val_true_by_t: dict[int, list[tuple[float, float]]] = {}
    val_pred_by_t: dict[int, list[tuple[float, float]]] = {}

    # Also include t0 frames (orange==green at start)
    start_frames = []  # list of (t0, (x0,y0))
    for oid in val_list:
        x0, y0, t0 = pairs[oid]["x0y0t0"]
        start_frames.append((t0, (x0, y0)))

    # Rollout each object and collect time-aligned points
    for oid in val_list:
        d = pairs[oid]
        inputs = d["inputs"]    # (S,6)
        targets = d["targets"]  # (S,2)
        times   = d["times"]    # (S,)
        x0, y0, t0 = d["x0y0t0"]

        # (A) Add the start frame with orange==green
        val_true_by_t.setdefault(int(t0), []).append((float(x0), float(y0)))
        val_pred_by_t.setdefault(int(t0), []).append((float(x0), float(y0)))

        # (B) Autoregressive rollout for this object
        preds = rollout_sequence(inputs, window, model, device, norm_mean, norm_std)  # (S,2)

        # Collect ground-truth & predictions at each target time
        for (t, gt, pr) in zip(times.tolist(), targets.tolist(), preds.tolist()):
            val_true_by_t.setdefault(int(t), []).append((float(gt[0]), float(gt[1])))
            val_pred_by_t.setdefault(int(t), []).append((float(pr[0]), float(pr[1])))

    # Build global timeline
    all_times = sorted(set(val_true_by_t.keys()) | set(val_pred_by_t.keys()))
    if t_from is not None:
        all_times = [t for t in all_times if t >= t_from]
    if t_to is not None:
        all_times = [t for t in all_times if t <= t_to]
    if not all_times:
        print("No frames to render. Check t_from/t_to or validation content.")
        return

    # Axis bounds
    x_min, x_max, y_min, y_max = compute_bounds(val_true_by_t, val_pred_by_t)

    # Prepare output folder
    (run_dir / "viz").mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = run_dir / "viz" / "rollout_val.mp4"

    # Matplotlib animation setup
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("LSTM Rollout on Validation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Artists
    scat_true = ax.scatter([], [], s=18, c="#2ca02c", label="Ground Truth")  # green
    scat_pred = ax.scatter([], [], s=18, c="#ff7f0e", label="Rollout Pred")  # orange
    segs = LineCollection([], colors="#d62728", linewidths=1.0, alpha=0.9)   # red
    ax.add_collection(segs)
    ax.legend(loc="upper right")

    def update(frame_idx):
        t = all_times[frame_idx]
        g_pts = val_true_by_t.get(t, [])
        p_pts = val_pred_by_t.get(t, [])

        gx = [p[0] for p in g_pts]; gy = [p[1] for p in g_pts]
        px = [p[0] for p in p_pts]; py = [p[1] for p in p_pts]

        # match lengths for segments (in case of missing preds/gt, pair by index)
        n = min(len(gx), len(px))
        seg_list = [((px[i], py[i]), (gx[i], gy[i])) for i in range(n)]

        # update artists
        if len(gx) > 0:
            scat_true.set_offsets(np.c_[gx, gy])
        else:
            scat_true.set_offsets(np.empty((0, 2)))
        if len(px) > 0:
            scat_pred.set_offsets(np.c_[px, py])
        else:
            scat_pred.set_offsets(np.empty((0, 2)))
        segs.set_segments(seg_list)

        ax.set_title(f"LSTM Rollout on Validation — t={t}  (objects: {n})")
        return scat_true, scat_pred, segs

    anim = FuncAnimation(fig, update, frames=len(all_times), interval=1000 // max(1, fps), blit=False)

    # Create RMSE plot
    rmse_plot_path = out_path.parent / "rmse_plot.png"
    plot_rmse_over_time(val_true_by_t, val_pred_by_t, rmse_plot_path)

    # Save (prefer mp4, fallback to gif)
    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps, metadata={"artist": "viz_lstm_rollout"}, bitrate=1800)
        anim.save(out_path.as_posix(), writer=writer)
        print(f"Saved video: {out_path}")
    except Exception as e:
        print(f"FFmpeg failed ({e}). Falling back to GIF...")
        from matplotlib.animation import PillowWriter
        gif_path = out_path.with_suffix(".gif")
        anim.save(gif_path.as_posix(), writer=PillowWriter(fps=fps))
        print(f"Saved GIF: {gif_path}")


# --------------------------- CLI ---------------------------

def main():
    # List available run folders
    runs_dir = Path("runs")
    run_folders = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    
    if not run_folders:
        print("No run folders found in runs/")
        return
        
    print("\nAvailable run folders:")
    for i, folder in enumerate(run_folders, 1):
        print(f"{i}. {folder.name}")
    
    # Ask user to choose
    while True:
        try:
            choice = int(input("\nChoose a run folder (enter number): "))
            if 1 <= choice <= len(run_folders):
                break
            print(f"Please enter a number between 1 and {len(run_folders)}")
        except ValueError:
            print("Please enter a valid number")
    
    run_dir = run_folders[choice - 1]
    print(f"\nSelected: {run_dir}")
    
    # Fixed parameters
    fps = 12
    dpi = 120
    limit_objects = None  # render all objects
    t_from = None        # start from beginning
    t_to = None          # render until end
    out = None          # use default output path

    # Load config & splits
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(run_dir / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Device (use same heuristic; model is small)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset
    npz_path = cfg["npz"]
    data = np.load(npz_path, allow_pickle=True)
    XY = data["XY"]                    # (T, N, 2)
    frame_index = data["frame_index"]  # (T,)
    object_ids = data["object_ids"]    # (N,)
    val_ids = set(map(str, splits["val_ids"]))
    print(f"Validation IDs ({len(val_ids)}): {sorted(list(val_ids))[:10]}{'...' if len(val_ids)>10 else ''}")

    # Load normalization (if present)
    mean_path = run_dir / "norm_mean.npy"
    std_path  = run_dir / "norm_std.npy"
    if mean_path.exists() and std_path.exists():
        norm_mean = np.load(mean_path)
        norm_std  = np.load(std_path)
        print("Loaded normalization stats.")
    else:
        norm_mean = None
        norm_std  = None
        print("No normalization stats found; using raw inputs.")

    # Build model from saved meta/config and load weights
    # Prefer meta in model.pt; otherwise fallback to config.json
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    meta = ckpt.get("meta", {})
    hidden = int(meta.get("hidden", cfg.get("hidden", 64)))
    layers = int(meta.get("layers", cfg.get("layers", 1)))
    dropout = float(meta.get("dropout", cfg.get("dropout", 0.0)))
    window = int(cfg.get("window", 5))

    model = LSTMSeq2Seq(input_size=6, hidden_size=hidden, num_layers=layers, dropout=dropout).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Model loaded: hidden={hidden}, layers={layers}, dropout={dropout}, window={window}")

    # Prepare output path
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / "rollout_val.mp4"

    # Render
    render_video(run_dir, out_path, fps=fps, dpi=dpi,
                 XY=XY, frame_index=frame_index, object_ids=object_ids,
                 val_ids=val_ids, model=model, device=device, window=window,
                 norm_mean=norm_mean, norm_std=norm_std,
                 limit_objects=limit_objects, t_from=t_from, t_to=t_to)


if __name__ == "__main__":
    main()
