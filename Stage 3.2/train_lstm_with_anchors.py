# python train_lstm_with_anchors.py --npz data/dataset.npz --val_frac 0.1 --window 25 --epochs 50 --normalize

"""
LSTM next-step trainer with anchors.

Input step:   [t, elapsed_t, x_t, y_t, x0, y0]
Target step:  [x_{t+1}, y_{t+1}]

We add:
  - x0, y0   : first observed position of this object
  - elapsed_t: t - t0 (time since start)

Validation split is by object IDs (held-out objects).

This version also creates a unique folder under runs/ and saves:
  - model.pt (state_dict)
  - config.json (args + env info)
  - splits.json (train_ids, val_ids)
  - norm_mean.npy / norm_std.npy (if --normalize)
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# --------------------------- Utilities ---------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numpy_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    mse = np.mean(np.sum(diff * diff, axis=1))
    return float(np.sqrt(mse))


def make_run_dir(run_root: str, run_name: str | None = None) -> Path:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    if run_name:
        run_dir = root / run_name
        # ensure uniqueness if name already exists
        if run_dir.exists():
            i = 1
            while (root / f"{run_name}-{i}").exists():
                i += 1
            run_dir = root / f"{run_name}-{i}"
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = root / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# --------------------------- Data prep ---------------------------

def build_seq_pairs_with_anchors(XY: np.ndarray, frame_index: np.ndarray, object_ids) -> dict:
    """
    For each object, produce arrays of consecutive input/target pairs:

      XY.shape = (T, N, 2)

      input[k]  = [t_k, elapsed_k, x_k, y_k, x0, y0]
      target[k] = [x_{k+1}, y_{k+1}]
      times[k]  = t_{k+1}

    Returns:
      {obj_id: {"inputs": (S,6), "targets": (S,2), "times": (S,)}}
    """
    obj_ids_str = np.array([str(x) for x in object_ids], dtype=object)
    T, N, _ = XY.shape
    out = {}

    for j, obj_id in enumerate(obj_ids_str):
        track = XY[:, j, :]  # (T, 2)
        valid_rows = np.where(~np.isnan(track[:, 0]))[0]
        if len(valid_rows) < 2:
            continue
         
        t0 = int(frame_index[valid_rows[0]])
        x0, y0 = track[valid_rows[0]]

        in_list, tgt_list, time_list = [], [], []
        for idx in range(len(valid_rows) - 1):
            i_curr = valid_rows[idx]
            i_next = valid_rows[idx + 1]

            t_curr = int(frame_index[i_curr])
            elapsed = t_curr - t0
            x_curr, y_curr = track[i_curr]

            x_next, y_next = track[i_next]
            t_next = int(frame_index[i_next])

            in_list.append([t_curr, elapsed, x_curr, y_curr, x0, y0])       # features X = [t, elapsed_t, x_t, y_t, x0, y0]
            tgt_list.append([x_next, y_next])                              # target y = [x_{t+1}, y_{t+1}]
            time_list.append(t_next)

        out[obj_id] = {
            "inputs": np.asarray(in_list, dtype=np.float32),
            "targets": np.asarray(tgt_list, dtype=np.float32),
            "times": np.asarray(time_list, dtype=np.int32),
        }

    return out


class Seq2SeqWindowDataset(Dataset):
    """
    Builds fixed-length sliding windows of pairs per object.

    For window W, each sample is:
      X: (W, 6)   -> inputs
      y: (W, 2)   -> targets
      tm: (W,)    -> target timestamps
      meta: obj_id
    """

    def __init__(self, pairs_by_obj: dict, selected_ids: set, window: int):
        self.samples = []
        self.window = int(window)

        for obj_id, d in pairs_by_obj.items():
            if obj_id not in selected_ids:
                continue
            X = d["inputs"]   # (S, 6)
            Y = d["targets"]  # (S, 2)
            Tm = d["times"]   # (S,)
            S = X.shape[0]
            if S < self.window:
                continue
            for end in range(self.window, S + 1):
                start = end - self.window
                self.samples.append((
                    X[start:end, :].copy(),
                    Y[start:end, :].copy(),
                    Tm[start:end].copy(),
                    obj_id
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, Y, Tm, obj_id = self.samples[idx]
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Tm), obj_id


# --------------------------- Model ---------------------------

class LSTMSeq2Seq(nn.Module):
    """
    LSTM that outputs a 2D prediction at every timestep.

    Input:  (B, W, 6)
    Output: (B, W, 2)
    """

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
        out, _ = self.lstm(x)     # (B, W, H)
        pred = self.head(out)     # (B, W, 2)
        return pred


# --------------------------- Training (UNCHANGED) + Eval + Saving ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/dataset.npz", help="Path to dataset.npz")
    ap.add_argument("--val_frac", type=float, default=0.1, help="Fraction of object IDs for validation")
    ap.add_argument("--window", type=int, default=25, help="Sliding window length")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--normalize", action="store_true",
                    help="Z-score normalize features using train stats.")
    # run saving
    ap.add_argument("--run_root", default="runs", help="Root folder where runs are stored")
    ap.add_argument("--run_name", default=None, help="Optional name for the run folder (else timestamped)")
    args = ap.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Device:", device)

    # Load dataset
    data = np.load(args.npz, allow_pickle=True)
    XY = data["XY"]
    frame_index = data["frame_index"]
    object_ids = data["object_ids"]
    obj_ids_str = np.array([str(x) for x in object_ids], dtype=object)

    # Build input/target pairs with anchors
    pairs_by_obj = build_seq_pairs_with_anchors(XY, frame_index, object_ids)

    # Split IDs
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(obj_ids_str)
    n_val = max(1, int(len(obj_ids_str) * args.val_frac)) if len(obj_ids_str) > 1 else 1
    val_ids = set(perm[:n_val])
    train_ids = set(perm[n_val:])
    print(f"Validation IDs ({len(val_ids)}): {sorted(list(val_ids))}")

    # Datasets
    train_ds = Seq2SeqWindowDataset(pairs_by_obj, train_ids, args.window)
    val_ds   = Seq2SeqWindowDataset(pairs_by_obj, val_ids,   args.window)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("Not enough windows to train/validate. Reduce --window or check your data.")
        return

    # Normalization (z-score)
    mean = None
    std = None
    if args.normalize:
        train_inputs_all = []
        for X, _, _, _ in train_ds:
            train_inputs_all.append(X.numpy())
        train_inputs_all = np.vstack(train_inputs_all)
        mean = train_inputs_all.mean(axis=0)
        std = train_inputs_all.std(axis=0)
        std[std == 0.0] = 1.0

        def norm_fn(x): return (x - mean) / std
        print("Normalization enabled.")
    else:
        norm_fn = lambda x: x

    def collate_fn(batch):
        X = torch.stack([torch.from_numpy(norm_fn(b[0].numpy())) for b in batch], dim=0)
        Y = torch.stack([b[1] for b in batch], dim=0)
        Tm = torch.stack([b[2] for b in batch], dim=0)
        meta = [b[3] for b in batch]
        return X.float(), Y.float(), Tm.int(), meta

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = LSTMSeq2Seq(input_size=6, hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ===== TRAINING (UNCHANGED) =====
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        nsamp = 0
        for Xw, Yw, _, _ in train_loader:
            Xw, Yw = Xw.to(device), Yw.to(device)
            optimizer.zero_grad()
            pred = model(Xw)
            loss = criterion(pred, Yw)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * Xw.size(0)
            nsamp += Xw.size(0)
        avg_train_loss = total_loss / max(1, nsamp)

        # Validation (teacher-forced windows)
        model.eval()
        preds, trues, times = [], [], []
        with torch.no_grad():
            for Xw, Yw, Tm, _ in val_loader:
                Xw = Xw.to(device)
                Pw = model(Xw).cpu().numpy()
                preds.append(Pw)
                trues.append(Yw.numpy())
                times.append(Tm.numpy())
        y_pred = np.concatenate([p.reshape(-1, 2) for p in preds], axis=0)
        y_true = np.concatenate([t.reshape(-1, 2) for t in trues], axis=0)
        t_meta = np.concatenate([tm.reshape(-1) for tm in times], axis=0)

        rmse = numpy_rmse(y_true, y_pred)
        print(f"Epoch {epoch:03d} | Train MSE: {avg_train_loss:.6f} | Val RMSE: {rmse:.6f}")

    # Teacher-forced RMSE by time (kept)
    errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    by_time = {}
    for t_val, e in zip(t_meta.tolist(), errors.tolist()):
        by_time.setdefault(int(t_val), []).append(float(e))
    print("\nRMSE by time (validation, teacher-forced windows):")
    for t in sorted(by_time):
        print(f"t={t:4d} | RMSE={np.mean(by_time[t]):.6f}")

    # ===== AUTOREGRESSIVE ROLLOUT EVALUATION (no training changes) =====
    def rollout_evaluate_lstm(pairs_by_obj: dict, val_ids: set, model: nn.Module,
                              window: int, norm_fn, device: torch.device):
        """
        Free-run per validation object:
          - start from the first TRUE input row (k=0) -> predict (k+1)
          - for k>=1, replace x_k,y_k in the input with your PREVIOUS prediction
          - feed last <= window steps to the model, take the LAST timestep's prediction
        Returns: y_true_all, y_pred_all, t_meta_all
        """
        y_true_list, y_pred_list, t_meta_list = [], [], []
        model.eval()
        with torch.no_grad():
            for obj_id, d in pairs_by_obj.items():
                if obj_id not in val_ids:
                    continue
                inputs = d["inputs"]   # (S, 6)
                targets = d["targets"] # (S, 2)
                times   = d["times"]   # (S,)
                S = inputs.shape[0]
                if S == 0:
                    continue

                buf = []
                current_xy = inputs[0, 2:4].astype(np.float32).copy()

                for k in range(S):
                    row = inputs[k].copy()
                    row[2:4] = current_xy
                    buf.append(row)

                    seq = np.stack(buf[-window:], axis=0) if window > 0 else np.stack(buf, axis=0)
                    seq = norm_fn(seq) if norm_fn is not None else seq
                    x_tensor = torch.from_numpy(seq).unsqueeze(0).float().to(device)
                    pred_seq = model(x_tensor).cpu().numpy()
                    pred = pred_seq[0, -1, :]

                    y_pred_list.append(pred.tolist())
                    y_true_list.append(targets[k].tolist())
                    t_meta_list.append(int(times[k]))

                    current_xy = pred.astype(np.float32)

        if not y_true_list:
            return (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty((0,), dtype=int))
        return (np.asarray(y_true_list, np.float32),
                np.asarray(y_pred_list, np.float32),
                np.asarray(t_meta_list, dtype=int))

    y_true_ro, y_pred_ro, t_meta_ro = rollout_evaluate_lstm(
        pairs_by_obj=pairs_by_obj,
        val_ids=val_ids,
        model=model,
        window=args.window,
        norm_fn=norm_fn,
        device=device
    )

    if y_true_ro.size == 0:
        print("\n[Rollout] No rollout samples produced (tracks too short?).")
        return

    overall_rmse_ro = numpy_rmse(y_true_ro, y_pred_ro)
    print(f"\n[Rollout] Validation RMSE (overall): {overall_rmse_ro:.6f}")

    errors_ro = np.sqrt(np.sum((y_true_ro - y_pred_ro) ** 2, axis=1))
    by_time_ro = {}
    for t_val, e in zip(t_meta_ro.tolist(), errors_ro.tolist()):
        by_time_ro.setdefault(int(t_val), []).append(float(e))

    print("\n[Rollout] RMSE by time (validation, autoregressive):")
    for t in sorted(by_time_ro):
        print(f"t={t:4d} | RMSE={np.mean(by_time_ro[t]):.6f}")

    # ===== SAVE RUN ARTEFACTS (no impact on training) =====
    run_dir = make_run_dir(args.run_root, args.run_name)
    print(f"\nSaving run artefacts to: {run_dir}")

    # 1) model weights (state_dict)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "input_size": 6,
                "hidden": args.hidden,
                "layers": args.layers,
                "dropout": args.dropout,
            },
        },
        run_dir / "model.pt",
    )

    # 2) config (args + brief info)
    cfg = {
        "npz": args.npz,
        "val_frac": args.val_frac,
        "window": args.window,
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "normalize": bool(args.normalize),
        "object_count": len(obj_ids_str),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # 3) splits
    splits = {
        "train_ids": sorted(list(map(str, train_ids))),
        "val_ids": sorted(list(map(str, val_ids))),
    }
    with open(run_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    # 4) normalization stats (if any)
    if args.normalize and mean is not None and std is not None:
        np.save(run_dir / "norm_mean.npy", mean.astype(np.float32))
        np.save(run_dir / "norm_std.npy", std.astype(np.float32))

    print("Saved: model.pt, config.json, splits.json", "(+ norm_mean.npy, norm_std.npy)" if args.normalize else "")
    print("Done.")


if __name__ == "__main__":
    main()
