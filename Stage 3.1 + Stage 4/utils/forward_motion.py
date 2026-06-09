import os
import numpy as np
import cv2
from numpy.linalg import LinAlgError

def _pick_codec(out_path, codec=None):
    if codec: return codec
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".avi":  return "MJPG"   # fast, widely supported in AVI
    if ext == ".mp4":  return "mp4v"   # MP4
    return "mp4v"

def _to_pixels(P, xmin, xmax, ymin, ymax, W, H):
    u = (P[:, 0] - xmin) / max(1e-12, (xmax - xmin))
    v = (P[:, 1] - ymin) / max(1e-12, (ymax - ymin))
    px = (u * (W - 1)).astype(np.int32)
    py = ((1.0 - v) * (H - 1)).astype(np.int32)
    return px, py

def sample_random_convex_points(X_base, n_points=5000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(X_base), size=(n_points, k))
    W = rng.dirichlet(np.ones(k), size=n_points)
    return (X_base[idx] * W[..., None]).sum(axis=1)

def _safe_forward_step(model, Xi, t_now, jitter=0.5):
    tvec = np.full(len(Xi), t_now, dtype=np.float32)
    try:
        return model.forward_step(Xi, tvec)
    except LinAlgError:
        # mix neighbor times to avoid degree-1 polynomial tail rank defect
        n = len(Xi); h = n // 2
        tvec[:h] += jitter; tvec[h:] -= jitter
        return model.forward_step(Xi, tvec)

def make_forward_video_random(
    model, X0, out_path,
    n_points=5000, k=4, seed=42,
    t_start=0.0, t_end=150.0, dt=1.0,
    width=960, height=960, fps=30,
    point_radius=1, bg_color=(0,0,0), point_color=(255,255,255),
    draw_trails=False, trail_len=8,
    codec=None, bounds=None, prepass=True, batch=100_000,
    jitter=0.5
):
    # Clamp to trained time range if the model exposes it (ForwardRBF does)
    if hasattr(model, "t0_") and hasattr(model, "dt_") and hasattr(model, "T_"):
        t0 = float(getattr(model, "t0_", 0.0))
        dtt = float(getattr(model, "dt_", 1.0))
        # forward displacement exists for times t in [t0, t0 + (T-2)*dt]
        t_end_safe = t0 + max(0, (int(model.T_) - 2)) * dtt
        if t_end > t_end_safe:
            t_end = t_end_safe

    X0 = np.asarray(X0, np.float32)
    assert X0.ndim == 2 and X0.shape[1] == 2

    # time grid
    n_frames = int(np.floor((t_end - t_start) / dt)) + 1
    times = t_start + np.arange(n_frames, dtype=np.float64) * dt

    # start cloud at t_start via convex combos of X0
    Xk0 = sample_random_convex_points(X0, n_points=n_points, k=k, seed=seed)

    # bounds (prepass along trajectory if asked)
    if bounds is None and prepass:
        Xtmp = Xk0.copy()
        xmin, ymin = Xtmp.min(axis=0); xmax, ymax = Xtmp.max(axis=0)
        for t_now in times[:-1]:
            tmp = []
            for j in range(0, len(Xtmp), batch):
                tmp.append(_safe_forward_step(model, Xtmp[j:j+batch], float(t_now), jitter=jitter))
            Xtmp = np.vstack(tmp)
            mn = Xtmp.min(axis=0); mx = Xtmp.max(axis=0)
            xmin, ymin = np.minimum([xmin, ymin], mn)
            xmax, ymax = np.maximum([xmax, ymax], mx)
        pad_x = 0.02 * (xmax - xmin + 1e-12); pad_y = 0.02 * (ymax - ymin + 1e-12)
        bounds = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

    if bounds is None:
        xmin, ymin = X0.min(axis=0); xmax, ymax = X0.max(axis=0)
        pad_x = 0.05 * (xmax - xmin + 1e-12); pad_y = 0.05 * (ymax - ymin + 1e-12)
        bounds = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)
    xmin, xmax, ymin, ymax = bounds

    # writer (match codec to extension)
    codec = _pick_codec(out_path, codec)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for '{out_path}' (codec={codec}).")

    Xk = Xk0.copy()
    if draw_trails: trail_buf = [Xk.copy()]

    for i, t_now in enumerate(times):
        # render
        frame = np.empty((height, width, 3), np.uint8); frame[:] = bg_color
        px, py = _to_pixels(Xk, xmin, xmax, ymin, ymax, width, height)
        if point_radius <= 1:
            m = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            frame[py[m], px[m]] = point_color
        else:
            for (x, y) in zip(px, py):
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), point_radius, point_color, -1, cv2.LINE_AA)

        if draw_trails and len(trail_buf) > 1:
            for tb in trail_buf[-trail_len:]:
                px_t, py_t = _to_pixels(tb, xmin, xmax, ymin, ymax, width, height)
                m = (px_t >= 0) & (px_t < width) & (py_t >= 0) & (py_t < height)
                frame[py_t[m], px_t[m]] = (
                    0.6 * frame[py_t[m], px_t[m]] + 0.4 * np.array(point_color, np.uint8)
                ).astype(np.uint8)

        cv2.putText(frame, f"t = {t_now:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
        writer.write(frame)

        if i == len(times) - 1: break

        # forward step (safe, time vector + dither fallback)
        nxt = []
        for j in range(0, len(Xk), batch):
            nxt.append(_safe_forward_step(model, Xk[j:j+batch], float(t_now), jitter=jitter))
        Xk = np.vstack(nxt)

        if draw_trails:
            trail_buf.append(Xk.copy())
            if len(trail_buf) > trail_len: trail_buf.pop(0)

    writer.release()
    print(f"Saved video to: {os.path.abspath(out_path)}")
