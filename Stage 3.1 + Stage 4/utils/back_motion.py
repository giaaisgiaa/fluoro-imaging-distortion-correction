import numpy as np
import cv2

def sample_random_convex_points(X_base, n_points=5000, k=4, seed=0):
    """
    Random convex combinations of 'k' points from X_base (n_base,2).
    Returns (n_points,2) points inside the convex hull of X_base.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(X_base), size=(n_points, k))
    W = rng.dirichlet(np.ones(k), size=n_points)              # rows sum to 1
    pts = (X_base[idx] * W[..., None]).sum(axis=1)            # (n_points,k,2) -> (n_points,2)
    return pts

def make_backstep_video_random(
    st,                      # fitted SpatioTemporalRBF with .backstep(X, t)
    X_all, t_all,            # stacked arrays over all frames
    out_path="random_backstep.mp4",
    n_points=4000,           # how many random points to animate
    k=4,                     # how many anchors per convex combo
    to_time=0.0,             # propagate back until this time
    fps=30, width=960, height=960,
    point_radius=1,
    bg_color=(0, 0, 0), point_color=(255, 255, 255),
    draw_trails=False, trail_len=8,
    batch=20000,             # predict in chunks if many points
    seed=0
):
    """
    Builds random points at the last time (max frame) via convex combos,
    backsteps them to time 'to_time' (inclusive), and makes a video.
    """
    X_all = np.asarray(X_all, float)
    t_all = np.asarray(t_all, float)
    t_max = float(np.max(t_all))

    # 1) take all dataset points at the last frame
    X_last = X_all[t_all == t_max]
    if len(X_last) == 0:
        raise ValueError("No points found at the last time step.")

    # 2) sample random convex-combo points inside that frame's convex hull
    Xk = sample_random_convex_points(X_last, n_points=n_points, k=k, seed=seed)

    # 3) bounds from the whole dataset for consistent rendering
    xmin, ymin = X_all.min(axis=0)
    xmax, ymax = X_all.max(axis=0)
    pad_x = 0.02 * (xmax - xmin + 1e-12)
    pad_y = 0.02 * (ymax - ymin + 1e-12)
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    def to_pixels(P):
        u = (P[:, 0] - xmin) / max(1e-12, (xmax - xmin))
        v = (P[:, 1] - ymin) / max(1e-12, (ymax - ymin))
        px = (u * (width  - 1)).astype(np.int32)
        py = ((1.0 - v) * (height - 1)).astype(np.int32)  # flip y for image coords
        return np.column_stack([px, py])

    # 4) video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot open VideoWriter; check codec/path.")

    # times: t_max, t_max-1, ..., to_time
    n_steps = int(np.floor(t_max - to_time)) + 1
    times = t_max - np.arange(n_steps)

    # optional trails
    if draw_trails:
        trail_buf = [Xk.copy()]

    for i, t_now in enumerate(times):
        # 5) draw current positions
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        if draw_trails and len(trail_buf) > 1:
            for tb in trail_buf[-trail_len:]:
                seg = to_pixels(tb)
                for (px, py) in seg:
                    if 0 <= px < width and 0 <= py < height:
                        frame[py, px] = (frame[py, px] * 0.6 + np.array(point_color) * 0.4).astype(np.uint8)

        pts = to_pixels(Xk)
        for (px, py) in pts:
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(frame, (px, py), point_radius, point_color, thickness=-1, lineType=cv2.LINE_AA)

        cv2.putText(frame, f"t = {t_now:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)

        writer.write(frame)

        # 6) backstep to previous time (skip if we're at 'to_time')
        if i < len(times) - 1:
            # chunked prediction for large n_points
            new_positions = []
            for j in range(0, len(Xk), batch):
                Xi = Xk[j:j+batch]
                ti = np.full(len(Xi), t_now, dtype=np.float32)
                Xi_prev = st.backstep(Xi, ti)      # x_{t-1} = x_t - u_back(x_t, t_now)
                new_positions.append(Xi_prev)
            Xk = np.vstack(new_positions)
            if draw_trails:
                trail_buf.append(Xk.copy())
                if len(trail_buf) > trail_len:
                    trail_buf.pop(0)

    writer.release()
    print(f"Saved video to: {out_path}")


# -----------------------------
# Example usage
# -----------------------------
# Assumptions:
#   - You already stacked your data: X_all, t_all, Yback_all
#   - You already fit the model: st = SpatioTemporalRBF(...).fit(X_all, t_all, Yback_all)

# make_backstep_video_random(
#     st, X_all, t_all,
#     out_path="random_backstep.mp4",
#     n_points=5000, k=4,
#     to_time=0.0,
#     fps=30, width=960, height=960,
#     point_radius=1,
#     draw_trails=True, trail_len=8,
#     batch=20000,
#     seed=42
# )
