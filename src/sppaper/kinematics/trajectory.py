import numpy as np
from scipy.signal import savgol_filter


def align_traj(traj, traj_ref, anchor_origin: bool = True):
    """Align two 2D trajectories using a rigid transformation (rotation + translation).

    This function transforms traj1 to best match traj2 by finding the optimal rotation
    and translation that minimizes MSE between the two trajectories.

    Two alignment modes are supported:

        * `anchor_origin=True` (default): traj1 is first translated so its starting
          point coincides with traj2's starting point, and then the optimal rotation
          (pivoted at the shared origin) is found. Translation is therefore fully
          determined by the start pointsand is **not** a free parameter.

        * `anchor_origin=False`: both rotation and translation are free parameters. The
          standard Kabsch algorithm is used: trajectories are centroid-aligned before
          solving for the optimal rotation, and the translation is chosen to minimize
          overall MSE.

    Args:
        traj: (L, 2) array, first trajectory to align.
        traj_ref: (L, 2) array, reference trajectory to align to.
        anchor_origin: If True, pin the start of traj to the start of traj_ref before
            optimizing rotation. If False, translation is a free parameter optimized
            jointly with rotation for best overall fit. Defaults to True.

    Returns:
        A dict with keys:
            "R": (2, 2) rotation matrix.
            "t": (2,) translation vector such that `traj_aligned = traj @ R.T + t`.
            "traj_aligned": transformed traj with shape (L, 2).
            "metrics": dict with keys "rmse", "mae", "median", "max", "normalized_rmse"
    """
    if traj.shape != traj_ref.shape or traj.ndim != 2 or traj.shape[1] != 2:
        raise ValueError("traj and traj_ref must have the same shape (L, 2)")

    if anchor_origin:
        # Anchor to starting point to enforce same start
        traj_0 = traj - traj[0]
        traj_ref_0 = traj_ref - traj_ref[0]
    else:
        # Center by centroid so translation becomes a free parameter
        traj_0 = traj - traj.mean(axis=0)
        traj_ref_0 = traj_ref - traj_ref.mean(axis=0)

    # Kabsch algorithm: find R that minimizes ||traj_0 - traj_ref_0 @ R.T||_F
    H = traj_0.T @ traj_ref_0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = +1, no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Derive translation from the chosen pivot point
    if anchor_origin:
        # Map P[0] exactly onto Q[0]
        t = traj_ref[0] - (R @ traj[0])
    else:
        # Map P's centroid onto Q's centroid
        t = traj_ref.mean(axis=0) - (R @ traj.mean(axis=0))

    # Apply transform
    traj_aligned = (traj @ R.T) + t

    # Residuals and metrics
    residuals = traj_aligned - traj_ref  # (L, 2)
    per_point_err = np.linalg.norm(residuals, axis=1)  # (L,)

    rmse = float(np.sqrt(np.mean(per_point_err**2)))

    # Normalize RMSE by path length to make it scale-free
    diffs = np.diff(traj_ref, axis=0)
    path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    normalized_rmse = float(rmse / path_len) if path_len > 0 else np.nan

    metrics = {"rmse": rmse, "normalized_rmse": normalized_rmse}

    return {"R": R, "t": t, "traj_aligned": traj_aligned, "metrics": metrics}


def window_size_sec2steps(size_sec, dt, ensure_odd=True):
    """Convert window size from seconds to number of steps.

    Args:
        size_sec: window size in seconds.
        dt: time step in seconds.
        ensure_odd: if True, round up to nearest odd number.

    Returns:
        Window size in number of steps (samples).
    """
    size_steps = int(np.round(size_sec / dt))
    if ensure_odd and size_steps % 2 == 0:
        size_steps += 1
    return size_steps


def get_denoised_traj_and_vel(
    traj, dt, sg_window_steps=None, sg_window_sec=None, sg_polyorder=3
):
    """Smooth a 2D trajectory and compute velocity using SG filter.

    Args:
        traj: (L, 2) raw trajectory array.
        dt: time step in seconds.
        sg_window_steps: SG window length in steps.
        sg_window_sec: SG window length in seconds.
        sg_polyorder: polynomial order for SG filter.

    Returns:
        traj_smooth (L, 2), velxy_smooth (L, 2)
    """
    if not ((sg_window_steps is None) ^ (sg_window_sec is None)):
        raise ValueError("Specify window size in steps or seconds, not both.")
    if sg_window_steps is None:
        sg_window_steps = window_size_sec2steps(sg_window_sec, dt)

    traj_smooth = savgol_filter(traj, sg_window_steps, sg_polyorder, deriv=0, axis=0)
    velxy = savgol_filter(
        traj, sg_window_steps, sg_polyorder, deriv=1, delta=dt, axis=0
    )

    return traj_smooth, velxy


def get_egocentric_vel(
    traj,
    dt,
    linspeed_sg_window_steps=None,
    linspeed_sg_window_sec=None,
    turnrate_sg_window_steps=None,
    turnrate_sg_window_sec=None,
    polyorder=3,
):
    """Compute linear speed and turn rate directly from raw trajectory.

    Linear speed and turn rate are estimated independently using separate SG windows,
    allowing fine-grained control over the smoothing of each quantity.

    Args:
        traj: (L, 2) raw trajectory array.
        dt: time step in seconds.
        linspeed_sg_window_steps: SG window length in steps for linear speed.
        linspeed_sg_window_sec: SG window length in seconds for linear speed.
        turnrate_sg_window_steps: SG window length in steps for turn rate.
        turnrate_sg_window_sec: SG window length in seconds for turn rate.
        polyorder: polynomial order for both SG filters.

    Returns:
        linspeed (L,), turnrate (L,)
    """
    if not ((linspeed_sg_window_steps is None) ^ (linspeed_sg_window_sec is None)):
        raise ValueError("Specify linspeed window size in steps or seconds, not both.")
    if not ((turnrate_sg_window_steps is None) ^ (turnrate_sg_window_sec is None)):
        raise ValueError("Specify turnrate window size in steps or seconds, not both.")

    if linspeed_sg_window_steps is None:
        linspeed_sg_window_steps = window_size_sec2steps(linspeed_sg_window_sec, dt)
    if turnrate_sg_window_steps is None:
        turnrate_sg_window_steps = window_size_sec2steps(turnrate_sg_window_sec, dt)

    # Linear speed: norm of velocity estimated with linspeed window
    velxy_linspeed = savgol_filter(
        traj, linspeed_sg_window_steps, polyorder, deriv=1, delta=dt, axis=0
    )
    linspeed = np.linalg.norm(velxy_linspeed, axis=1)

    # Turn rate: heading derived from velocity estimated with (wider) turnrate window
    velxy_turnrate = savgol_filter(
        traj, turnrate_sg_window_steps, polyorder, deriv=1, delta=dt, axis=0
    )
    heading_unwrapped = np.unwrap(
        np.arctan2(velxy_turnrate[:, 1], velxy_turnrate[:, 0])
    )
    turnrate = np.gradient(heading_unwrapped, dt)

    return linspeed, heading_unwrapped, turnrate
