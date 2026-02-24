from sppaper.common.plot import setup_matplotlib_params

setup_matplotlib_params()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    heading_unwrapped = np.unwrap(np.arctan2(velxy_turnrate[:, 1], velxy_turnrate[:, 0]))
    turnrate = np.gradient(heading_unwrapped, dt)

    return linspeed, heading_unwrapped, turnrate


### LEGACY CODE BELOW - TO BE DELETED ###
# def compute_traj_similarity(
#     traj,
#     traj_ref,
#     window_sec,
#     angvel_window_2ndpass_sec,
#     fps,
#     w_speed=1,
#     w_angvel=2,
#     w_align_nrmse=200,
#     w_ruggedness=50,
#     polyorder=2,
#     kpct=90,
#     decompose_traj_mismatch=False,
# ):
#     # Compute integer window size
#     dt = 1 / fps
#     window_size = int(np.round(window_sec * fps))
#     if window_size % 2 == 0:
#         window_size += 1  # window_size must be odd for savgol_filter
#     if angvel_window_2ndpass_sec is not None:
#         window_size_angvel_2ndpass = int(np.round(angvel_window_2ndpass_sec * fps))
#         if window_size_angvel_2ndpass % 2 == 0:
#             window_size_angvel_2ndpass += 1  # must be odd for savgol_filter

#     metrics = {"total": 0}
#     debug_vars = {"traj": traj, "traj_ref": traj_ref, "dt": dt}

#     if decompose_traj_mismatch:
#         # Compute match in egocentric kinematics (forward speed + turn rate time series)
#         speed, angvel = traj_to_egocentric_kinematics(
#             traj, dt, window_size, window_size_angvel_2ndpass, polyorder
#         )
#         speed_ref, angvel_ref = traj_to_egocentric_kinematics(
#             traj_ref, dt, window_size, window_size_angvel_2ndpass, polyorder
#         )
#         speed_mismatch_ts = np.abs(speed - speed_ref)
#         angvel_mismatch_ts = np.abs(angvel - angvel_ref)
#         speed_err_mean = mean_of_lowest_kpct(speed_mismatch_ts, kpct)
#         angvel_err_mean = mean_of_lowest_kpct(angvel_mismatch_ts, kpct)
#         traj_mismatch_total = w_speed * speed_err_mean + w_angvel * angvel_err_mean

#         metrics["speed_mismatch"] = speed_err_mean
#         metrics["angvel_mismatch"] = angvel_err_mean
#         metrics["traj_mismatch"] = traj_mismatch_total
#         metrics["total"] += traj_mismatch_total
#         debug_vars["speed"] = speed
#         debug_vars["angvel"] = angvel
#         debug_vars["speed_ref"] = speed_ref
#         debug_vars["angvel_ref"] = angvel_ref
#         debug_vars["speed_mismatch_ts_weighted"] = w_speed * speed_mismatch_ts
#         debug_vars["angvel_mismatch_ts_weighted"] = w_angvel * angvel_mismatch_ts
#         debug_vars["speed_mismatch_weighted"] = w_speed * speed_err_mean
#         debug_vars["angvel_mismatch_weighted"] = w_angvel * angvel_err_mean

#     else:
#         # Compute trajectory alignment metrics
#         traj_zeroed = traj - traj[0, :]
#         _, _, traj_ref_aligned, alignment_metrics = align_2d_traj(
#             traj_ref, traj_zeroed, anchor_origin=False
#         )
#         speed_err_mean = np.nan
#         angvel_err_mean = np.nan
#         align_nrmse = alignment_metrics["normalized_rmse"]

#         metrics["traj_mismatch"] = align_nrmse
#         metrics["total"] += w_align_nrmse * align_nrmse
#         debug_vars["align_nrmse_weighted"] = w_align_nrmse * align_nrmse

#     # Compute ruggedness of trajectory
#     cartvel_smoothed = savgol_filter(
#         traj, window_size, polyorder, deriv=1, delta=dt, axis=0
#     )
#     traj_smoothed = np.cumsum(cartvel_smoothed, axis=0) * dt + traj[0, :]
#     traj_len = compute_traj_len(traj)
#     traj_smoothed_len = compute_traj_len(traj_smoothed)

#     cartvel_ref_smoothed = savgol_filter(
#         traj_ref, window_size, polyorder, deriv=1, delta=dt, axis=0
#     )
#     traj_ref_smoothed = np.cumsum(cartvel_ref_smoothed, axis=0) * dt + traj_ref[0, :]
#     traj_ref_len = compute_traj_len(traj_ref)
#     traj_ref_smoothed_len = compute_traj_len(traj_ref_smoothed)

#     lengths = [traj_len, traj_smoothed_len, traj_ref_len, traj_ref_smoothed_len]
#     if any(l == 0 for l in lengths):
#         raise ValueError("At least one trajectory has 0 length. This shouldn't happen.")
#     traj_ruggedness = traj_len / traj_smoothed_len
#     traj_ref_ruggedness = traj_ref_len / traj_ref_smoothed_len
#     ruggedness_mismatch = np.abs(np.log(traj_ruggedness / traj_ref_ruggedness))

#     metrics["ruggedness_mismatch"] = ruggedness_mismatch
#     metrics["total"] += w_ruggedness * ruggedness_mismatch
#     debug_vars["traj_smoothed"] = traj_smoothed
#     debug_vars["traj_ref_smoothed"] = traj_ref_smoothed
#     debug_vars["ruggedness_mismatch_weighted"] = w_ruggedness * ruggedness_mismatch

#     return metrics, debug_vars


# def compute_traj_len(traj):
#     return np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()


# def mean_of_lowest_kpct(arr, kpct):
#     """Compute the mean of the lowest kpct% of values in arr."""
#     if len(arr.shape) != 1:
#         raise ValueError("Input array must be 1D")
#     k = max(1, int(arr.size * kpct / 100))
#     return np.mean(np.partition(arr, k)[:k])


# def traj_to_egocentric_kinematics(
#     traj, dt, window_size, window_size_angvel_2ndpass=None, polyorder=2
# ):
#     """Compute ego-centric speed and angular velocity from a 2D trajectory.

#     Uses Savitzky-Golay filtering throughout to avoid double-differentiation:
#     speed comes from a single SG deriv=1 pass on position, and angular velocity
#     comes from a single SG deriv=1 pass on the unwrapped heading angle.

#     Args:
#         traj: (L, 2) raw world-frame x-y positions.
#         dt: time step in seconds (1 / fps).
#         window_size: SG window size (number of samples, must be odd) used
#             for speed computation.
#         window_size_angvel_2ndpass: SG window size (number of samples, must be odd) used
#             for angular velocity computation in a second pass to reduce noise.
#         polyorder: SG polynomial order.

#     Returns:
#         speed:   (L,) scalar forward speed in units of traj / second.
#         angvel: (L,) signed angular velocity in rad/s. Positive = turning
#                  counter-clockwise.
#     """
#     # Ensure odd window lengths
#     if window_size % 2 == 0:
#         window_size += 1
#     if window_size_angvel_2ndpass is not None and window_size_angvel_2ndpass % 2 == 0:
#         window_size_angvel_2ndpass += 1

#     # Speed: one SG pass on position
#     vel = savgol_filter(traj, window_size, polyorder, deriv=1, delta=dt, axis=0)
#     speed = np.linalg.norm(vel, axis=1)

#     # Angular velocity: differentiate heading directly, avoiding double-differentiation
#     # Unwrap to remove +-pi discontinuities before filtering
#     heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
#     if window_size_angvel_2ndpass is None:
#         angvel = np.gradient(heading, dt)
#     else:
#         angvel = savgol_filter(
#             heading, window_size_angvel_2ndpass, polyorder, deriv=1, delta=dt
#         )

#     return speed, angvel


# def make_debug_plots(
#     traj_similarity_metrics, debug_vars, decompose_traj_mismatch=False, output_path=None
# ):
#     if decompose_traj_mismatch:
#         fig, axes = _make_debug_plots_decompose_traj_mismatch(
#             traj_similarity_metrics, debug_vars, output_path
#         )
#     else:
#         fig, axes = _make_debug_plots_global_traj_mismatch(
#             traj_similarity_metrics, debug_vars, output_path
#         )

#     if output_path is not None:
#         fig.savefig(output_path)
#         plt.close(fig)
#     else:
#         return fig, axes


# def _make_debug_plots_decompose_traj_mismatch(
#     traj_similarity_metrics, debug_vars, output_path=None
# ):
#     # Align ref trajectory to simulated trajectory for better visual comparison
#     traj_sim = debug_vars["traj"]
#     traj_sim = traj_sim - traj_sim[0, :]  # anchor to origin
#     traj_ref_unaligned = debug_vars["traj_ref"]
#     _, _, traj_ref, _ = align_2d_traj(traj_ref_unaligned, traj_sim, anchor_origin=True)
#     n_steps = traj_sim.shape[0]
#     t_grid = np.arange(n_steps) * debug_vars["dt"]

#     # Set up figure
#     fig = plt.figure(figsize=(12, 6))
#     gs = gridspec.GridSpec(
#         3, 2, figure=fig, width_ratios=[1, 0.7], hspace=0.3, wspace=0.2
#     )
#     ax_traj = plt.subplot(gs[:, 0])
#     ax_speed = plt.subplot(gs[0, 1])
#     ax_angvel = plt.subplot(gs[1, 1])
#     ax_scores = plt.subplot(gs[2, 1])
#     color_rec = "#546A76"
#     color_sim = "#FC7A1E"
#     color_speed = "#79A63D"
#     color_angvel = "#B7CF98"
#     color_align_nrmse = color_speed
#     color_ruggedness = "#F7AEF8"

#     # Aligned trajectories
#     ax_traj.plot(traj_ref[:, 0], traj_ref[:, 1], label="Recorded", color=color_rec)
#     ax_traj.plot(traj_sim[:, 0], traj_sim[:, 1], label="Simulated", color=color_sim)
#     ax_traj.scatter([0], [0], color="black", marker="o", label="Origin", zorder=10)
#     ax_traj.set_aspect("equal")
#     xlim = ax_traj.get_xlim()
#     ylim = ax_traj.get_ylim()
#     xcenter = np.mean(xlim)
#     ycenter = np.mean(ylim)
#     max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])  # already has some padding
#     half_range = max_range / 2
#     ax_traj.set_xlim(xcenter - half_range, xcenter + half_range)
#     ax_traj.set_ylim(ycenter - half_range, ycenter + half_range)
#     ax_traj.legend(frameon=False)

#     # Linear speed
#     ax_speed.plot(t_grid, debug_vars["speed_ref"], label="Recorded", color=color_rec)
#     ax_speed.plot(t_grid, debug_vars["speed"], label="Simulated", color=color_sim)
#     ax_speed.set_title("Linear speed")
#     ax_speed.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
#     ax_speed.set_ylabel("Speed (mm/s)")
#     ax_speed.set_xticklabels([])
#     ax_speed.set_xlim(t_grid[0], t_grid[-1])
#     ax_speed.set_ylim(bottom=0)

#     # Angular velocity
#     angvel_ref_turns = debug_vars["angvel_ref"] / (2 * np.pi)
#     angvel_turns = debug_vars["angvel"] / (2 * np.pi)
#     ax_angvel.axhline(0, color="black", linestyle="-", linewidth=1, zorder=-10)
#     ax_angvel.plot(t_grid, angvel_ref_turns, label="Recorded", color=color_rec)
#     ax_angvel.plot(t_grid, angvel_turns, label="Simulated", color=color_sim)
#     ax_angvel.set_title("Turn rate")
#     ax_angvel.set_ylabel("Ang. vel. (turns/s)")
#     ax_angvel.set_xticklabels([])
#     ax_angvel.set_xlim(t_grid[0], t_grid[-1])
#     max_mag = np.abs(ax_angvel.get_ylim()).max()
#     ax_angvel.set_ylim(-max_mag, max_mag)

#     # Similarity scores
#     ruggedness_mismatch_weighted = (
#         np.ones(n_steps) * debug_vars["ruggedness_mismatch_weighted"]
#     )
#     speed_mismatch_ts_weighted = debug_vars["speed_mismatch_ts_weighted"]
#     angvel_mismatch_ts_weighted = debug_vars["angvel_mismatch_ts_weighted"]
#     ax_scores.stackplot(
#         t_grid,
#         speed_mismatch_ts_weighted,
#         angvel_mismatch_ts_weighted,
#         ruggedness_mismatch_weighted,
#         labels=["Speed", "Angular velocity", "Ruggedness"],
#         colors=[color_speed, color_angvel, color_ruggedness],
#     )
#     ax_scores.set_title("Mismatch metric breakdown")
#     ax_scores.set_xlabel("Time (s)")
#     ax_scores.set_ylabel("Metric (a.u.)")
#     ax_scores.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
#     ax_scores.set_xlim(t_grid[0], t_grid[-1])
#     score_text = (
#         f"Total mismatch: {traj_similarity_metrics['total']:.2e}\n"
#         f"Speed term: {debug_vars['speed_mismatch_weighted']:.2e}\n"
#         f"Ang. vel. term: {debug_vars['angvel_mismatch_weighted']:.2e}\n"
#         f"Ruggedness term: {debug_vars['ruggedness_mismatch_weighted']:.2e}"
#     )
#     ax_scores.text(
#         0.99,
#         0.97,
#         score_text,
#         transform=ax_scores.transAxes,
#         horizontalalignment="right",
#         verticalalignment="top",
#         fontsize="small",
#     )

#     axes = {
#         "traj": ax_traj,
#         "speed": ax_speed,
#         "angvel": ax_angvel,
#         "scores": ax_scores,
#     }
#     return fig, axes


# def _make_debug_plots_global_traj_mismatch(
#     traj_similarity_metrics, debug_vars, output_path=None
# ):
#     # Align ref trajectory to simulated trajectory for better visual comparison
#     traj_sim = debug_vars["traj"]
#     traj_sim = traj_sim - traj_sim[0, :]  # anchor to origin
#     traj_ref_unaligned = debug_vars["traj_ref"]
#     _, _, traj_ref, _ = align_2d_traj(traj_ref_unaligned, traj_sim, anchor_origin=True)

#     # Set up figure
#     fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
#     color_rec = "#546A76"
#     color_sim = "#FC7A1E"

#     # Aligned trajectories
#     ax.plot(traj_ref[:, 0], traj_ref[:, 1], label="Recorded", color=color_rec)
#     ax.plot(traj_sim[:, 0], traj_sim[:, 1], label="Simulated", color=color_sim)
#     ax.scatter([0], [0], color="black", marker="o", label="Origin", zorder=10)
#     ax.set_aspect("equal")
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     xcenter = np.mean(xlim)
#     ycenter = np.mean(ylim)
#     max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])  # already has some padding
#     half_range = max_range / 2
#     ax.set_xlim(xcenter - half_range, xcenter + half_range)
#     ax.set_ylim(ycenter - half_range, ycenter + half_range)
#     ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
#     score_text = (
#         f"Total mismatch: {traj_similarity_metrics['total']:.2e}\n"
#         f"Trajectory nRMSE: {debug_vars['align_nrmse_weighted']:.2e}\n"
#         f"Ruggedness term: {debug_vars['ruggedness_mismatch_weighted']:.2e}"
#     )
#     ax.text(
#         1.04,
#         0.02,
#         score_text,
#         transform=ax.transAxes,
#         horizontalalignment="left",
#         verticalalignment="bottom",
#         fontsize="medium",
#     )

#     return fig, ax
