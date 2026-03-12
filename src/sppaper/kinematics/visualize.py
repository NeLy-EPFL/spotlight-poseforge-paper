from sppaper.common.plot import setup_matplotlib_params, find_font_path

setup_matplotlib_params()

import pickle
from pathlib import Path


import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cmasher
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from vidstab import VidStab
from tqdm import trange, tqdm

from flygym.anatomy import LEGS

import sppaper.kinematics.shared_constants as const
from sppaper.kinematics.data import align_smooth_decompose_trajs
from sppaper.common.io import load_precise_sparse_frames

KIN_FILTER_WINDOW_SIZE = 3
DOF_DISPLAY_NAMES = {
    "ThC_pitch": "ThC-p",
    "ThC_roll": "ThC-r",
    "ThC_yaw": "ThC-y",
    "CTr_pitch": "CTr-p",
    "CTr_roll": "CTr-r",
    "FTi_pitch": "FTi-p",
    "TiTa_pitch": "TiTa-p",
}
LEG_DISP_NAMES = {
    "LF": "Left front leg",
    "LM": "Left middle leg",
    "LH": "Left hind leg",
    "RF": "Right front leg",
    "RM": "Right middle leg",
    "RH": "Right hind leg",
}
AXIS_DISPLAY_NAMES = {"x": "fore/aft", "y": "med/lat", "z": "height"}
MM_TO_IN = 1 / 25.4
SWING_SPEED_THRESHOLD = 25
CONTACT_FORCE_DISP_THRESHOLD = 0.5

CLAW_XYZ_COLOR = "#546a76"
DOF_ANGLES_COLOR = "#a23e48"
ACTUATOR_FORCES_COLOR = "#689829"
REC_COLOR = "#546a76"
SIM_COLOR = "#689829"


def reduce_timeseries_sim2rec(ts, ctrl_update_mask, reduce_fn=np.mean, stride=1):
    groupids = np.cumsum(ctrl_update_mask) - 1
    groupids = (groupids / stride).astype(np.uint32)
    n_groups = groupids[-1] + 1
    ts_chunks = [ts[groupids == i, ...] for i in range(n_groups)]
    reduced = np.stack([reduce_fn(chunk, axis=0) for chunk in ts_chunks])
    return reduced


def plot_time_series(
    sim_dir,
    leg,
    gait_info,
    t_range=None,
    claw_xyz_range=1.5,
    dof_angles_range=90,
    actuator_forces_range=80,
    contact_force_vmax=8,
    claw_xyz_color=CLAW_XYZ_COLOR,
    dof_angles_color=DOF_ANGLES_COLOR,
    actuator_forces_color=ACTUATOR_FORCES_COLOR,
    medkernel_size=const.KIN_MEDFILTER_SIZE,
    contact_force_disp_threshold=CONTACT_FORCE_DISP_THRESHOLD,
    ratelim_xyz=const.XYZ_RATELIM,
    ratelim_angles=const.JOINT_ANGLE_RATELIM,
    xticks_interval=0.5,
):
    with open(sim_dir / "sim_data.pkl", "rb") as f:
        data = pickle.load(f)
        sim_results = data["sim_results"]
        kinematic_snippet = data["snippet"]
        if "adhforce_by_leg" in data:
            adhforce_by_leg = data["adhforce_by_leg"]
        else:
            # # Legacy format from replay_walking.py
            # leg_adhesion_force = data["leg_adhesion_force"]
            # adhforce_by_leg = {
            #     leg: replay_manager.leg_adhesion_gain[leg] * leg_adhesion_force
            #     for leg in LEGS
            # }
            raise ValueError("adhforce_by_leg not found in sim_data.pkl - old format!")
        replay_manager = data["replay_manager"]

    if t_range is not None:
        start_idx_before = kinematic_snippet.start_idx
        kinematic_snippet = kinematic_snippet.get_subselection(*t_range)
        steps_offset = kinematic_snippet.start_idx - start_idx_before
    else:
        steps_offset = 0

    fwdkin_xyz = kinematic_snippet.get_filtered_fwdkin_world_xyz(
        medkernel_size, ratelim_xyz
    )
    joint_angles = kinematic_snippet.get_filtered_joint_angles(
        medkernel_size, ratelim_angles
    )
    t_grid = np.arange(fwdkin_xyz.shape[0]) / kinematic_snippet.data_fps

    i_leg = kinematic_snippet.metadata["legs_order"].index(leg.upper())
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(100 * MM_TO_IN, 120 * MM_TO_IN),
        gridspec_kw={"height_ratios": [2, 3, 3, 3, 3]},
        tight_layout=True,
    )

    last_xtick = t_grid[-1] + 1 / kinematic_snippet.data_fps
    xticks = list(np.arange(0, last_xtick + 1e-6, xticks_interval))
    xticklabels = [f"{x:.1f}" for x in xticks]
    xticklabels[-1] += " s"

    # ===== Claw xyz =====
    ax = axes[0]
    sns.despine(ax=ax)
    claw_idx = -1
    yticks = []
    yticklabels = []
    for i_axis, axis in enumerate("xyz"):
        ts = fwdkin_xyz[:, i_leg, claw_idx, i_axis]
        span = np.percentile(ts, [1, 99])
        y_center = i_axis * claw_xyz_range + 0.5 * claw_xyz_range
        y_offset = np.mean(span) - y_center
        ax.plot(
            t_grid,
            ts - y_offset,
            label=axis,
            color=claw_xyz_color,
            clip_on=False,
        )
        yticks.append(y_center)
        yticklabels.append(AXIS_DISPLAY_NAMES[axis])

    # Add scale bar
    ax.plot(
        [t_grid[-1] * 1.02] * 2,
        [0.2, 1.2],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.text(t_grid[-1] * 1.03, 0.7, "1 mm", va="center", ha="left", fontsize="small")

    # Configure y axis
    y_max = 3 * claw_xyz_range
    ax.set_ylim(y_max + 0.25 * claw_xyz_range, 0)
    ax.set_yticks(yticks, yticklabels)
    ax.set_ylabel("Claw position", color=claw_xyz_color)
    ax.yaxis.set_label_coords(-0.13, 0.5)

    # Configure x axis
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlim(t_grid[0], t_grid[-1] + 1 / kinematic_snippet.data_fps)

    # ===== Joint angles =====
    ax = axes[1]
    yticks = []
    yticklabels = []
    sns.despine(ax=ax)
    for i_dof, dof in enumerate(kinematic_snippet.metadata["dofs_order_per_leg"]):
        idx = kinematic_snippet.metadata["joints_order"].index(f"{leg.upper()}{dof}")
        ts = np.rad2deg(joint_angles[:, idx])
        span = np.percentile(ts, [1, 99])
        y_center = i_dof * dof_angles_range + 0.5 * dof_angles_range
        y_offset = np.mean(span) - y_center
        ax.plot(
            t_grid,
            ts - y_offset,
            label=dof,
            color=dof_angles_color,
            clip_on=False,
        )
        yticks.append(y_center)
        yticklabels.append(DOF_DISPLAY_NAMES[dof])

    # Add scale bar
    ax.plot(
        [t_grid[-1] * 1.02] * 2,
        [30, 75],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.text(t_grid[-1] * 1.03, 52.5, "45°", va="center", ha="left", fontsize="small")

    # Configure y axis
    y_max = len(kinematic_snippet.metadata["dofs_order_per_leg"]) * dof_angles_range
    ax.set_ylim(y_max + 0.25 * dof_angles_range, 0)
    ax.set_yticks(yticks, yticklabels)
    ax.set_ylabel("Joint angles", color=dof_angles_color)
    ax.yaxis.set_label_coords(-0.13, 0.5)

    # Configure x axis
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlim(t_grid[0], t_grid[-1] + 1 / kinematic_snippet.data_fps)

    # ===== Gait diagram =====
    ax = axes[2]
    ax.imshow(
        np.repeat(~gait_info["swing_mask"].T, axis=0, repeats=50),
        aspect="auto",
        interpolation="none",
        extent=[0, t_range[1] - t_range[0], 6, 0],
        cmap="gray",
    )
    leg_disp_names = [x.upper() for x in LEGS]
    ax.set_yticks(np.arange(6) + 0.5, leg_disp_names)
    ax.set_ylabel("Gait diagram", color="black")
    ax.yaxis.set_label_coords(-0.13, 0.5)
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlabel("Time (s)")

    # ===== Actuator force =====
    ax = axes[3]
    sns.despine(ax=ax)
    yticks = []
    yticklabels = []
    for i_dof, dof in enumerate(kinematic_snippet.metadata["dofs_order_per_leg"]):
        idx = kinematic_snippet.metadata["joints_order"].index(f"{leg.upper()}{dof}")
        ts = reduce_timeseries_sim2rec(
            sim_results["actuator_forces"][:, idx], sim_results["ctrl_update_mask"]
        )
        ts = ts[steps_offset : steps_offset + len(t_grid)]
        ts = medfilt(ts, medkernel_size)
        y_offset = i_dof * actuator_forces_range
        ax.axhline(y_offset, linewidth=0.5, color="#bbbbbb", linestyle="-", zorder=0)
        ax.plot(
            t_grid,
            ts + y_offset,
            label=dof,
            color=actuator_forces_color,
            zorder=10,
        )
        yticks.append(y_offset)
        yticklabels.append(DOF_DISPLAY_NAMES[dof])

    # Add scale bar
    ax.plot(
        [t_grid[-1] * 1.02] * 2,
        [-5, 35],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.text(t_grid[-1] * 1.03, 15, "40 μN", va="center", ha="left", fontsize="small")

    # Configure y axis
    ax.set_ylim(6.5 * actuator_forces_range, -0.5 * actuator_forces_range)
    ax.set_yticks(yticks, yticklabels)
    ax.set_ylabel("Forces applied", color=actuator_forces_color)
    ax.yaxis.set_label_coords(-0.13, 0.5)

    # Configure x axis
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlim(t_grid[0], t_grid[-1] + 1 / kinematic_snippet.data_fps)

    # ===== Load by leg =====
    ax = axes[4]
    ts = sim_results["ground_contacts"]["forces_world"][steps_offset:, :, 2].copy() * -1
    ts[np.isnan(ts)] = 0
    leg_adhesion_forces = np.array([adhforce_by_leg[leg] for leg in LEGS])
    ts -= leg_adhesion_forces[None, :]
    ts[ts < contact_force_disp_threshold] = 0
    ts = reduce_timeseries_sim2rec(
        ts, sim_results["ctrl_update_mask"][steps_offset:], stride=3
    )

    im = ax.imshow(
        np.repeat(ts.T, axis=0, repeats=50),
        aspect="auto",
        interpolation="none",
        extent=[t_grid[0], t_grid[-1] + 1 / kinematic_snippet.data_fps, 6, 0],
        vmin=0,
        vmax=contact_force_vmax,
        cmap=cmasher.arctic_r,
    )
    cax = ax.inset_axes([1.02, 0, 0.03, 1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([0, 4, 8], labels=["0", "4", "8 μN"])
    leg_disp_names = [x.upper() for x in LEGS]
    ax.set_yticks(np.arange(6) + 0.5, leg_disp_names)
    ax.set_ylabel("Weight load by leg", color="black")
    ax.yaxis.set_label_coords(-0.13, 0.5)
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlabel("Time (s)")

    return fig, axes


def plot_trajectory(
    sim_dir: Path,
    t_range=None,
    rec_color=REC_COLOR,
    sim_color=SIM_COLOR,
    xticks_interval=0.5,
    base_rot_deg=0.0,
):
    """Plot trajectory comparison between recorded and simulated data.

    Args:
        sim_dir: Directory containing sim_data.pkl.
        t_range: Optional (start, end) time range in seconds to plot.
        rec_color: Color for recorded trajectory lines.
        sim_color: Color for simulated trajectory lines.
        xticks_interval: Interval between x-axis ticks in seconds.
        base_rot_deg: Degrees to rotate all 2D trajectories and world-frame
            velocity vectors before plotting. For aesthetics only — does not
            affect any underlying data or derived quantities (linear speed,
            turn rate, etc.).
    """
    with open(sim_dir / "sim_data.pkl", "rb") as f:
        data = pickle.load(f)
        sim_results = data["sim_results"]
        kinematic_snippet = data["snippet"]
    trajs_info = align_smooth_decompose_trajs(
        kinematic_snippet, sim_results, t_range=t_range
    )
    if t_range is not None:
        kinematic_snippet = kinematic_snippet.get_subselection(*t_range)

    dt = 1 / kinematic_snippet.data_fps
    t_grid = np.arange(len(kinematic_snippet)) * dt

    last_xtick = t_grid[-1] + 1 / kinematic_snippet.data_fps
    xticks = list(np.arange(0, last_xtick + 1e-6, xticks_interval))
    xticklabels = [f"{x:.1f}" for x in xticks]
    xticklabels[-1] += " s"

    # Build display rotation matrix (aesthetics only — applied to local copies)
    _angle = np.deg2rad(base_rot_deg)
    _c, _s = np.cos(_angle), np.sin(_angle)
    _R = np.array([[_c, -_s], [_s, _c]])
    basetraj_rec = trajs_info["basetraj_rec"] @ _R.T
    basetraj_rec_filtered = trajs_info["basetraj_rec_filtered"] @ _R.T
    basetraj_sim = trajs_info["basetraj_sim"] @ _R.T
    basetraj_sim_filtered = trajs_info["basetraj_sim_filtered"] @ _R.T
    basevelxy_rec = trajs_info["basevelxy_rec"] @ _R.T
    basevelxy_sim = trajs_info["basevelxy_sim"] @ _R.T

    fig = plt.figure(figsize=(140 * MM_TO_IN, 50 * MM_TO_IN), tight_layout=True)
    gs = gridspec.GridSpec(2, 3)
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_velx = fig.add_subplot(gs[0, 1])
    ax_vely = fig.add_subplot(gs[1, 1])
    ax_linspeed = fig.add_subplot(gs[0, 2])
    ax_turnrate = fig.add_subplot(gs[1, 2])

    # Trajectories
    ax_traj.plot(
        basetraj_rec[:, 0],
        basetraj_rec[:, 1],
        label="Recorded (raw)",
        color=rec_color,
        linestyle="--",
    )
    ax_traj.plot(
        basetraj_rec_filtered[:, 0],
        basetraj_rec_filtered[:, 1],
        label="Recorded (smoothed)",
        color=rec_color,
        linestyle="-",
    )
    ax_traj.plot(
        basetraj_sim[:, 0],
        basetraj_sim[:, 1],
        label="Simulated (raw)",
        color=sim_color,
        linestyle="--",
    )
    ax_traj.plot(
        basetraj_sim_filtered[:, 0],
        basetraj_sim_filtered[:, 1],
        label="Simulated (smoothed)",
        color=sim_color,
        linestyle="-",
    )
    ax_traj.scatter([0], [0], color="black", label="Origin", s=5, zorder=10)
    ax_traj.legend()
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.set_xlabel("x pos. (mm)")
    ax_traj.set_ylabel("y pos. (mm)")
    ax_traj.set_title("Trajectory")

    # Velocity in world coordinates
    ax_velx.plot(t_grid, basevelxy_rec[:, 0], label="Recorded", color=rec_color)
    ax_velx.plot(t_grid, basevelxy_sim[:, 0], label="Simulated", color=sim_color)
    ax_velx.set_ylabel("x vel. (mm/s)")
    ax_velx.set_xlim(0, t_grid[-1] + dt)
    ax_velx.set_xticks(xticks, xticklabels)
    ax_velx.set_title("Velocity (global x)")
    sns.despine(ax=ax_velx)
    ax_vely.plot(t_grid, basevelxy_rec[:, 1], label="Recorded", color=rec_color)
    ax_vely.plot(t_grid, basevelxy_sim[:, 1], label="Simulated", color=sim_color)
    ax_vely.set_ylabel("y vel. (mm/s)")
    ax_vely.set_xticks(xticks, xticklabels)
    ax_vely.set_xlim(0, t_grid[-1] + dt)
    ax_vely.set_title("Velocity (global y)")
    sns.despine(ax=ax_vely)

    # Velocity in egocentric coordinates
    ax_linspeed.plot(
        t_grid, trajs_info["baselinspeed_rec"], label="Recorded", color=rec_color
    )
    ax_linspeed.plot(
        t_grid, trajs_info["baselinspeed_sim"], label="Simulated", color=sim_color
    )
    ax_linspeed.set_ylabel("Linear speed (mm/s)")
    ax_linspeed.set_xticks(xticks, xticklabels)
    ax_linspeed.set_xlim(0, t_grid[-1] + dt)
    ax_linspeed.set_title("Linear speed")
    sns.despine(ax=ax_linspeed)
    ax_turnrate.plot(
        t_grid,
        trajs_info["baseturnrate_rec"] / (2 * np.pi),
        label="Recorded",
        color=rec_color,
    )
    ax_turnrate.plot(
        t_grid,
        trajs_info["baseturnrate_sim"] / (2 * np.pi),
        label="Simulated",
        color=sim_color,
    )
    ax_turnrate.set_xlabel("Time (s)")
    ax_turnrate.set_ylabel("Turn rate (turns/s)")
    ax_turnrate.set_xticks(xticks, xticklabels)
    ax_turnrate.set_xlim(0, t_grid[-1] + dt)
    ax_turnrate.set_title("Turn rate")
    sns.despine(ax=ax_turnrate)

    return fig, (ax_traj, ax_velx, ax_vely, ax_linspeed, ax_turnrate)


def get_centerpos_and_heading(neck_pos, thorax_pos, filter_window=None):
    """
    Compute smoothed center position and heading of the fly.

    Args:
        neck_pos:      (L, 2) array of (x, y) pixel positions of the neck.
        thorax_pos:    (L, 2) array of (x, y) pixel positions of the thorax.
        filter_window: Odd integer window length for Savitzky-Golay filter,
                       or None to skip filtering.

    Returns:
        center_pos: (L, 2) smoothed thorax positions.
        heading:    (L,) unwrapped heading in radians.
    """
    diff = neck_pos - thorax_pos
    heading_raw = np.arctan2(diff[:, 1], diff[:, 0])
    heading = np.unwrap(heading_raw)

    if filter_window is not None and filter_window > 1:
        center_pos = np.stack(
            [
                savgol_filter(thorax_pos[:, 0], filter_window, 2),
                savgol_filter(thorax_pos[:, 1], filter_window, 2),
            ],
            axis=1,
        )
        heading = savgol_filter(heading, filter_window, 2)
    else:
        center_pos = thorax_pos.astype(float).copy()

    return center_pos, heading


def crop_single_image(
    full_image, center_pos, heading, bbox_sidelen=720, center_offset=(0, 0)
):
    """
    Rotate the image so the fly faces up, then crop a square around the center.

    Args:
        full_image:    H x W or H x W x C numpy array.
        center_pos:    (x, y) pixel coordinate of the thorax.
        heading:       Scalar heading in radians.
        bbox_sidelen:  Side length of the square output patch in pixels.
        center_offset: (lateral, forward) offset in pixels in the fly's body frame.
                       forward (+) is toward the neck; lateral (+) is to the fly's right.
                       e.g. (0, 10) shifts the center 10 px toward the neck.

    Returns:
        cropped_image: (bbox_sidelen, bbox_sidelen[, C]) array, zero-padded if out of bounds.
    """
    # Apply center_offset in the fly's body frame
    lat_off, fwd_off = center_offset
    fwd = np.array([np.cos(heading), np.sin(heading)])
    rgt = np.array([np.sin(heading), -np.cos(heading)])
    cx, cy = center_pos + fwd_off * fwd + lat_off * rgt

    cx, cy = int(round(cx)), int(round(cy))
    h, w = full_image.shape[:2]

    angle_deg = -np.degrees(heading) - 90.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale=1.0)
    rotated = cv2.warpAffine(
        full_image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    half = bbox_sidelen // 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half

    pad_left = max(0, -x0)
    pad_right = max(0, x1 - w)
    pad_top = max(0, -y0)
    pad_bot = max(0, y1 - h)

    x0c, x1c = x0 + pad_left, x1 - pad_right
    y0c, y1c = y0 + pad_top, y1 - pad_bot

    crop = rotated[y0c:y1c, x0c:x1c]

    if any([pad_left, pad_right, pad_top, pad_bot]):
        pad_width = [(pad_top, pad_bot), (pad_left, pad_right)]
        if full_image.ndim == 3:
            pad_width.append((0, 0))
        crop = np.pad(crop, pad_width, mode="constant", constant_values=0)

    return crop


class TrajectoryVideoPlotter:
    def __init__(
        self,
        xypos_ts,
        heading_ts,
        linspeed_ts,
        turnrate_ts,
        dt,
        datacolor="tab:blue",
        fgcolor="#cccccc",
        extent=15,
        ylim_margin=0.2,
        width_px=900,
        dpi=300,
    ):
        self.traj = np.array(xypos_ts)
        self.heading_ts = heading_ts
        self.linspeed = np.array(linspeed_ts)
        self.turnrate = np.array(turnrate_ts) / (2 * np.pi)  # convert to turns/s
        self.t_grid = np.arange(self.traj.shape[0]) * dt

        self.traj_halfextent = extent / 2
        self.ylim_margin = ylim_margin
        self.dpi = dpi
        self.datacolor = datacolor
        self.fgcolor = fgcolor

        figsize_px = (width_px, width_px // 2)
        figsize_in = [x / dpi for x in figsize_px]
        self.fig = plt.figure(figsize=figsize_in, tight_layout=True, dpi=self.dpi)
        gs = gridspec.GridSpec(2, 2, figure=self.fig)
        self.ax_traj = self.fig.add_subplot(gs[:, 0])
        self.ax_linspeed = self.fig.add_subplot(gs[0, 1])
        self.ax_turnrate = self.fig.add_subplot(gs[1, 1])

        self._setup()

    def _style_ax(self, ax):
        ax.set_facecolor((0, 0, 0, 0.5))
        for spine in ax.spines.values():
            spine.set_color(self.fgcolor)
        ax.tick_params(colors=self.fgcolor, labelsize=6)

    def _setup(self):
        self.fig.patch.set_facecolor((0, 0, 0, 1))
        self.fig.subplots_adjust(0, 0, 1, 1, wspace=0.3, hspace=0.3)

        # Trajectory axes
        self.ax_traj.set_aspect("equal")
        self._style_ax(self.ax_traj)
        self.ax_traj.set_xticks([])
        self.ax_traj.set_yticks([])
        (self.line_past,) = self.ax_traj.plot(
            [], [], color=self.datacolor, linewidth=1.5
        )
        (self.line_future,) = self.ax_traj.plot(
            [], [], color=self.datacolor, linewidth=0.5, alpha=0.5
        )
        (self.dot,) = self.ax_traj.plot(
            [], [], marker="o", color=self.datacolor, markersize=2, linestyle="none"
        )
        self.ax_traj.plot(
            [-0.9 * self.traj_halfextent, -0.9 * self.traj_halfextent + 2],
            [-0.87 * self.traj_halfextent, -0.87 * self.traj_halfextent],
            color=self.fgcolor,
            linewidth=1,
        )
        self.ax_traj.text(
            -0.9 * self.traj_halfextent + 2.5,
            -0.87 * self.traj_halfextent,
            "2 mm",
            color=self.fgcolor,
            fontsize=5,
            va="center",
            ha="left",
        )
        self.ax_traj.set_title("Trajectory", color=self.fgcolor, fontsize=6)

        # Linspeed axes
        self._style_ax(self.ax_linspeed)
        self.ax_linspeed.set_xlim(0, self.t_grid[-1])
        self.ax_linspeed.set_xticklabels([])
        self.ax_linspeed.set_ylim(0, self.linspeed.max() * (1 + self.ylim_margin))
        self.ax_linspeed.set_ylabel("(mm/s)", color=self.fgcolor, fontsize=5)
        (self.ls_past,) = self.ax_linspeed.plot(
            [], [], color=self.datacolor, linewidth=1.5
        )
        (self.ls_future,) = self.ax_linspeed.plot(
            [], [], color=self.datacolor, linewidth=0.5, alpha=0.5
        )
        (self.ls_dot,) = self.ax_linspeed.plot(
            [], [], marker="o", color=self.datacolor, markersize=2, linestyle="none"
        )
        self.ax_linspeed.set_title("Linear Speed", color=self.fgcolor, fontsize=6)
        sns.despine(ax=self.ax_linspeed)

        # Turnrate axes
        self._style_ax(self.ax_turnrate)
        self.ax_turnrate.set_xlim(0, self.t_grid[-1])
        y_absmax_with_margin = np.max(np.abs(self.turnrate)) * (1 + self.ylim_margin)
        self.ax_turnrate.set_ylim(-y_absmax_with_margin, y_absmax_with_margin)
        self.ax_turnrate.set_ylabel("(turns/s)", color=self.fgcolor, fontsize=5)
        (self.tr_past,) = self.ax_turnrate.plot(
            [], [], color=self.datacolor, linewidth=1.5
        )
        xticklabels = self.ax_turnrate.get_xticklabels()
        xticklabels[-1].set_text(xticklabels[-1].get_text() + " s")
        self.ax_turnrate.set_xticks(
            self.ax_turnrate.get_xticks(), labels=xticklabels, fontsize=5
        )
        (self.tr_future,) = self.ax_turnrate.plot(
            [], [], color=self.datacolor, linewidth=0.5, alpha=0.5
        )
        (self.tr_dot,) = self.ax_turnrate.plot(
            [], [], marker="o", color=self.datacolor, markersize=2, linestyle="none"
        )
        self.ax_turnrate.set_title("Turn Rate", color=self.fgcolor, fontsize=6)
        sns.despine(ax=self.ax_turnrate, bottom=True)
        self.ax_turnrate.axhline(0, color=self.fgcolor, linewidth=0.75, zorder=-100)

    def _rotate(self, t):
        angle = np.pi / 2 - self.heading_ts[t]
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return (self.traj - self.traj[t]) @ R.T

    def plot_snapshot(self, t):
        # Trajectory
        traj_r = self._rotate(t)
        cx, cy = traj_r[t]
        self.line_past.set_data(traj_r[:t, 0], traj_r[:t, 1])
        self.line_future.set_data(traj_r[t:, 0], traj_r[t:, 1])
        self.dot.set_data([traj_r[t, 0]], [traj_r[t, 1]])
        self.ax_traj.set_xlim(cx - self.traj_halfextent, cx + self.traj_halfextent)
        self.ax_traj.set_ylim(cy - self.traj_halfextent, cy + self.traj_halfextent)

        # Linear speed
        self.ls_past.set_data(self.t_grid[:t], self.linspeed[:t])
        self.ls_future.set_data(self.t_grid[t:], self.linspeed[t:])
        self.ls_dot.set_data([self.t_grid[t]], [self.linspeed[t]])

        # Turn rate
        self.tr_past.set_data(self.t_grid[:t], self.turnrate[:t])
        self.tr_future.set_data(self.t_grid[t:], self.turnrate[t:])
        self.tr_dot.set_data([self.t_grid[t]], [self.turnrate[t]])

        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())

    def make_video(self, output_path, fps):
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with iio.imopen(str(output_path), "w", plugin="pyav") as writer:
            writer.init_video_stream("h264", fps=fps)
            for t in trange(len(self.traj)):
                frame = self.plot_snapshot(t)
                writer.write_frame(frame[:, :, :3])

    def close(self):
        plt.close(self.fig)


def load_frames_with_stabilization(input_path, smoothing_window, overwrite=False):
    output_path = input_path.parent / f"{input_path.stem}_stabilized.mp4"
    if not output_path.exists() or overwrite:
        cv2.destroyAllWindows = lambda: None
        stabilizer = VidStab()
        stabilizer.stabilize(
            input_path=input_path,
            output_path=output_path,
            smoothing_window=smoothing_window,
            show_progress=False,
        )
    return load_precise_sparse_frames(output_path)


def make_replay_video(
    sim_dir: Path,
    output_path: Path,
    t_range: tuple[float, float] | None = None,
    intermediate_video_playback_speed=const.VIDEO_PLAYBACK_SPEED,
    intermediate_video_output_fps=const.VIDEO_OUTPUT_FPS,
    final_output_playback_speed=const.VIDEO_PLAYBACK_SPEED,
    rec_color=REC_COLOR,
    sim_color=SIM_COLOR,
    video_blockdim=900,
    heading_filtersize_for_cropping=33,
    video_stabilization_window=33,
    traj_extent=15,
    coarse_frames_interval=None,
):
    with open(sim_dir / "sim_data.pkl", "rb") as f:
        data = pickle.load(f)
        sim_results = data["sim_results"]
        kinematic_snippet = data["snippet"]
        sim_timestep = data["sim_timestep"]
    trajs_info = align_smooth_decompose_trajs(kinematic_snippet, sim_results)

    rec_centerpos, rec_heading = get_centerpos_and_heading(
        kinematic_snippet.neck_pos_px,
        kinematic_snippet.thorax_pos_px,
        filter_window=heading_filtersize_for_cropping,
    )

    # Load sim rendering frames and compute smoothed headings for cropping
    sim_bottomcam_frames = load_precise_sparse_frames(sim_dir / "bottom_cam.mp4")
    sim_frameheight, sim_framewidth = sim_bottomcam_frames[0].shape[:2]
    sim_frame_center = (sim_framewidth / 2, sim_frameheight / 2)
    sim_headings_smooth = savgol_filter(
        sim_results["heading_inputmatched"],
        heading_filtersize_for_cropping,
        polyorder=2,
    )

    # Load stabilized version of tracking cam rendering during simulation
    sim_sidecam_frames = load_frames_with_stabilization(
        input_path=sim_dir / "nmf_trackcam.mp4",
        smoothing_window=video_stabilization_window,
    )

    # Match indices in different data sources based on time
    frame_pairs = []
    for i_nmf_frame in range(len(sim_bottomcam_frames)):
        sim_time_s = (
            intermediate_video_playback_speed
            * i_nmf_frame
            / intermediate_video_output_fps
        )
        i_recsnippet_frame = int(np.round((sim_time_s) * kinematic_snippet.data_fps))
        i_fullrecording_frame = i_recsnippet_frame + kinematic_snippet.start_idx

        if t_range is None or t_range[0] <= sim_time_s <= t_range[1]:
            frame_pairs.append((i_nmf_frame, i_recsnippet_frame, i_fullrecording_frame))

    # Load Spotlight behavior recording frames
    spotlight_recording_path = (
        kinematic_snippet.exp_trial_dir / "processed/fullsize_behavior_video.mkv"
    )
    frames_to_read = [
        i_fullrecording_frame
        for i_nmf_frame, i_recsnippet_frame, i_fullrecording_frame in frame_pairs
    ]
    spotlight_frames_unmatched = load_precise_sparse_frames(
        spotlight_recording_path, frames_to_read
    )
    spotlight_frames = dict(zip(frames_to_read, spotlight_frames_unmatched))

    # Recorded and simulated trajectory plotters
    traj_indices = [
        i_recsnippet_frame
        for i_nmf_frame, i_recsnippet_frame, i_fullrecording_frame in frame_pairs
    ]
    video_output_dt = (
        1 / intermediate_video_output_fps * intermediate_video_playback_speed
    )
    rectraj_plotter = TrajectoryVideoPlotter(
        xypos_ts=trajs_info["basetraj_rec_filtered"][traj_indices],
        heading_ts=trajs_info["baseheading_rec"][traj_indices],
        linspeed_ts=trajs_info["baselinspeed_rec"][traj_indices],
        turnrate_ts=trajs_info["baseturnrate_rec"][traj_indices],
        dt=video_output_dt,
        datacolor=rec_color,
        width_px=video_blockdim,
        extent=traj_extent,
    )
    simtraj_plotter = TrajectoryVideoPlotter(
        xypos_ts=trajs_info["basetraj_sim_filtered"][traj_indices],
        heading_ts=trajs_info["baseheading_sim"][traj_indices],
        linspeed_ts=trajs_info["baselinspeed_sim"][traj_indices],
        turnrate_ts=trajs_info["baseturnrate_sim"][traj_indices],
        dt=video_output_dt,
        datacolor=sim_color,
        width_px=video_blockdim,
        extent=traj_extent,
    )

    # Font etc for text labels
    font_normal = ImageFont.truetype(find_font_path("Arial", weight="bold"), 30)
    font_small = ImageFont.truetype(find_font_path("Arial"), 25)

    # Output video writer
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    writer = iio.imopen(output_path, "w", plugin="pyav")
    final_output_fps = (
        final_output_playback_speed / intermediate_video_playback_speed
    ) * intermediate_video_output_fps
    writer.init_video_stream("h264", fps=final_output_fps)

    # Save a few frames without text overlay for figure making, etc.
    if coarse_frames_interval is not None:
        coarse_frames_dir = output_path.parent / f"{output_path.stem}_coarse_frames"
        coarse_frames_dir.mkdir(exist_ok=True)

    for i, (i_nmf_frame, i_recsnippet_frame, i_fullrecording_frame) in tqdm(
        enumerate(frame_pairs), desc="Making summary video", total=len(frame_pairs)
    ):
        # Crop sim rendering (bottom camera)
        # sim_heading is already input-matched, so just use rec idx withing the snippet
        sim_heading = sim_headings_smooth[i_recsnippet_frame]
        sim_frame = sim_bottomcam_frames[i_nmf_frame]
        sim_frame_cropped = crop_single_image(
            sim_frame,
            sim_frame_center,
            -sim_heading + np.pi,
            bbox_sidelen=video_blockdim,
        )

        # Crop Spotlight recording
        spotlight_full_frame = spotlight_frames[i_fullrecording_frame]
        centerpos = rec_centerpos[i_recsnippet_frame]
        heading = rec_heading[i_recsnippet_frame]
        spotlight_frame_cropped = crop_single_image(
            spotlight_full_frame,
            centerpos,
            -heading + np.pi,
            bbox_sidelen=video_blockdim,
        )

        # Make trajectory visualization panels
        i_trajplotter = i_recsnippet_frame - frame_pairs[0][1]  # apply time offset
        rectraj_panels = rectraj_plotter.plot_snapshot(i_trajplotter)[:, :, :3]
        simtraj_panels = simtraj_plotter.plot_snapshot(i_trajplotter)[:, :, :3]

        # Add NMF side camera view
        nmf_sidecam_frame = sim_sidecam_frames[i_nmf_frame]
        rmin = int((nmf_sidecam_frame.shape[0] - 1.5 * video_blockdim) // 2)
        rmax = int(rmin + 1.5 * video_blockdim)
        cmin = int((nmf_sidecam_frame.shape[1] - video_blockdim) // 2)
        cmax = int(cmin + video_blockdim)
        nmf_sidecam_frame_cropped = nmf_sidecam_frame[rmin:rmax, cmin:cmax, :]

        # Put elements together into a frame
        col1 = np.concatenate([spotlight_frame_cropped, rectraj_panels], axis=0)
        col2 = np.concatenate([sim_frame_cropped, simtraj_panels], axis=0)
        out_frame = np.concatenate([col1, col2, nmf_sidecam_frame_cropped], axis=1)

        # Save coarse frames for later use in figures/presentations etc
        if coarse_frames_interval is not None and i % coarse_frames_interval == 0:
            filename = (
                f"frame{i_nmf_frame:04d}_"
                f"recsnippet{i_recsnippet_frame:04d}_"
                f"fullrec{i_fullrecording_frame:04d}.png"
            )
            Image.fromarray(out_frame).save(coarse_frames_dir / filename)

        # Add text labels
        out_img = Image.fromarray(out_frame)
        draw = ImageDraw.Draw(out_img)
        spotlight_str = "Spotlight recording"
        nmf_str = "NeuroMechFly replay"
        rec_fps_str = f"Recorded at {kinematic_snippet.data_fps} Hz"
        sim_fps_str = f"Simulated at {int(1 / sim_timestep):,} Hz"
        playspeed_str = f"Played at {final_output_playback_speed}x real-time"
        draw.text((20, 30), spotlight_str, fill=(255, 255, 255), font=font_normal)
        draw.text((20, 70), rec_fps_str, fill=(255, 255, 255), font=font_small)
        draw.text((20, 98), playspeed_str, fill=(255, 255, 255), font=font_small)
        draw.text((920, 30), nmf_str, fill=(255, 255, 255), font=font_normal)
        draw.text((920, 70), sim_fps_str, fill=(255, 255, 255), font=font_small)
        draw.text((920, 98), playspeed_str, fill=(255, 255, 255), font=font_small)
        draw.text((1820, 30), nmf_str, fill=(255, 255, 255), font=font_normal)
        draw.text((1820, 70), sim_fps_str, fill=(255, 255, 255), font=font_small)
        draw.text((1820, 98), playspeed_str, fill=(255, 255, 255), font=font_small)
        out_frame = np.array(out_img)

        writer.write_frame(out_frame)

    writer.close()
    rectraj_plotter.close()
    simtraj_plotter.close()


def plot_invkin_frame(
    sim_dir: Path,
    frame_among_full_recording: int,
    rawpred_color=REC_COLOR,
    fwdkin_color=SIM_COLOR,
    elev=40,
    azim=-60,
):
    with open(sim_dir / "sim_data.pkl", "rb") as f:
        data = pickle.load(f)
        kinematic_snippet = data["snippet"]

    with h5py.File(
        kinematic_snippet.exp_trial_dir
        / "poseforge_output/inverse_kinematics_output.h5",
        "r",
    ) as f:
        raw_world_xyz = f["rawpred_world_xyz"][:, :30, :].reshape(-1, 6, 5, 3)
        fwdkin_world_xyz = f["fwdkin_world_xyz"][:, :30, :].reshape(-1, 6, 5, 3)
    assert raw_world_xyz.shape == fwdkin_world_xyz.shape

    fwkin_snapshot = fwdkin_world_xyz[frame_among_full_recording, :, :]
    raw_snapshot = raw_world_xyz[frame_among_full_recording, :, :]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "3d"})
    root_offset = (raw_snapshot[:, 0, :] - fwkin_snapshot[:, 0, :])[:, None, :]
    raw_snapshot_aligned = raw_snapshot - root_offset
    for i in range(6):
        ax.plot(
            raw_snapshot_aligned[i, :, 0],
            raw_snapshot_aligned[i, :, 1],
            raw_snapshot_aligned[i, :, 2],
            linewidth=2,
            zorder=0,
            color=rawpred_color,
        )
        ax.plot(
            fwkin_snapshot[i, :, 0],
            fwkin_snapshot[i, :, 1],
            fwkin_snapshot[i, :, 2],
            linewidth=2,
            zorder=10,
            color=fwdkin_color,
        )
    ax.set_aspect("equal")
    ax.view_init(elev=elev, azim=azim)

    return fig, ax


def plot_claw_traj_by_swing_stance(sim_dir, gait_info, t_range=None):
    with open(sim_dir / "sim_data.pkl", "rb") as f:
        data = pickle.load(f)
        kinematic_snippet = data["snippet"]
        sim_results = data["sim_results"]

    trajs_info = align_smooth_decompose_trajs(
        kinematic_snippet, sim_results, t_range=t_range
    )
    origin_offset = trajs_info["origin_offset"]

    claw_xypos = gait_info["claw_xypos_arena"] - origin_offset
    swing_mask = gait_info["swing_mask"]
    claw_xypos_stance_only = claw_xypos.copy()
    claw_xypos_stance_only[swing_mask] = np.nan

    fig, ax = plt.subplots(figsize=(3, 3), tight_layout=True)
    for i in range(6):
        ax.plot(
            claw_xypos[:, i, 0],
            claw_xypos[:, i, 1],
            linewidth=1,
            color="#546a76",
            label="Swing" if i == 0 else None,
        )
        ax.plot(
            claw_xypos_stance_only[:, i, 0],
            claw_xypos_stance_only[:, i, 1],
            linewidth=1,
            color="#a23e48",
            zorder=10,
            label="Stance" if i == 0 else None,
        )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xyrange = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    xlim = np.mean(xlim) + np.array([-xyrange, xyrange]) * 0.5
    ylim = np.mean(ylim) + np.array([-xyrange, xyrange]) * 0.5
    scalebar_x0 = xlim[1] + 0.05 * (xlim[1] - xlim[0])
    scalebar_y = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax.plot(
        [scalebar_x0, scalebar_x0 + 1],
        [scalebar_y] * 2,
        color="black",
        linewidth=2,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.text(
        scalebar_x0 + 1.5,
        scalebar_y,
        "1 mm",
        color="black",
        fontsize=5,
        va="center",
        clip_on=False,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_aspect("equal")
    ax.set_title("Claw trajectories in arena")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    return fig, ax
