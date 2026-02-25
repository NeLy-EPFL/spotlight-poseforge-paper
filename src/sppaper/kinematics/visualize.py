from sppaper.common.plot import setup_matplotlib_params

setup_matplotlib_params()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cmasher
import imageio as iio
from scipy.signal import medfilt
from tqdm import trange

from flygym.anatomy import LEGS

import sppaper.kinematics.shared_constants as const
from sppaper.kinematics.data import KinematicsSnippet

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
    kinematic_snippet: KinematicsSnippet,
    sim_results: dict,
    leg,
    t_range=None,
    claw_xyz_range=1.5,
    dof_angles_range=90,
    actuator_forces_range=80,
    contact_force_vmax=8,
    claw_xyz_color=CLAW_XYZ_COLOR,
    dof_angles_color=DOF_ANGLES_COLOR,
    actuator_forces_color=ACTUATOR_FORCES_COLOR,
    medkernel_size=const.KIN_MEDFILTER_SIZE,
    ratelim_xyz=const.XYZ_RATELIM,
    ratelim_angles=const.JOINT_ANGLE_RATELIM,
    xticks_interval=0.5,
):
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
        4,
        1,
        figsize=(100 * MM_TO_IN, 100 * MM_TO_IN),
        gridspec_kw={"height_ratios": [2, 3, 3, 3]},
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
        [t_grid[-1] * 1.02] * 2, [0.2, 1.2], color="black", linewidth=1, clip_on=False
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

    # ===== Actuator force =====
    yticks = []
    yticklabels = []
    ax = axes[2]
    sns.despine(ax=ax)
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
        [-5, 15],
        color="black",
        linewidth=1,
        clip_on=False,
    )
    ax.text(t_grid[-1] * 1.03, 5, "20 μN", va="center", ha="left", fontsize="small")

    # Configure y axis
    ax.set_ylim(6.5 * actuator_forces_range, -0.5 * actuator_forces_range)
    ax.set_yticks(yticks, yticklabels)
    ax.set_ylabel("Forces applied", color=actuator_forces_color)
    ax.yaxis.set_label_coords(-0.13, 0.5)

    # Configure x axis
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlim(t_grid[0], t_grid[-1] + 1 / kinematic_snippet.data_fps)

    # ===== Load by leg =====
    ax = axes[3]
    ts = sim_results["ground_contacts"]["forces_world"][steps_offset:, :, 2].copy() * -1
    ts[np.isnan(ts)] = 0
    ts = reduce_timeseries_sim2rec(
        ts, sim_results["ctrl_update_mask"][steps_offset:], stride=3
    )
    im = ax.imshow(
        np.repeat(ts.T, axis=0, repeats=50),
        aspect="auto",
        interpolation="none",
        extent=[t_grid[0], t_grid[-1] + 1 / kinematic_snippet.data_fps, 5.5, -0.5],
        vmin=0,
        vmax=contact_force_vmax,
        cmap=cmasher.lavender,
    )
    cax = ax.inset_axes([1.02, 0, 0.03, 1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([0, 4, 8], labels=["0", "4", "8 μN"])
    leg_disp_names = [x.upper() for x in LEGS]
    ax.set_yticks(np.arange(6), leg_disp_names)
    ax.set_ylabel("Weight load by leg", color="black")
    ax.yaxis.set_label_coords(-0.13, 0.5)
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlabel("Time (s)")

    return fig, axes


def plot_trajectory(
    kinematic_snippet: KinematicsSnippet,
    trajs_info: dict,
    t_range=None,
    rec_color=REC_COLOR,
    sim_color=SIM_COLOR,
    xticks_interval=0.5,
):
    if t_range is not None:
        kinematic_snippet = kinematic_snippet.get_subselection(*t_range)
    dt = 1 / kinematic_snippet.data_fps
    t_grid = np.arange(len(kinematic_snippet)) * dt

    last_xtick = t_grid[-1] + 1 / kinematic_snippet.data_fps
    xticks = list(np.arange(0, last_xtick + 1e-6, xticks_interval))
    xticklabels = [f"{x:.1f}" for x in xticks]
    xticklabels[-1] += " s"

    fig = plt.figure(figsize=(140 * MM_TO_IN, 50 * MM_TO_IN), tight_layout=True)
    gs = gridspec.GridSpec(2, 3)
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_velx = fig.add_subplot(gs[0, 1])
    ax_vely = fig.add_subplot(gs[1, 1])
    ax_linspeed = fig.add_subplot(gs[0, 2])
    ax_turnrate = fig.add_subplot(gs[1, 2])

    # Trajectories
    ax_traj.plot(
        trajs_info["basetraj_rec"][:, 0],
        trajs_info["basetraj_rec"][:, 1],
        label="Recorded (raw)",
        color=rec_color,
        linestyle="--",
    )
    ax_traj.plot(
        trajs_info["basetraj_rec_filtered"][:, 0],
        trajs_info["basetraj_rec_filtered"][:, 1],
        label="Recorded (smoothed)",
        color=rec_color,
        linestyle="-",
    )
    ax_traj.plot(
        trajs_info["basetraj_sim"][:, 0],
        trajs_info["basetraj_sim"][:, 1],
        label="Simulated (raw)",
        color=sim_color,
        linestyle="--",
    )
    ax_traj.plot(
        trajs_info["basetraj_sim_filtered"][:, 0],
        trajs_info["basetraj_sim_filtered"][:, 1],
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
    ax_velx.plot(
        t_grid, trajs_info["basevelxy_rec"][:, 0], label="Recorded", color=rec_color
    )
    ax_velx.plot(
        t_grid, trajs_info["basevelxy_sim"][:, 0], label="Simulated", color=sim_color
    )
    ax_velx.set_ylabel("x vel. (mm/s)")
    ax_velx.set_xlim(0, t_grid[-1] + dt)
    ax_velx.set_xticks(xticks, xticklabels)
    ax_velx.set_title("Velocity (global x)")
    sns.despine(ax=ax_velx)
    ax_vely.plot(
        t_grid, trajs_info["basevelxy_rec"][:, 1], label="Recorded", color=rec_color
    )
    ax_vely.plot(
        t_grid, trajs_info["basevelxy_sim"][:, 1], label="Simulated", color=sim_color
    )
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


class TrajectoryVideoPlotter:
    def __init__(
        self,
        xypos_ts,
        heading_ts,
        linspeed_ts,
        turnrate_ts,
        dt,
        datacolor="tab:blue",
        fgcolor="#aaaaaa",
        extent=20,
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