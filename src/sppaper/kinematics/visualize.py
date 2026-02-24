from sppaper.common.plot import setup_matplotlib_params

setup_matplotlib_params()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cmasher
from scipy.signal import medfilt

from flygym.anatomy import LEGS

import sppaper.kinematics.shared_constants as const
import sppaper.kinematics.trajectory as traj
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
    claw_xyz_color="#546a76",
    dof_angles_color="#a23e48",
    actuator_forces_color="#689829",
    medkernel_size=const.KIN_MEDFILTER_SIZE,
    ratelim_xyz=const.XYZ_RATELIM,
    ratelim_angles=const.JOINT_ANGLE_RATELIM,
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
        figsize=(6, 6),
        gridspec_kw={"height_ratios": [2, 3, 3, 3]},
        tight_layout=True,
    )

    ax = axes[0]
    sns.despine(ax=ax)
    claw_idx = -1
    for i_axis, axis in enumerate("xyz"):
        ts = fwdkin_xyz[:, i_leg, claw_idx, i_axis]
        span = np.percentile(ts, [1, 99])
        y_center = i_axis * claw_xyz_range + 0.5 * claw_xyz_range
        y_offset = np.mean(span) - y_center
        ax.plot(
            t_grid,
            ts - y_offset,
            linewidth=1,
            label=axis,
            color=claw_xyz_color,
            clip_on=False,
        )
        ax.text(
            -0.005 * (t_grid[-1] - t_grid[0]),
            y_center,
            AXIS_DISPLAY_NAMES[axis],
            va="center",
            ha="right",
            clip_on=False,
            color=claw_xyz_color,
            fontsize="small",
        )
    # Add scale bar
    ax.plot(
        [t_grid[-1] * 1.02] * 2, [0.2, 1.2], color="black", linewidth=2, clip_on=False
    )
    ax.text(t_grid[-1] * 1.04, 0.7, "1 mm", va="center", ha="left")
    # Configure y axis
    y_max = 3 * claw_xyz_range
    ax.set_ylim(y_max + 0.25 * claw_xyz_range, 0)
    ax.set_yticks([])
    ax.set_ylabel("Claw position", color=claw_xyz_color)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    # Configure x axis
    ax.set_xticklabels([])
    ax.set_xlim(t_grid[0], t_grid[-1] + 1 / kinematic_snippet.data_fps)

    ax = axes[1]
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
            linewidth=1,
            label=dof,
            color=dof_angles_color,
            clip_on=False,
        )
        ax.text(
            -0.005 * (t_grid[-1] - t_grid[0]),
            y_center,
            DOF_DISPLAY_NAMES[dof],
            va="center",
            ha="right",
            clip_on=False,
            color=dof_angles_color,
            fontsize="small",
        )
    # Add scale bar
    ax.plot(
        [t_grid[-1] * 1.02] * 2,
        [30, 75],
        color="black",
        linewidth=2,
        clip_on=False,
    )
    ax.text(t_grid[-1] * 1.04, 52.5, "45°", va="center", ha="left")
    # Configure y axis
    y_max = len(kinematic_snippet.metadata["dofs_order_per_leg"]) * dof_angles_range
    ax.set_ylim(y_max + 0.25 * dof_angles_range, 0)
    ax.set_yticks([])
    ax.set_ylabel("Joint angles", color=dof_angles_color)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    # Configure x axis
    ax.set_xticklabels([])
    ax.set_xlim(t_grid[0], t_grid[-1] + 1 / kinematic_snippet.data_fps)

    # Actuator force
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
            linewidth=1,
            label=dof,
            color=actuator_forces_color,
            zorder=10,
        )
        ax.text(
            -0.005 * (t_grid[-1] - t_grid[0]),
            y_offset,
            DOF_DISPLAY_NAMES[dof],
            va="center",
            ha="right",
            clip_on=False,
            color=actuator_forces_color,
            fontsize="small",
        )
    # Add scale bar
    ax.plot(
        [t_grid[-1] * 1.02] * 2,
        [-5, 15],
        color="black",
        linewidth=2,
        clip_on=False,
    )
    ax.text(t_grid[-1] * 1.04, 5, "20 μN", va="center", ha="left")
    # Configure y axis
    ax.set_ylim(6.5 * actuator_forces_range, -0.5 * actuator_forces_range)
    ax.set_yticks([])
    ax.set_ylabel("Forces applied", color=actuator_forces_color)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    # Configure x axis
    ax.set_xticklabels([])
    ax.set_xlim(t_grid[0], t_grid[-1] + 1 / kinematic_snippet.data_fps)

    # Load by leg
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
    ax.set_yticks(np.arange(6), leg_disp_names, fontsize="small")
    ax.set_ylabel("Weight load by leg", color="black")
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_xlabel("Time (s)")

    return fig, axes


def align_smooth_decompose_trajectories(
    kinematic_snippet: KinematicsSnippet,
    sim_results: dict,
    t_range=None,
    posvelxy_sg_window_sec=0.5,
    linspeed_sg_window_sec=0.5,
    sg_window_turnrate_sec=1.0,
):
    if t_range is not None:
        start_idx_before = kinematic_snippet.start_idx
        kinematic_snippet = kinematic_snippet.get_subselection(*t_range)
        steps_offset = kinematic_snippet.start_idx - start_idx_before
    else:
        steps_offset = 0
    slice_ = slice(steps_offset, steps_offset + len(kinematic_snippet))

    dt = 1 / kinematic_snippet.data_fps
    basetraj_sim = sim_results["thorax_pos_inputmatched"][slice_]
    align_info = traj.align_traj(kinematic_snippet.thorax_pos_mm, basetraj_sim)
    basetraj_rec = align_info["traj_aligned"]
    basetraj_sim_filtered, basevelxy_sim = traj.get_denoised_traj_and_vel(
        basetraj_sim, dt, sg_window_sec=posvelxy_sg_window_sec
    )
    basetraj_rec_filtered, basevelxy_rec = traj.get_denoised_traj_and_vel(
        basetraj_rec, dt, sg_window_sec=posvelxy_sg_window_sec
    )
    origin_offset = basetraj_sim[0].copy()
    basetraj_sim -= origin_offset
    basetraj_sim_filtered -= origin_offset
    basetraj_rec -= origin_offset
    basetraj_rec_filtered -= origin_offset

    baselinspeed_sim, baseheading_sim, baseturnrate_sim = traj.get_egocentric_vel(
        basetraj_sim,
        dt,
        linspeed_sg_window_sec=linspeed_sg_window_sec,
        turnrate_sg_window_sec=sg_window_turnrate_sec,
    )
    baselinspeed_rec, baseheading_rec, baseturnrate_rec = traj.get_egocentric_vel(
        basetraj_rec,
        dt,
        linspeed_sg_window_sec=linspeed_sg_window_sec,
        turnrate_sg_window_sec=sg_window_turnrate_sec,
    )

    return {
        "basetraj_sim": basetraj_sim,
        "basetraj_sim_filtered": basetraj_sim_filtered,
        "basevelxy_sim": basevelxy_sim,
        "baselinspeed_sim": baselinspeed_sim,
        "baseheading_sim": baseheading_sim,
        "baseturnrate_sim": baseturnrate_sim,
        "basetraj_rec": basetraj_rec,
        "basetraj_rec_filtered": basetraj_rec_filtered,
        "basevelxy_rec": basevelxy_rec,
        "baselinspeed_rec": baselinspeed_rec,
        "baseheading_rec": baseheading_rec,
        "baseturnrate_rec": baseturnrate_rec,
        "steps_offset": steps_offset,
        "slice": slice_,
        "origin_offset": origin_offset,
    }


def plot_trajectory(
    kinematic_snippet: KinematicsSnippet,
    trajs_info: dict,
    t_range=None,
    rec_color="#546a76",
    sim_color="#689829",
):
    if t_range is not None:
        kinematic_snippet = kinematic_snippet.get_subselection(*t_range)
    dt = 1 / kinematic_snippet.data_fps
    t_grid = np.arange(len(kinematic_snippet)) * dt

    fig = plt.figure(figsize=(12, 4), tight_layout=True)
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
    ax_traj.scatter([0], [0], color="black", label="Origin", s=15, zorder=10)
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
    ax_velx.set_xticklabels([])
    ax_velx.set_title("Velocity (global x)")
    sns.despine(ax=ax_velx)
    ax_vely.plot(
        t_grid, trajs_info["basevelxy_rec"][:, 1], label="Recorded", color=rec_color
    )
    ax_vely.plot(
        t_grid, trajs_info["basevelxy_sim"][:, 1], label="Simulated", color=sim_color
    )
    ax_vely.set_xlabel("Time (s)")
    ax_vely.set_ylabel("y vel. (mm/s)")
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
    ax_linspeed.set_xticklabels([])
    ax_linspeed.set_xlim(0, t_grid[-1] + dt)
    ax_linspeed.set_title("Linear speed")
    sns.despine(ax=ax_linspeed)
    ax_turnrate.plot(
        t_grid, trajs_info["baseturnrate_rec"], label="Recorded", color=rec_color
    )
    ax_turnrate.plot(
        t_grid, trajs_info["baseturnrate_sim"], label="Simulated", color=sim_color
    )
    ax_turnrate.set_xlabel("Time (s)")
    ax_turnrate.set_ylabel("Turn rate (rad/s)")
    ax_turnrate.set_xlim(0, t_grid[-1] + dt)
    ax_turnrate.set_title("Turn rate")
    sns.despine(ax=ax_turnrate)

    return fig, (ax_traj, ax_velx, ax_vely, ax_linspeed, ax_turnrate)
