from wc26.common.plot import setup_matplotlib_params, palette

setup_matplotlib_params()

import pickle

import numpy as np
import polars as pl
import h5py
import optuna
import yaml
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import trange

from flygym import Simulation
from flygym.compose import (
    Fly,
    FlatGroundWorld,
    KinematicPose,
    ContactParams,
    ActuatorType,
)
from flygym.anatomy import Skeleton
from flygym.utils.math import Rotation3D

from spotlight_tools.calibration import SpotlightPositionMapper

import wc26.common.io as io
import wc26.kinematics.config as config
from wc26.kinematics.naming_convention import joint_name_seqikpy2flygym
from wc26.common.filter import median_filter_over_time


def optimize_sim_params(
    target_angles_arr_sim,
    thorax_pos_rec,
    rec_match_mask,
    n_trials=20,
    n_jobs=1,
    sg_window_sec=0.2,
    params_config=config.OPTIMIZABLE_PHYS_PARAMS,
    out_dir=config.PHYS_PARAMS_TUNING_DIR,
    fps=config.DATA_FPS,
):
    out_dir.mkdir(exist_ok=True, parents=True)

    def objective(trial):
        # Sample physical parameters to optimize over
        phys_params = {}
        for name, param_config in params_config.items():
            phys_params[name] = trial.suggest_float(name, *param_config["lim"])

        # Run simulation
        sim, fly = set_up_simulation(**phys_params)
        disable_pbar = n_jobs > 1
        sim_results = run_simulation(
            sim, fly, target_angles_arr_sim, disable_pbar=disable_pbar
        )

        # Compute trajectory match
        thorax_pos_sim = sim_results["thorax_pos"][rec_match_mask, :2]
        thorax_pos_sim = thorax_pos_sim - thorax_pos_sim[0, :]
        R, t, thorax_pos_rec_aligned, align_metrics = align_2d_traj(
            thorax_pos_rec, thorax_pos_sim, anchor_origin=False
        )
        traj_similarity_metrics, debug_vars = compute_traj_similarity(
            thorax_pos_sim, thorax_pos_rec_aligned, window_sec=sg_window_sec, fps=fps
        )
        debug_plot_path = out_dir / f"trial{trial.number:04d}_debug.png"
        make_debug_plots(
            traj_similarity_metrics, align_metrics, debug_vars, debug_plot_path
        )

        # Save output and metadata
        sim.renderer.save_video(out_dir / f"trial{trial.number:04d}.mp4")
        with open(out_dir / f"trial{trial.number:04d}.pkl", "wb") as f:
            metadata = {
                "phys_params": phys_params,
                "sim_results": sim_results,
                "traj_similarity_metrics": traj_similarity_metrics,
            }
            pickle.dump(metadata, f)

        print(f"Trial {trial.number}:", traj_similarity_metrics)
        # total = (
        #     align_metrics["normalized_rmse"]
        #     + traj_similarity_metrics["ruggedness_mismatch"]
        # )
        # return total
        return (
            align_metrics["normalized_rmse"],
            traj_similarity_metrics["ruggedness_mismatch"],
        )

    study = optuna.create_study(
        directions=["minimize", "minimize"], sampler=optuna.samplers.NSGAIISampler()
    )
    initial_params = {
        name: param_config["init"] for name, param_config in params_config.items()
    }
    study.enqueue_trial(initial_params)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # trials_df = study.trials_dataframe()
    # print("Best trial:", study.best_trial.number)
    # print("Best parameters:", study.best_params)
    # with open(out_dir / "best.yaml", "w") as f:
    #     yaml.dump(study.best_params, f)
    pareto_trials = study.best_trials
    res = {}
    print("Pareto-optimal trials:")
    for trial in pareto_trials:
        res[trial.number] = {"values": trial.values, "params": trial.params}
        print(f"  Trial {trial.number}:", trial.values, trial.params)
    with open(out_dir / "pareto_optimal_trials.yaml", "w") as f:
        yaml.dump(res, f)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "trials_summary.csv", index=False)


def run_simulation(sim, fly, target_angles_arr, disable_pbar=None):
    sim.reset()
    sim.warmup()
    actuator_ty = ActuatorType.POSITION

    body_pos_li = []
    actuator_forces_li = []
    n_steps = target_angles_arr.shape[0]
    for i_step in trange(n_steps, desc="Simulating", disable=disable_pbar):
        target_angles = target_angles_arr[i_step, :]
        sim.set_actuator_inputs(fly.name, actuator_ty, target_angles)
        sim.step()
        sim.render_as_needed()
        body_pos_li.append(sim.get_body_positions(fly.name).copy())
        actuator_forces_li.append(sim.get_actuator_forces(fly.name, actuator_ty).copy())
    # Postproces recorded simulation states
    thorax_idx = [dof.name for dof in fly.get_bodysegs_order()].index("c_thorax")
    thorax_pos = np.array(body_pos_li)[:, thorax_idx]

    actuator_forces_arr = np.array(actuator_forces_li)
    actuator_forces_dict = {
        dof_name: actuator_forces_arr[:, i_act]
        for i_act, dof_name in enumerate(fly.get_actuated_jointdofs_order(actuator_ty))
    }

    return {"thorax_pos": thorax_pos, "actuator_forces": actuator_forces_dict}


def set_up_simulation(
    joint_stiffness,
    joint_damping,
    actuator_gain,
    actuator_dampratio,
    actuator_timeconst_nsteps,
    sliding_friction,
    torsional_friction,
):
    # Create fly and add joints
    fly = Fly()
    fly.mjcf_root.option.integrator = "implicitfast"

    skeleton = Skeleton(
        axis_order=config.AXIS_ORDER, joint_preset=config.ARTICULATED_JOINTS
    )
    neutral_pose = KinematicPose(path=config.NEUTRAL_POSE_FILE)
    fly.add_joints(
        skeleton,
        neutral_pose=neutral_pose,
        stiffness=joint_stiffness,
        damping=joint_damping,
    )
    n_tarsus_overrides = 0
    for dof, mjcf_joint in fly.jointdof_to_mjcfjoint.items():
        if dof.child.link in ("tarsus2", "tarsus3", "tarsus4", "tarsus5"):
            mjcf_joint.stiffness = 10
            mjcf_joint.damping = 0.5
            n_tarsus_overrides += 1
    assert n_tarsus_overrides == 4 * 6, "error overriding tarsus joint params"

    # Add position actuators and set them to the neutral pose
    actuated_dofs_list = fly.skeleton.get_actuated_dofs_from_preset(
        config.ACTUATED_DOFS
    )
    fly.add_actuators(
        actuated_dofs_list,
        actuator_type=ActuatorType.POSITION,
        kp=actuator_gain,
        neutral_input=neutral_pose,
        dampratio=actuator_dampratio,
        timeconst=actuator_timeconst_nsteps * fly.mjcf_root.option.timestep,
        forcerange=(-30, 30),
    )

    # Add visuals
    fly.colorize()
    camera = fly.add_tracking_camera()

    # Create a world and spawn the fly
    world = FlatGroundWorld()
    spawn_position = (0, 0, config.SPAWN_HEIGHT)
    spawn_rotation = Rotation3D("quat", (1, 0, 0, 0))
    contact_params = ContactParams(
        sliding_friction=sliding_friction, torsional_friction=torsional_friction
    )
    world.add_fly(
        fly, spawn_position, spawn_rotation, ground_contact_params=contact_params
    )

    # Create a simulation and set up the renderer
    sim = Simulation(world)
    sim.set_renderer(
        camera,
        playback_speed=config.VIDEO_PLAYBACK_SPEED,
        output_fps=config.VIDEO_OUTPUT_FPS,
    )

    return sim, fly


def make_target_joint_angles_array(
    joint_angles_dict,
    dof_order_in_sim,
    rec_fps,
    sim_timestep,
):
    n_rec_steps = len(next(iter(joint_angles_dict.values())))
    sim_duration_sec = n_rec_steps / rec_fps
    n_sim_steps = int(sim_duration_sec / sim_timestep)

    # Make target angles array of shape (n_steps, n_dofs), using DoF order in simulation
    target_angles_arr_rec = np.full((n_rec_steps, len(dof_order_in_sim)), np.nan)
    for dof_name_seqikpy, angles in joint_angles_dict.items():
        dof_name_flygym = joint_name_seqikpy2flygym(dof_name_seqikpy)
        i_dof_sim = dof_order_in_sim.index(dof_name_flygym)
        target_angles_arr_rec[:, i_dof_sim] = angles
    assert not np.isnan(target_angles_arr_rec).any(), "error filling target angle array"

    # Interpolate to simulation timestep
    rec_times = np.arange(n_rec_steps) / rec_fps
    f = interp1d(
        rec_times,
        target_angles_arr_rec,
        axis=0,
        kind="linear",
        fill_value="extrapolate",
    )
    sim_times = np.arange(n_sim_steps) * sim_timestep
    target_angles_arr_sim = f(sim_times)

    # Make a mask of shape (n_sim_steps,) indicating which simulation steps roughly
    # correspond to frames in the recording
    matching_frameids_in_rec = np.linspace(
        0, n_rec_steps, n_sim_steps, endpoint=False, dtype=np.int32
    )
    rec_match_mask = np.diff(matching_frameids_in_rec, prepend=-1) != 0  # find id jumps

    return target_angles_arr_sim, rec_match_mask


def load_kinematics_data(walking_period_npz_file, frame_range=None):
    # Load inverse kinematics output and extract joint angles
    kin_data = np.load(walking_period_npz_file)
    joint_angles = kin_data["joint_angles"]  # (L, n_legs, n_dofs_per_leg)
    if frame_range is None:
        frame_range = (0, joint_angles.shape[0])
    joint_angles = joint_angles[slice(*frame_range), ...]
    joint_angles_dict = {
        f"{leg}{dof}": joint_angles[:, i_leg, i_dof]
        for i_leg, leg in enumerate(kin_data["legs_order"])
        for i_dof, dof in enumerate(kin_data["dofs_order_per_leg"])
    }

    # FlyGym uses an anatomical convention where left and right roll and yaw axis are
    # inverted (so positive roll always corresponds to rotating outward, for example).
    # SeqIKPy doesn't do this, so we need to flip right-side roll and yaw angles.
    for dof_name in joint_angles_dict:
        if dof_name[0] == "R" and dof_name.split("_")[-1] in ("roll", "yaw"):
            joint_angles_dict[dof_name] *= -1

    # Match this to the Spotlight experimental trial that the snippet belongs to
    exp_trial = str(kin_data["trial"])
    spotlight_trial_dir = io.get_inputs_dir() / "spotlight_trials" / exp_trial

    # The snippet of Spotlight kinematic recordings is already a slice of the full trial
    # (a perioud of continuous walking). So if frame_subselection is provided, we need
    # to offset the frame indices by start_idx of the pre-extracted inverse kinematics
    # snippet to get the range in the full Spotlight trial that it belongs to.
    offset = kin_data["start_idx"]
    frame_range_fulltrial = (frame_range[0] + offset, frame_range[1] + offset)

    # Load trajectory of the fly in the arena
    # Frames originally recorded by Spotlight are rotated and cropped around the fly
    # thorax before the smaller crop is fed into PoseForge models. So we need to
    # retrieve thorax positions in the original recording in pixel space (as predicted
    # by a tiny 3-keypoint SLEAP model)
    align_info_file = spotlight_trial_dir / "processed/behavior_alignment_transforms.h5"
    with h5py.File(align_info_file, "r") as f:
        rough_keypoints = list(f["keypoints_xy_pre_alignment"].attrs["keypoint_names"])
        thorax_idx = rough_keypoints.index("thorax")
        thorax_pos_px = f["keypoints_xy_pre_alignment"][
            slice(*frame_range_fulltrial), thorax_idx, :
        ]

    # Load translation stage positions at the time each frame was recorded
    beh_frame_metadata_file = (
        spotlight_trial_dir / "processed/behavior_frames_metadata.csv"
    )
    frame_metadata = pl.read_csv(beh_frame_metadata_file)
    frame_metadata = frame_metadata[slice(*frame_range_fulltrial)]
    stage_pos = frame_metadata.select("x_pos_mm_interp", "y_pos_mm_interp").to_numpy()

    # Load camera calibration parameters and map (translation stage pos, pixel corrds)
    # to physical coordinates in the arena
    calib_file = spotlight_trial_dir / "metadata/calibration_parameters_behavior.yaml"
    mapper = SpotlightPositionMapper(calib_file)
    thorax_pos_mm = mapper.stage_and_pixel_to_physical(stage_pos, thorax_pos_px)

    return joint_angles_dict, thorax_pos_mm, spotlight_trial_dir


def align_2d_traj(traj, traj_ref, anchor_origin: bool = True):
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
        A tuple of:
            R: (2, 2) rotation matrix.
            t: (2,) translation vector such that ``traj_aligned = traj @ R.T + t``.
            traj_aligned: transformed traj with shape (L, 2).
            metrics: dict with keys 'rmse', 'mae', 'median', 'max', 'normalized_rmse'
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

    return R, t, traj_aligned, metrics


def compute_traj_similarity(
    traj, traj_ref, window_sec=0.2, fps=config.DATA_FPS, polyorder=2
):
    # Smooth velocity time series using Savitzky-Golay filter
    dt = 1 / fps
    window_length = int(np.round(window_sec * fps))
    if window_length % 2 == 0:
        window_length += 1  # window_length must be odd for savgol_filter
    vel = savgol_filter(
        traj, window_length, polyorder=polyorder, deriv=1, delta=dt, axis=0
    )
    vel_ref = savgol_filter(
        traj_ref, window_length, polyorder=polyorder, deriv=1, delta=dt, axis=0
    )

    # Compute similarity score based on velocity match
    speed, ang_vel = cartesian_vel_to_speed_and_angular_vel(vel)
    speed_ref, ang_vel_ref = cartesian_vel_to_speed_and_angular_vel(vel_ref)
    mean_ref_speed = np.mean(speed_ref)
    if mean_ref_speed == 0:
        raise ValueError(
            "ref_traj has zero mean speed. Cannot compute speed mismatch score."
        )
    speed_mismatch = np.mean(np.abs(speed - speed_ref)) / mean_ref_speed
    ang_vel_mismatch = np.mean(np.abs(ang_vel - ang_vel_ref))

    # Compute score for ruggedness of the trajectory
    traj_len = compute_traj_len(traj)
    traj_smoothed = np.cumsum(vel, axis=0) * dt
    traj_smoothed_len = compute_traj_len(traj_smoothed)
    traj_ref_len = compute_traj_len(traj_ref)
    traj_ref_smoothed = np.cumsum(vel_ref, axis=0) * dt
    traj_ref_smoothed += traj_ref[0, :]  # in case alignment is not anchored at origin
    traj_ref_smoothed_len = compute_traj_len(traj_ref_smoothed)
    if traj_smoothed_len == 0 or traj_ref_smoothed_len == 0:
        raise ValueError(
            "Smoothed traj or ref_traj has 0 length. Cannot compute ruggedness score."
        )
    traj_ruggedness = traj_len / traj_smoothed_len
    traj_ref_ruggedness = traj_ref_len / traj_ref_smoothed_len
    ruggedness_ratio = traj_ruggedness / traj_ref_ruggedness
    # ruggedness_mismatch_score = max(ruggedness_ratio, 1 / ruggedness_ratio) - 1
    ruggedness_mismatch_score = np.log(ruggedness_ratio) ** 2

    metrics = {
        "speed_mismatch": float(speed_mismatch),
        "ang_vel_mismatch": float(ang_vel_mismatch),
        "ruggedness_mismatch": float(ruggedness_mismatch_score),
    }
    debug_vars = {
        "traj": traj,
        "traj_ref": traj_ref,
        "vel": vel,
        "vel_ref": vel_ref,
        "traj_smoothed": traj_smoothed,
        "traj_ref_smoothed": traj_ref_smoothed,
        "speed": speed,
        "speed_ref": speed_ref,
        "ang_vel": ang_vel,
        "ang_vel_ref": ang_vel_ref,
        "traj_len": traj_len,
        "traj_smoothed_len": traj_smoothed_len,
        "traj_ref_len": traj_ref_len,
        "traj_ref_smoothed_len": traj_ref_smoothed_len,
        "traj_ruggedness": traj_ruggedness,
        "traj_ref_ruggedness": traj_ref_ruggedness,
        "dt": dt,
    }
    return metrics, debug_vars


def compute_traj_len(traj):
    return np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()


def cartesian_vel_to_speed_and_angular_vel(vel):
    """Return speed and signed angular velocity from a velocity array (L, 2)."""
    speed = np.linalg.norm(vel, axis=1)  # (L,)

    # ang_vel = (vx * ay - vy * ax) / speed ** 2  -  signed curvature * speed
    # Use finite differences on velocity to get acceleration
    # (or you could pass deriv=2 through savgol separately)
    vx, vy = vel[:, 0], vel[:, 1]
    ax = np.gradient(vx)
    ay = np.gradient(vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        ang_vel = np.where(speed > 0, (vx * ay - vy * ax) / speed**2, 0.0)

    return speed, ang_vel


def make_debug_plots(traj_similarity_metrics, align_metrics, debug_vars, output_path):
    # fig, axes = plt.subplots(
    #     3,
    #     1,
    #     figsize=(6, 6),
    #     tight_layout=True,
    #     gridspec_kw={"height_ratios": [2, 1, 1]},
    # )
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    color_rec = palette[0]
    color_sim = palette[1]

    # Aligned trajectories
    # ax = axes[0]
    ax.plot(
        debug_vars["traj_ref"][:, 0],
        debug_vars["traj_ref"][:, 1],
        label="Recorded (raw)",
        color=color_rec,
        linestyle="--",
    )
    ax.plot(
        debug_vars["traj_ref_smoothed"][:, 0],
        debug_vars["traj_ref_smoothed"][:, 1],
        label="Recorded (smoothed)",
        color=color_rec,
        linestyle="-",
    )
    ax.plot(
        debug_vars["traj"][:, 0],
        debug_vars["traj"][:, 1],
        label="Simulated (raw)",
        color=color_sim,
        linestyle="--",
    )
    ax.plot(
        debug_vars["traj_smoothed"][:, 0],
        debug_vars["traj_smoothed"][:, 1],
        label="Simulated (smoothed)",
        color=color_sim,
        linestyle="-",
    )
    ax.set_title(
        "Aligned trajectories\n"
        f"ruggedness mismatch: {traj_similarity_metrics['ruggedness_mismatch']:.3f}\n"
        f"Normalized RMSE: {align_metrics['normalized_rmse']:.3f}"
    )
    ax.set_aspect("equal")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # # Scalar speed
    # ax = axes[1]
    # t_grid = np.arange(debug_vars["traj"].shape[0]) * debug_vars["dt"]
    # ax.plot(
    #     t_grid, debug_vars["speed_ref"], label="Recorded (smoothed)", color=color_rec
    # )
    # ax.plot(t_grid, debug_vars["speed"], label="Simulated (smoothed)", color=color_sim)
    # ax.set_title(f"Speed mismatch: {traj_similarity_metrics['speed_mismatch']:.3f}")
    # ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # ax.set_ylabel("Speed (mm/s)")
    # ax.set_xlabel("Time (s)")
    # ax.set_xlim(t_grid[0], t_grid[-1])

    # # Angular velocity
    # ax = axes[2]
    # ax.plot(
    #     t_grid, debug_vars["ang_vel_ref"], label="Recorded (smoothed)", color=color_rec
    # )
    # ax.plot(
    #     t_grid, debug_vars["ang_vel"], label="Simulated (smoothed)", color=color_sim
    # )
    # ax.set_title(
    #     f"Angular velocity mismatch: {traj_similarity_metrics['ang_vel_mismatch']:.3f}"
    # )
    # ax.set_ylabel("Angular velocity (rad/s)")
    # ax.set_xlabel("Time (s)")
    # ax.set_xlim(t_grid[0], t_grid[-1])

    fig.savefig(output_path)
    plt.close(fig)
