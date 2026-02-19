from wc26.common.plot import setup_matplotlib_params

setup_matplotlib_params()

import pickle

import numpy as np
import polars as pl
import h5py
import optuna
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
import wc26.kinematics.shared_constants as const
from wc26.kinematics.naming_convention import joint_name_seqikpy2flygym


def optimize_sim_params(
    study_name,
    target_angles_arr_sim,
    thorax_pos_rec,
    rec_match_mask,
    params_config,
    out_dir,
    n_trials=20,
    n_jobs=1,
    window_sec=0.2,
    angvel_window_2ndpass_sec=0.1,
    fps=const.DATA_FPS,
    w_speed=1,
    w_angvel=2,
    w_ruggedness=50,
    w_align_nrmse=200,
    multiobjective=False,
    decompose_traj_mismatch=False,
    load_if_exists=False,
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
        metrics, debug_vars = compute_traj_similarity(
            sim_results["thorax_pos"][rec_match_mask, :2],
            thorax_pos_rec,
            window_sec=window_sec,
            angvel_window_2ndpass_sec=angvel_window_2ndpass_sec,
            fps=fps,
            w_speed=w_speed,
            w_angvel=w_angvel,
            w_align_nrmse=w_align_nrmse,
            w_ruggedness=w_ruggedness,
            decompose_traj_mismatch=decompose_traj_mismatch,
        )

        # Save output and metadata
        stem = f"trial{trial.number:04d}"
        make_debug_plots(
            metrics,
            debug_vars,
            decompose_traj_mismatch=decompose_traj_mismatch,
            output_path=out_dir / f"{stem}.png",
        )
        sim.renderer.save_video(out_dir / f"{stem}.mp4")
        sim.renderer.mj_renderer.close()  # gc manually - fixes weird renderer __del__ error
        del sim.renderer
        with open(out_dir / f"{stem}.pkl", "wb") as f:
            metadata = {
                "phys_params": phys_params,
                "sim_results": sim_results,
                "metrics": metrics,
                "debug_vars": debug_vars,
            }
            pickle.dump(metadata, f)

        # Return final metrics for optimization
        if multiobjective:
            return metrics["traj_mismatch"], metrics["ruggedness_mismatch"]
        else:
            return metrics["total"]

    # Set up Optuna study
    storage_path = f"sqlite:///{out_dir.absolute() / 'optuna_study.db'}"
    if multiobjective:
        study = optuna.create_study(
            study_name=study_name,
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(),
            storage=storage_path,
            load_if_exists=load_if_exists,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=storage_path,
            load_if_exists=load_if_exists,
        )

    # Enqueue initial trial with reasonable defaults
    initial_params = {
        name: param_config["init"] for name, param_config in params_config.items()
    }
    study.enqueue_trial(initial_params)

    # Run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Summarize and save best trial(s)
    if multiobjective:
        best_trials = study.best_trials
        print("Pareto optimal trials:")
        res = {}
        for trial in best_trials:
            print(f"Trial {trial.number}: value={trial.values}, params={trial.params}")
            res[trial.number] = {
                "value": trial.values,
                "params": trial.params,
            }
        with open(out_dir / "best.yaml", "w") as f:
            yaml.dump(res, f)
    else:
        trial = study.best_trial
        print(f"Best trial: {trial.number}, value={trial.value}, params={trial.params}")
        with open(out_dir / "best.yaml", "w") as f:
            yaml.dump(
                {"trial": trial.number, "value": trial.value, "params": trial.params}, f
            )

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
    passive_tarsal_stiffness=const.PASSIVE_TARSAL_STIFFNESS,
    passive_tarsal_damping=const.PASSIVE_TARSAL_DAMPING,
):
    # Create fly and add joints
    fly = Fly()
    fly.mjcf_root.option.integrator = "implicitfast"

    skeleton = Skeleton(
        axis_order=const.AXIS_ORDER, joint_preset=const.ARTICULATED_JOINTS
    )
    neutral_pose = KinematicPose(path=const.NEUTRAL_POSE_FILE)
    fly.add_joints(
        skeleton,
        neutral_pose=neutral_pose,
        stiffness=joint_stiffness,
        damping=joint_damping,
    )
    n_tarsus_overrides = 0
    for dof, mjcf_joint in fly.jointdof_to_mjcfjoint.items():
        if dof.child.link in ("tarsus2", "tarsus3", "tarsus4", "tarsus5"):
            mjcf_joint.stiffness = passive_tarsal_stiffness
            mjcf_joint.damping = passive_tarsal_damping
            n_tarsus_overrides += 1
    assert n_tarsus_overrides == 4 * 6, "error overriding tarsus joint params"

    # Add position actuators and set them to the neutral pose
    actuated_dofs_list = fly.skeleton.get_actuated_dofs_from_preset(const.ACTUATED_DOFS)
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
    spawn_position = (0, 0, const.SPAWN_HEIGHT)
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
        playback_speed=const.VIDEO_PLAYBACK_SPEED,
        output_fps=const.VIDEO_OUTPUT_FPS,
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
    traj,
    traj_ref,
    window_sec,
    angvel_window_2ndpass_sec,
    fps,
    w_speed=1,
    w_angvel=2,
    w_align_nrmse=200,
    w_ruggedness=50,
    polyorder=2,
    kpct=90,
    decompose_traj_mismatch=False,
):
    # Compute integer window size
    dt = 1 / fps
    window_size = int(np.round(window_sec * fps))
    if window_size % 2 == 0:
        window_size += 1  # window_size must be odd for savgol_filter
    if angvel_window_2ndpass_sec is not None:
        window_size_angvel_2ndpass = int(np.round(angvel_window_2ndpass_sec * fps))
        if window_size_angvel_2ndpass % 2 == 0:
            window_size_angvel_2ndpass += 1  # must be odd for savgol_filter

    metrics = {"total": 0}
    debug_vars = {"traj": traj, "traj_ref": traj_ref, "dt": dt}

    if decompose_traj_mismatch:
        # Compute match in egocentric kinematics (forward speed and turn rate time series)
        speed, angvel = traj_to_egocentric_kinematics(
            traj, dt, window_size, window_size_angvel_2ndpass, polyorder
        )
        speed_ref, angvel_ref = traj_to_egocentric_kinematics(
            traj_ref, dt, window_size, window_size_angvel_2ndpass, polyorder
        )
        speed_mismatch_ts = np.abs(speed - speed_ref)
        angvel_mismatch_ts = np.abs(angvel - angvel_ref)
        speed_err_mean = mean_of_lowest_kpct(speed_mismatch_ts, kpct)
        angvel_err_mean = mean_of_lowest_kpct(angvel_mismatch_ts, kpct)
        traj_mismatch_total = w_speed * speed_err_mean + w_angvel * angvel_err_mean

        metrics["speed_mismatch"] = speed_err_mean
        metrics["angvel_mismatch"] = angvel_err_mean
        metrics["traj_mismatch"] = traj_mismatch_total
        metrics["total"] += traj_mismatch_total
        debug_vars["speed"] = speed
        debug_vars["angvel"] = angvel
        debug_vars["speed_ref"] = speed_ref
        debug_vars["angvel_ref"] = angvel_ref
        debug_vars["speed_mismatch_ts_weighted"] = w_speed * speed_mismatch_ts
        debug_vars["angvel_mismatch_ts_weighted"] = w_angvel * angvel_mismatch_ts
        debug_vars["speed_mismatch_weighted"] = w_speed * speed_err_mean
        debug_vars["angvel_mismatch_weighted"] = w_angvel * angvel_err_mean

    else:
        # Compute trajectory alignment metrics
        traj_zeroed = traj - traj[0, :]
        _, _, traj_ref_aligned, alignment_metrics = align_2d_traj(
            traj_ref, traj_zeroed, anchor_origin=False
        )
        speed_err_mean = np.nan
        angvel_err_mean = np.nan
        align_nrmse = alignment_metrics["normalized_rmse"]

        metrics["traj_mismatch"] = align_nrmse
        metrics["total"] += w_align_nrmse * align_nrmse
        debug_vars["align_nrmse_weighted"] = w_align_nrmse * align_nrmse

    # Compute ruggedness of trajectory
    cartvel_smoothed = savgol_filter(
        traj, window_size, polyorder, deriv=1, delta=dt, axis=0
    )
    traj_smoothed = np.cumsum(cartvel_smoothed, axis=0) * dt + traj[0, :]
    traj_len = compute_traj_len(traj)
    traj_smoothed_len = compute_traj_len(traj_smoothed)

    cartvel_ref_smoothed = savgol_filter(
        traj_ref, window_size, polyorder, deriv=1, delta=dt, axis=0
    )
    traj_ref_smoothed = np.cumsum(cartvel_ref_smoothed, axis=0) * dt + traj_ref[0, :]
    traj_ref_len = compute_traj_len(traj_ref)
    traj_ref_smoothed_len = compute_traj_len(traj_ref_smoothed)

    lengths = [traj_len, traj_smoothed_len, traj_ref_len, traj_ref_smoothed_len]
    if any(l == 0 for l in lengths):
        raise ValueError("At least one trajectory has 0 length. This shouldn't happen.")
    traj_ruggedness = traj_len / traj_smoothed_len
    traj_ref_ruggedness = traj_ref_len / traj_ref_smoothed_len
    ruggedness_mismatch = np.abs(np.log(traj_ruggedness / traj_ref_ruggedness))

    metrics["ruggedness_mismatch"] = ruggedness_mismatch
    metrics["total"] += w_ruggedness * ruggedness_mismatch
    debug_vars["traj_smoothed"] = traj_smoothed
    debug_vars["traj_ref_smoothed"] = traj_ref_smoothed
    debug_vars["ruggedness_mismatch_weighted"] = w_ruggedness * ruggedness_mismatch

    return metrics, debug_vars


def compute_traj_len(traj):
    return np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()


def mean_of_lowest_kpct(arr, kpct):
    """Compute the mean of the lowest kpct% of values in arr."""
    if len(arr.shape) != 1:
        raise ValueError("Input array must be 1D")
    k = max(1, int(arr.size * kpct / 100))
    return np.mean(np.partition(arr, k)[:k])


def traj_to_egocentric_kinematics(
    traj, dt, window_size, window_size_angvel_2ndpass=None, polyorder=2
):
    """Compute ego-centric speed and angular velocity from a 2D trajectory.

    Uses Savitzky-Golay filtering throughout to avoid double-differentiation:
    speed comes from a single SG deriv=1 pass on position, and angular velocity
    comes from a single SG deriv=1 pass on the unwrapped heading angle.

    Args:
        traj: (L, 2) raw world-frame x-y positions.
        dt: time step in seconds (1 / fps).
        window_size: SG window size (number of samples, must be odd) used
            for speed computation.
        window_size_angvel_2ndpass: SG window size (number of samples, must be odd) used
            for angular velocity computation in a second pass to reduce noise.
        polyorder: SG polynomial order.

    Returns:
        speed:   (L,) scalar forward speed in units of traj / second.
        angvel: (L,) signed angular velocity in rad/s. Positive = turning
                 counter-clockwise.
    """
    # Ensure odd window lengths
    if window_size % 2 == 0:
        window_size += 1
    if window_size_angvel_2ndpass is not None and window_size_angvel_2ndpass % 2 == 0:
        window_size_angvel_2ndpass += 1

    # Speed: one SG pass on position
    vel = savgol_filter(traj, window_size, polyorder, deriv=1, delta=dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    # Angular velocity: differentiate heading directly, avoiding double-differentiation
    # Unwrap to remove +-pi discontinuities before filtering
    heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    if window_size_angvel_2ndpass is None:
        angvel = np.gradient(heading, dt)
    else:
        angvel = savgol_filter(
            heading, window_size_angvel_2ndpass, polyorder, deriv=1, delta=dt
        )

    return speed, angvel


def make_debug_plots(
    traj_similarity_metrics, debug_vars, decompose_traj_mismatch=False, output_path=None
):
    if decompose_traj_mismatch:
        fig, axes = _make_debug_plots_decompose_traj_mismatch(
            traj_similarity_metrics, debug_vars, output_path
        )
    else:
        fig, axes = _make_debug_plots_global_traj_mismatch(
            traj_similarity_metrics, debug_vars, output_path
        )

    if output_path is not None:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        return fig, axes


def _make_debug_plots_decompose_traj_mismatch(
    traj_similarity_metrics, debug_vars, output_path=None
):
    # Align ref trajectory to simulated trajectory for better visual comparison
    traj_sim = debug_vars["traj"]
    traj_sim = traj_sim - traj_sim[0, :]  # anchor to origin
    traj_ref_unaligned = debug_vars["traj_ref"]
    _, _, traj_ref, _ = align_2d_traj(traj_ref_unaligned, traj_sim, anchor_origin=True)
    n_steps = traj_sim.shape[0]
    t_grid = np.arange(n_steps) * debug_vars["dt"]

    # Set up figure
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(
        3, 2, figure=fig, width_ratios=[1, 0.7], hspace=0.3, wspace=0.2
    )
    ax_traj = plt.subplot(gs[:, 0])
    ax_speed = plt.subplot(gs[0, 1])
    ax_angvel = plt.subplot(gs[1, 1])
    ax_scores = plt.subplot(gs[2, 1])
    color_rec = "#546A76"
    color_sim = "#FC7A1E"
    color_speed = "#79A63D"
    color_angvel = "#B7CF98"
    color_align_nrmse = color_speed
    color_ruggedness = "#F7AEF8"

    # Aligned trajectories
    ax_traj.plot(traj_ref[:, 0], traj_ref[:, 1], label="Recorded", color=color_rec)
    ax_traj.plot(traj_sim[:, 0], traj_sim[:, 1], label="Simulated", color=color_sim)
    ax_traj.scatter([0], [0], color="black", marker="o", label="Origin", zorder=10)
    ax_traj.set_aspect("equal")
    xlim = ax_traj.get_xlim()
    ylim = ax_traj.get_ylim()
    xcenter = np.mean(xlim)
    ycenter = np.mean(ylim)
    max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])  # already has some padding
    half_range = max_range / 2
    ax_traj.set_xlim(xcenter - half_range, xcenter + half_range)
    ax_traj.set_ylim(ycenter - half_range, ycenter + half_range)
    ax_traj.legend(frameon=False)

    # Linear speed
    ax_speed.plot(t_grid, debug_vars["speed_ref"], label="Recorded", color=color_rec)
    ax_speed.plot(t_grid, debug_vars["speed"], label="Simulated", color=color_sim)
    ax_speed.set_title("Linear speed")
    ax_speed.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax_speed.set_ylabel("Speed (mm/s)")
    ax_speed.set_xticklabels([])
    ax_speed.set_xlim(t_grid[0], t_grid[-1])
    ax_speed.set_ylim(bottom=0)

    # Angular velocity
    angvel_ref_turns = debug_vars["angvel_ref"] / (2 * np.pi)
    angvel_turns = debug_vars["angvel"] / (2 * np.pi)
    ax_angvel.axhline(0, color="black", linestyle="-", linewidth=1, zorder=-10)
    ax_angvel.plot(t_grid, angvel_ref_turns, label="Recorded", color=color_rec)
    ax_angvel.plot(t_grid, angvel_turns, label="Simulated", color=color_sim)
    ax_angvel.set_title("Turn rate")
    ax_angvel.set_ylabel("Ang. vel. (turns/s)")
    ax_angvel.set_xticklabels([])
    ax_angvel.set_xlim(t_grid[0], t_grid[-1])
    max_mag = np.abs(ax_angvel.get_ylim()).max()
    ax_angvel.set_ylim(-max_mag, max_mag)

    # Similarity scores
    ruggedness_mismatch_weighted = (
        np.ones(n_steps) * debug_vars["ruggedness_mismatch_weighted"]
    )
    speed_mismatch_ts_weighted = debug_vars["speed_mismatch_ts_weighted"]
    angvel_mismatch_ts_weighted = debug_vars["angvel_mismatch_ts_weighted"]
    ax_scores.stackplot(
        t_grid,
        speed_mismatch_ts_weighted,
        angvel_mismatch_ts_weighted,
        ruggedness_mismatch_weighted,
        labels=["Speed", "Angular velocity", "Ruggedness"],
        colors=[color_speed, color_angvel, color_ruggedness],
    )
    ax_scores.set_title("Mismatch metric breakdown")
    ax_scores.set_xlabel("Time (s)")
    ax_scores.set_ylabel("Metric (a.u.)")
    ax_scores.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax_scores.set_xlim(t_grid[0], t_grid[-1])
    score_text = (
        f"Total mismatch: {traj_similarity_metrics['total']:.2e}\n"
        f"Speed term: {debug_vars['speed_mismatch_weighted']:.2e}\n"
        f"Ang. vel. term: {debug_vars['angvel_mismatch_weighted']:.2e}\n"
        f"Ruggedness term: {debug_vars['ruggedness_mismatch_weighted']:.2e}"
    )
    ax_scores.text(
        0.99,
        0.97,
        score_text,
        transform=ax_scores.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize="small",
    )

    axes = {
        "traj": ax_traj,
        "speed": ax_speed,
        "angvel": ax_angvel,
        "scores": ax_scores,
    }
    return fig, axes


def _make_debug_plots_global_traj_mismatch(
    traj_similarity_metrics, debug_vars, output_path=None
):
    # Align ref trajectory to simulated trajectory for better visual comparison
    traj_sim = debug_vars["traj"]
    traj_sim = traj_sim - traj_sim[0, :]  # anchor to origin
    traj_ref_unaligned = debug_vars["traj_ref"]
    _, _, traj_ref, _ = align_2d_traj(traj_ref_unaligned, traj_sim, anchor_origin=True)

    # Set up figure
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    color_rec = "#546A76"
    color_sim = "#FC7A1E"

    # Aligned trajectories
    ax.plot(traj_ref[:, 0], traj_ref[:, 1], label="Recorded", color=color_rec)
    ax.plot(traj_sim[:, 0], traj_sim[:, 1], label="Simulated", color=color_sim)
    ax.scatter([0], [0], color="black", marker="o", label="Origin", zorder=10)
    ax.set_aspect("equal")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xcenter = np.mean(xlim)
    ycenter = np.mean(ylim)
    max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0])  # already has some padding
    half_range = max_range / 2
    ax.set_xlim(xcenter - half_range, xcenter + half_range)
    ax.set_ylim(ycenter - half_range, ycenter + half_range)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    score_text = (
        f"Total mismatch: {traj_similarity_metrics['total']:.2e}\n"
        f"Trajectory nRMSE: {debug_vars['align_nrmse_weighted']:.2e}\n"
        f"Ruggedness term: {debug_vars['ruggedness_mismatch_weighted']:.2e}"
    )
    ax.text(
        1.04,
        0.02,
        score_text,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize="medium",
    )

    return fig, ax
