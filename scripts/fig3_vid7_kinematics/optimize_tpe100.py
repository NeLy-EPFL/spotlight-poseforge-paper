import multiprocessing as mp
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from flygym.anatomy import LEGS

from sppaper.common.resources import get_poseforge_datadir, get_outputs_dir
from sppaper.kinematics.shared_constants import DATA_FPS
from sppaper.kinematics.nmf_sim import NeuroMechFlyReplayManager
from sppaper.kinematics.data import (
    KinematicsDataset,
    align_smooth_decompose_trajs,
    get_gait_info,
)
from sppaper.kinematics.visualize import plot_trajectory, make_replay_video, reduce_timeseries_sim2rec

# ── Constants ─────────────────────────────────────────────────────────────────

KPT_3D_OUTPUT_BASEDIR = (
    get_poseforge_datadir()
    / "pose_estimation/keypoints3d/trial_20251118a/production/epoch19_step9167/"
)
OUTPUT_DIR = get_outputs_dir() / "neuromechfly_replay/optim/"
MIN_XY_CONF = 0.58
MASK_DENOISE_KERNEL_SIZE_SEC = 0.1
MIN_DURATION_SEC = 1
WALKING_PERIOD_IDX = 21                  # same snippet as make_figures_and_videos.py
WALKING_PERIOD_TIMERANGE = (0.5, 2.5)   # same t_range as make_figures_and_videos.py
PASSIVE_TARSUS_STIFFNESS = 10
PASSIVE_TARSUS_DAMPING = 0.5

# Test-run params — copied verbatim from replay_walking.py
TEST_ACTUATOR_GAIN = 150
TEST_JOINT_DAMPING = 0.5
TEST_SLIDING_FRICTION = 2.0
TEST_LEG_ADHESION_FORCE = 1.0   # scalar; will be multiplied by per-leg gain below
TEST_LEG_ADHESION_GAIN = {
    "lf": 1.0, "lm": 1.0, "lh": 0.6,
    "rf": 1.0, "rm": 1.0, "rh": 0.6,
}

# Optimization hyper-parameters
TURNRATE_WEIGHT = 3
CONTACT_FORCE_DISP_THRESHOLD = 0.2
N_WORKERS = 4
GRF_USE_ZONLY = False  # False = total 3D force magnitude; True = z-component only

INITIAL_TRIAL_PARAMS = {
    "actuator_gain": 150,
    "joint_damping": 0.5,
    "sliding_friction": 2.0,
    "adhforce_fmlegs": 1.0,       # gain=1.0, force=1.0 → effective=1.0 for f/m legs
    "adhforce_hlegs_scale": 0.6,  # gain=0.6, force=1.0 → effective=0.6 for h legs
}


# ── Helper functions ──────────────────────────────────────────────────────────

def mse_error(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean((a - b) ** 2))


def intersection_over_union(mask_a, mask_b):
    """Element-wise IOU for two boolean arrays of the same shape."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 1.0


def compute_traj_error(trajs_info, turnrate_weight):
    """Weighted MSE between recorded and simulated linear speed + turn rate."""
    linspeed_error = mse_error(
        trajs_info["baselinspeed_rec"], trajs_info["baselinspeed_sim"]
    )
    turnrate_error = mse_error(
        trajs_info["baseturnrate_rec"], trajs_info["baseturnrate_sim"]
    )
    return linspeed_error + turnrate_weight * turnrate_error


def get_steps_offset(kinematic_snippet, t_range):
    """Return the index into sim_results arrays corresponding to t_range[0].

    Mirrors the logic used in plot_time_series / align_smooth_decompose_trajs:
    get_subselection(start_sec, end_sec) sets new start_idx = old_start_idx + int(start_sec * fps),
    so steps_offset = new.start_idx - old.start_idx = int(start_sec * fps).
    """
    if t_range is None:
        return 0
    snippet_sub = kinematic_snippet.get_subselection(*t_range)
    return snippet_sub.start_idx - kinematic_snippet.start_idx


def get_adjusted_grf(sim_results, adhesion_force_per_leg, steps_offset, use_zonly: bool):
    """GRF (T_kin, n_legs) at kinematic rate, with per-leg adhesion subtracted.

    forces_world and ctrl_update_mask are stored at sim rate. steps_offset is in
    kinematic frames (330 Hz). We must therefore reduce the full sim-rate array to
    kinematic rate first, then slice — the same pattern used for actuator_forces in
    visualize.plot_time_series (lines 231-234).

    Applying steps_offset directly to the sim-rate arrays before reduction would only
    skip ~steps_offset sim steps (~0.016s for offset=165), not the intended 0.5s.

    Args:
        use_zonly: If True, use the z-component of forces_world (negated) as the GRF —
            the vertical reaction force on flat ground. If False, use the full 3D force
            magnitude (np.linalg.norm over the xyz components), which includes shear.
    """
    forces_world = sim_results["ground_contacts"]["forces_world"]  # (T_sim, n_legs, 3)
    if use_zonly:
        grf_ts = forces_world[:, :, 2] * -1  # upward reaction force
    else:
        grf_ts = np.linalg.norm(forces_world, axis=-1)  # total 3D magnitude
    grf_ts = grf_ts.copy()
    grf_ts[np.isnan(grf_ts)] = 0.0
    # Reduce the full array from sim rate to kinematic rate, then slice
    grf_ts = reduce_timeseries_sim2rec(grf_ts, sim_results["ctrl_update_mask"], stride=1)
    grf_ts = grf_ts[steps_offset:]  # now in kinematic-rate space: safe to index
    # Subtract per-leg adhesion forces
    adhesion_arr = np.array([adhesion_force_per_leg[leg] for leg in LEGS])
    grf_ts -= adhesion_arr[None, :]
    return grf_ts


def compute_contact_error(
    sim_results, gait_info, adhesion_force_per_leg, steps_offset, threshold, use_zonly: bool
):
    """1 - IOU between GRF-derived contact mask and kinematic stance mask.

    swing_mask=True means the leg is in swing (claw moving fast on xy-plane),
    so kinematic stance = ~swing_mask.
    """
    grf_ts = get_adjusted_grf(sim_results, adhesion_force_per_leg, steps_offset, use_zonly)
    sim_contact_mask = grf_ts >= threshold                      # (T_grf, n_legs)
    kinematic_stance_mask = ~gait_info["swing_mask"]            # (T_kin, n_legs)
    min_len = min(sim_contact_mask.shape[0], kinematic_stance_mask.shape[0])
    iou = intersection_over_union(
        sim_contact_mask[:min_len], kinematic_stance_mask[:min_len]
    )
    return 1.0 - iou


def save_sim_data(output_path, kinematic_snippet, sim_results, replay_manager,
                  replay_instance, adhforce_by_leg, **extra_params):
    """Pickle sim data in the format expected by sppaper's visualize functions.

    Note: visualize.plot_time_series reads 'leg_adhesion_force' from the pkl,
    so it must always be present.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "sim_data.pkl", "wb") as f:
        pickle.dump(
            {
                "snippet": kinematic_snippet,
                "sim_results": sim_results,
                "replay_manager": replay_manager,
                "adhforce_by_leg": adhforce_by_leg,
                "sim_timestep": replay_instance.sim.mj_model.opt.timestep,
                **extra_params,
            },
            f,
        )


# ── Debug plots ───────────────────────────────────────────────────────────────

def make_debug_plots(
    output_path,
    sim_results,
    gait_info,
    adhesion_force_per_leg,
    contact_force_disp_threshold,
    t_range,
    steps_offset,
    use_zonly: bool,
):
    """Save three debug outputs: replay video, trajectory comparison, GRF vs gait."""

    # 1. Kinematic replay video (requires sim_data.pkl and rendered frames in output_path)
    print("  Rendering replay video...")
    make_replay_video(
        sim_dir=output_path,
        output_path=output_path / "replay_debug.mp4",
        t_range=t_range,
        final_output_playback_speed=0.2,
        coarse_frames_interval=10,
    )

    # 2. Trajectory comparison (recorded vs simulated base trajectory)
    print("  Generating trajectory comparison figure...")
    fig, _ = plot_trajectory(sim_dir=output_path, t_range=t_range)
    fig.savefig(output_path / "trajectory_debug.pdf", dpi=300)
    plt.close(fig)

    # 3. Per-leg GRF time series vs kinematic gait diagram
    print("  Generating GRF vs gait figure...")
    grf_ts = get_adjusted_grf(sim_results, adhesion_force_per_leg, steps_offset, use_zonly)
    sim_contact_mask = grf_ts >= contact_force_disp_threshold   # True = sim in contact
    kinematic_stance_mask = ~gait_info["swing_mask"]            # True = kinematic stance
    min_len = min(grf_ts.shape[0], kinematic_stance_mask.shape[0])
    t_ax = np.arange(min_len) / DATA_FPS

    fig, axes = plt.subplots(len(LEGS), 1, figsize=(10, 9), sharex=True)
    for i, (ax, leg) in enumerate(zip(axes, LEGS)):
        # GRF trace
        ax.plot(
            t_ax, grf_ts[:min_len, i],
            color="steelblue", lw=0.9, label="GRF − adhesion"
        )
        ax.axhline(
            contact_force_disp_threshold,
            color="steelblue", ls="--", lw=0.7, alpha=0.6, label="threshold"
        )
        # Sim contact (from GRF)
        ax.fill_between(
            t_ax, 0, 1,
            where=sim_contact_mask[:min_len, i],
            transform=ax.get_xaxis_transform(),
            alpha=0.25, color="steelblue", label="sim contact"
        )
        # Kinematic stance (from claw speed)
        ax.fill_between(
            t_ax, 0, 1,
            where=kinematic_stance_mask[:min_len, i],
            transform=ax.get_xaxis_transform(),
            alpha=0.25, color="tomato", label="kin. stance"
        )
        ax.set_ylabel(leg.upper(), rotation=0, labelpad=30, va="center")
        if i == 0:
            ax.legend(fontsize=7, ncol=4, loc="upper right")

    axes[-1].set_xlabel("Time within t_range (s)")
    fig.suptitle(
        f"GRF vs kinematic stance — debug test run\n"
        f"threshold={contact_force_disp_threshold}"
    )
    fig.tight_layout()
    fig.savefig(output_path / "grf_vs_gait_debug.pdf")
    plt.close(fig)

    print(f"  All debug outputs saved to {output_path}")


# ── Optimization ──────────────────────────────────────────────────────────────

def _optimization_worker(
    worker_id,
    study_name,
    storage_url,
    kinematic_snippet,
    replay_manager,
    t_range,
    output_basedir,
    turnrate_weight,
    contact_force_disp_threshold,
    use_zonly,
    n_trials_per_worker,
    sampler_name,
):
    """Worker function run in a separate process.

    Each process independently loads the shared Optuna study from SQLite and
    runs its own trials. MuJoCo's EGL context is per-process so there are no
    GPU context conflicts.
    """
    steps_offset = get_steps_offset(kinematic_snippet, t_range)

    def objective(trial):
        actuator_gain = trial.suggest_float("actuator_gain", 50, 300)
        joint_damping = trial.suggest_float("joint_damping", 0.2, 3.0)
        sliding_friction = trial.suggest_float("sliding_friction", 0.5, 5.0)
        adhforce_fmlegs = trial.suggest_float("adhforce_fmlegs", 0.3, 3.0)
        adhforce_hlegs_scale = trial.suggest_float("adhforce_hlegs_scale", 0.5, 2.0)

        adhesion_force_per_leg = {
            "lf": adhforce_fmlegs,
            "lm": adhforce_fmlegs,
            "lh": adhforce_fmlegs * adhforce_hlegs_scale,
            "rf": adhforce_fmlegs,
            "rm": adhforce_fmlegs,
            "rh": adhforce_fmlegs * adhforce_hlegs_scale,
        }

        replay_instance = replay_manager.create_sim(
            actuator_gain=actuator_gain,
            joint_damping=joint_damping,
            sliding_friction=sliding_friction,
            adhforce_by_leg=adhesion_force_per_leg,
        )
        sim_results = replay_instance.replay_invkin_snippet(kinematic_snippet, disable_progress=True)
        replay_instance.sim.renderer.close()

        # Namespace trial dirs by study name to avoid collisions between runs
        trial_dir = output_basedir / study_name / f"trial{trial.number:03d}"
        save_sim_data(
            trial_dir,
            kinematic_snippet,
            sim_results,
            replay_manager,
            replay_instance,
            adhforce_by_leg=adhesion_force_per_leg,
            actuator_gain=actuator_gain,
            joint_damping=joint_damping,
            sliding_friction=sliding_friction,
            adhforce_fmlegs=adhforce_fmlegs,
            adhforce_hlegs_scale=adhforce_hlegs_scale,
        )

        trajs_info = align_smooth_decompose_trajs(kinematic_snippet, sim_results, t_range)
        obj1 = compute_traj_error(trajs_info, turnrate_weight)

        gait_info = get_gait_info(trial_dir, t_range)
        obj2 = compute_contact_error(
            sim_results,
            gait_info,
            adhesion_force_per_leg,
            steps_offset,
            contact_force_disp_threshold,
            use_zonly,
        )

        return obj1, obj2

    if sampler_name == "TPE":
        sampler = optuna.samplers.TPESampler(seed=worker_id)
    elif sampler_name == "NSGA-II":
        sampler = optuna.samplers.NSGAIISampler(seed=worker_id)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name!r}. Choose 'TPE' or 'NSGA-II'.")

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials_per_worker, show_progress_bar=False)


def run_optimization(
    replay_manager,
    kinematic_snippet,
    t_range,
    output_basedir,
    study_name,
    storage_url,
    turnrate_weight,
    contact_force_disp_threshold,
    use_zonly,
    sampler,
    rewrite,
    n_trials,
    n_workers,
):
    """Create an Optuna study and run multi-objective optimization across parallel processes.

    The study is persisted in a SQLite database so all worker processes share state.
    Workers are spawned (not forked) to avoid EGL context inheritance issues.
    Trials are distributed evenly; any remainder goes to the last worker.

    Args:
        sampler: "TPE" or "NSGA-II".
        rewrite: If True, delete the study from the DB before creating it fresh.
            If False, resume an existing study or create a new one.
    """
    if rewrite:
        try:
            optuna.delete_study(study_name=study_name, storage=storage_url)
        except KeyError:
            pass  # didn't exist yet, that's fine

    if sampler == "TPE":
        initial_sampler = optuna.samplers.TPESampler(seed=42)
    elif sampler == "NSGA-II":
        initial_sampler = optuna.samplers.NSGAIISampler(seed=42)
    else:
        raise ValueError(f"Unknown sampler: {sampler!r}. Choose 'TPE' or 'NSGA-II'.")

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=initial_sampler,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    study.enqueue_trial(INITIAL_TRIAL_PARAMS, skip_if_exists=True)

    n_trials_per_worker = n_trials // n_workers
    remainder = n_trials % n_workers
    worker_trial_counts = [
        n_trials_per_worker + (1 if i < remainder else 0) for i in range(n_workers)
    ]

    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(
            target=_optimization_worker,
            args=(
                i,
                study_name,
                storage_url,
                kinematic_snippet,
                replay_manager,
                t_range,
                output_basedir,
                turnrate_weight,
                contact_force_disp_threshold,
                use_zonly,
                worker_trial_counts[i],
                sampler,
            ),
        )
        for i in range(n_workers)
    ]

    print(f"Launching {n_workers} worker processes ({n_trials} total trials, sampler={sampler})...")
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    print(
        f"\nOptimization complete ({sampler}). "
        f"Pareto front contains {len(study.best_trials)} trials."
    )
    return study


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset and select snippet ──
    print("Loading kinematics dataset...")
    invkin_dataset = KinematicsDataset(
        keypoints3d_output_dir=KPT_3D_OUTPUT_BASEDIR,
        min_xy_conf=MIN_XY_CONF,
        mask_denoise_kernel_size_sec=MASK_DENOISE_KERNEL_SIZE_SEC,
        min_duration_sec=MIN_DURATION_SEC,
        data_fps=DATA_FPS,
    )
    kinematic_snippet = invkin_dataset[WALKING_PERIOD_IDX]

    # ── Debug test run — kept for reference, uncomment to re-run ──────────────
    # debug_dir = OUTPUT_DIR / "debug_test_run"
    # test_replay_manager = NeuroMechFlyReplayManager(
    #     sample_invkin_snippet=kinematic_snippet,
    #     passive_tarsus_stiffness=PASSIVE_TARSUS_STIFFNESS,
    #     passive_tarsus_damping=PASSIVE_TARSUS_DAMPING,
    # )
    # test_replay_instance = test_replay_manager.create_sim(
    #     actuator_gain=TEST_ACTUATOR_GAIN,
    #     joint_damping=TEST_JOINT_DAMPING,
    #     sliding_friction=TEST_SLIDING_FRICTION,
    #     adhforce_by_leg=TEST_LEG_ADHESION_GAIN,
    # )
    # test_sim_results = test_replay_instance.replay_invkin_snippet(kinematic_snippet)
    # debug_dir.mkdir(parents=True, exist_ok=True)
    # test_replay_instance.sim.renderer.save_video(debug_dir)
    # test_replay_instance.sim.renderer.close()
    # test_adhesion_per_leg = {
    #     leg: TEST_LEG_ADHESION_GAIN[leg] * TEST_LEG_ADHESION_FORCE for leg in LEGS
    # }
    # save_sim_data(
    #     debug_dir, kinematic_snippet, test_sim_results, test_replay_manager,
    #     test_replay_instance, adhforce_by_leg=TEST_LEG_ADHESION_GAIN,
    #     actuator_gain=TEST_ACTUATOR_GAIN, joint_damping=TEST_JOINT_DAMPING,
    #     sliding_friction=TEST_SLIDING_FRICTION,
    # )
    # test_steps_offset = get_steps_offset(kinematic_snippet, WALKING_PERIOD_TIMERANGE)
    # test_gait_info = get_gait_info(debug_dir, t_range=WALKING_PERIOD_TIMERANGE)
    # make_debug_plots(
    #     output_path=debug_dir, sim_results=test_sim_results, gait_info=test_gait_info,
    #     adhesion_force_per_leg=test_adhesion_per_leg,
    #     contact_force_disp_threshold=CONTACT_FORCE_DISP_THRESHOLD,
    #     t_range=WALKING_PERIOD_TIMERANGE, steps_offset=test_steps_offset,
    #     use_zonly=GRF_USE_ZONLY,
    # )
    # test_trajs_info = align_smooth_decompose_trajs(
    #     kinematic_snippet, test_sim_results, WALKING_PERIOD_TIMERANGE
    # )
    # test_traj_err = compute_traj_error(test_trajs_info, TURNRATE_WEIGHT)
    # test_contact_err = compute_contact_error(
    #     test_sim_results, test_gait_info, test_adhesion_per_leg,
    #     test_steps_offset, CONTACT_FORCE_DISP_THRESHOLD, GRF_USE_ZONLY,
    # )
    # print(f"Test-run objective values:")
    # print(f"  Trajectory error : {test_traj_err:.4f}")
    # print(f"  Contact error    : {test_contact_err:.4f}")

    # ── Multi-objective optimization ──────────────────────────────────────────
    storage_url = f"sqlite:///{OUTPUT_DIR / 'optuna_study.db'}"

    print("\nInitializing replay manager for optimization...")
    optim_replay_manager = NeuroMechFlyReplayManager(
        sample_invkin_snippet=kinematic_snippet,
        passive_tarsus_stiffness=PASSIVE_TARSUS_STIFFNESS,
        passive_tarsus_damping=PASSIVE_TARSUS_DAMPING,
    )

    common_kwargs = dict(
        replay_manager=optim_replay_manager,
        kinematic_snippet=kinematic_snippet,
        t_range=WALKING_PERIOD_TIMERANGE,
        output_basedir=OUTPUT_DIR,
        storage_url=storage_url,
        turnrate_weight=TURNRATE_WEIGHT,
        contact_force_disp_threshold=CONTACT_FORCE_DISP_THRESHOLD,
        use_zonly=GRF_USE_ZONLY,
        n_trials=500,
        n_workers=N_WORKERS,
        rewrite=True,
    )

    print("\nStarting TPE optimization...")
    run_optimization(study_name="nmf_physics_optim_tpe", sampler="TPE", **common_kwargs)

    print("\nStarting NSGA-II optimization...")
    run_optimization(study_name="nmf_physics_optim_nsgaii", sampler="NSGA-II", **common_kwargs)

    # ── Print Pareto-optimal parameters for each study ────────────────────────
    for study_name in ("nmf_physics_optim_tpe", "nmf_physics_optim_nsgaii"):
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"\n{'='*60}")
        print(f"Study: {study_name}  ({len(study.best_trials)} Pareto-optimal trials)")
        print(f"{'='*60}")
        for trial in sorted(study.best_trials, key=lambda t: t.values[0]):
            print(
                f"  Trial {trial.number:03d} | "
                f"traj_err={trial.values[0]:.4f}  contact_err={trial.values[1]:.4f}"
            )
            for param, value in trial.params.items():
                print(f"    {param:<22} = {value:.4f}")
