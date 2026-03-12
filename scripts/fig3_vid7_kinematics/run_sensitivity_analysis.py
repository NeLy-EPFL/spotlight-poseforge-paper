"""Optimizes 5 parameters (actuator_gain, joint_damping, sliding_friction,
adhforce_fmlegs, adhforce_hlegs_scale) to minimize the weighted MSE between
recorded and simulated linear speed and turn rate.
"""

import multiprocessing as mp
import pickle

import numpy as np
import optuna

from sppaper.common.resources import get_poseforge_datadir, get_outputs_dir
from sppaper.kinematics.shared_constants import DATA_FPS
from sppaper.kinematics.nmf_sim import NeuroMechFlyReplayManager
from sppaper.kinematics.data import KinematicsDataset, align_smooth_decompose_trajs


def mse(a, b):
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))

def traj_loss(trajs_info, turnrate_weight):
    return mse(
        trajs_info["baselinspeed_rec"], trajs_info["baselinspeed_sim"]
    ) + turnrate_weight * mse(
        trajs_info["baseturnrate_rec"], trajs_info["baseturnrate_sim"]
    )
    
def _worker(
    worker_id,
    study_name,
    storage_url,
    kinematic_snippet,
    replay_manager,
    t_range,
    output_basedir,
    turnrate_weight,
    n_trials_per_worker,
):
    def objective(trial):
        actuator_gain = trial.suggest_float("actuator_gain", 50, 300)
        joint_damping = trial.suggest_float("joint_damping", 0.2, 3.0)
        sliding_friction = trial.suggest_float("sliding_friction", 0.5, 5.0)
        adhforce_fmlegs = trial.suggest_float("adhforce_fmlegs", 0.3, 3.0)
        adhforce_hlegs_scale = trial.suggest_float("adhforce_hlegs_scale", 0.5, 2.0)

        adhforce_by_leg = {
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
            adhforce_by_leg=adhforce_by_leg,
        )
        sim_results = replay_instance.replay_invkin_snippet(
            kinematic_snippet, disable_progress=True
        )
        replay_instance.sim.renderer.close()

        # Save minimal trial data for post-hoc inspection
        trial_dir = output_basedir / f"trial{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_dir / "sim_data.pkl", "wb") as f:
            pickle.dump(
                {
                    "snippet": kinematic_snippet,
                    "sim_results": sim_results,
                    "replay_manager": replay_manager,
                    "adhforce_by_leg": adhforce_by_leg,
                    "sim_timestep": replay_instance.sim.mj_model.opt.timestep,
                    "actuator_gain": actuator_gain,
                    "joint_damping": joint_damping,
                    "sliding_friction": sliding_friction,
                    "adhforce_fmlegs": adhforce_fmlegs,
                    "adhforce_hlegs_scale": adhforce_hlegs_scale,
                },
                f,
            )

        trajs_info = align_smooth_decompose_trajs(
            kinematic_snippet, sim_results, t_range
        )
        return traj_loss(trajs_info, turnrate_weight)

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url,
        sampler=optuna.samplers.TPESampler(seed=worker_id),
    )
    study.optimize(objective, n_trials=n_trials_per_worker, show_progress_bar=False)


if __name__ == "__main__":
    # Fixed parameters (not tuned)
    passive_tarsus_stiffness = 10
    passive_tarsus_damping = 0.5

    # Loss specs
    turnrate_weight = 3

    # Initial parameters
    initial_trial_params = {
        "actuator_gain": 150,
        "joint_damping": 0.5,
        "sliding_friction": 2.0,
        "adhforce_fmlegs": 1.0,
        "adhforce_hlegs_scale": 0.6,
    }
    
    # Specify input dataset
    kpt3d_output_dir = (
        get_poseforge_datadir()
        / "pose_estimation/keypoints3d/trial_20251118a/production/epoch19_step9167/"
    )
    min_xy_conf = 0.58
    mask_denoise_kernel_size_sec = 0.1
    min_duration_sec = 1
    walking_period_idx = 21
    walking_period_timerange = (0.5, 2.5)
    
    # Specify output location
    output_dir = get_outputs_dir() / "neuromechfly_replay/hparam_optim/"
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{output_dir / 'optuna_study.db'}"
    study_name = "trajmse_walkingsnippet21"
    
    # Compute parameters
    n_workers = 4
    n_trials = 100
    seed = 42
    
    print("Loading kinematics dataset...")
    invkin_dataset = KinematicsDataset(
        keypoints3d_output_dir=kpt3d_output_dir,
        min_xy_conf=min_xy_conf,
        mask_denoise_kernel_size_sec=mask_denoise_kernel_size_sec,
        min_duration_sec=min_duration_sec,
        data_fps=DATA_FPS,
    )
    kinematic_snippet = invkin_dataset[walking_period_idx]

    print("Initializing replay manager...")
    replay_manager = NeuroMechFlyReplayManager(
        sample_invkin_snippet=kinematic_snippet,
        passive_tarsus_stiffness=passive_tarsus_stiffness,
        passive_tarsus_damping=passive_tarsus_damping,
    )

    # Create study and seed with the known-good starting point
    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
    except KeyError:
        pass
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=study_name,
        storage=storage_url,
    )
    study.enqueue_trial(initial_trial_params, skip_if_exists=True)

    # Distribute trials across workers
    n_per_worker = n_trials // n_workers
    remainder = n_trials % n_workers
    trial_counts = [n_per_worker] * n_workers
    for i in range(remainder):
        trial_counts[i] += 1

    print(f"Launching {n_workers} workers ({n_trials} trials total)...")
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(
            target=_worker,
            args=(
                i,
                study_name,
                storage_url,
                kinematic_snippet,
                replay_manager,
                walking_period_timerange,
                output_dir,
                turnrate_weight,
                trial_counts[i],
            ),
        )
        for i in range(n_workers)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # Report results
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    best = study.best_trial
    print(f"\nBest trial: #{best.number}  loss={best.value:.6f}")
    for k, v in best.params.items():
        print(f"  {k:<25} = {v:.4f}")
