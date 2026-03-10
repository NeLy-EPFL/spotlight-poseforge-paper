import pickle

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


def traj_error(trajs_info, turnrate_weight):
    linspeed_error = mse_error(
        trajs_info["baselinspeed_rec"], trajs_info["baselinspeed_sim"]
    )
    turnrate_error = mse_error(
        trajs_info["baseturnrate_rec"], trajs_info["baseturnrate_sim"]
    )
    return linspeed_error + turnrate_weight * turnrate_error


def ground_contact_error(grf_ts, gait_info, contact_force_disp_threshold):
    nmf_contact_mask = (
        sim_results["ground_contacts"]["contact_mask"] >= contact_force_disp_threshold
    )
    kinematic_stance_mask = gait_info["swing_mask"]
    iou = intersection_over_union(nmf_contact_mask, kinematic_stance_mask)
    return 1 - iou


def run_optimization(
    replay_manager,
    kinematic_snippet,
    t_range,
    output_basedir,
    turnrate_weight,
    contact_force_disp_threshold,
):

    def objective(trial):
        # Define the hyperparameters to optimize
        actuator_gain = trial.suggest_float("actuator_gain", 50, 300)
        joint_damping = trial.suggest_float("joint_damping", 0.2, 3)
        sliding_friction = trial.suggest_float("sliding_friction", 0.5, 5)
        adhforce_fmlegs = trial.suggest_float("adhforce_fmlegs", 0.3, 3)
        adhforce_hlegs_scale = trial.suggest_float("adhforce_hlegs_scale", 0.5, 2)

        # Run simulation and save sim data
        adhesion_forces = {
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
            leg_adhesion_force=adhesion_forces,
        )
        sim_results = replay_instance.replay_invkin_snippet(kinematic_snippet)
        trial_name = f"trial{trial.number:03d}"
        output_path = output_basedir / f"{trial_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / f"sim_data.pkl", "wb") as f:
            data = {
                "snippet": kinematic_snippet,
                "sim_results": sim_results,
                "replay_manager": replay_manager,
                "actuator_gain": actuator_gain,
                "joint_damping": joint_damping,
                "sliding_friction": sliding_friction,
                "adhforce_fmlegs": adhforce_fmlegs,
                "adhforce_hlegs_scale": adhforce_hlegs_scale,
                "sim_timestep": replay_instance.sim.mj_model.opt.timestep,
            }
            pickle.dump(data, f)

        # Compute optimization metrics
        trajs_info = align_smooth_decompose_trajs(
            kinematic_snippet, sim_results, t_range
        )
        traj_error = traj_error(trajs_info, turnrate_weight)
        gait_info = get_gait_info(output_path, t_range)
        if t_range is not None:
            start_idx_before = kinematic_snippet.start_idx
            kinematic_snippet = kinematic_snippet.get_subselection(*t_range)
            steps_offset = kinematic_snippet.start_idx - start_idx_before
        else:
            steps_offset = 0
        grf_ts = np.linalg.norm(
            sim_results["ground_contacts"]["forces_world"][steps_offset:, :, ...],
            axis=-1,
        )
        grf_ts[np.isnan(grf_ts)] = 0
        leg_adhesion_forces = (
            np.array([replay_manager.leg_adhesion_gain[leg] for leg in LEGS])
            * leg_adhesion_force
        )
        grf_ts -= leg_adhesion_forces[None, :]
        contact_error = ground_contact_error(
            grf_ts, gait_info, contact_force_disp_threshold
        )

        return traj_error, contact_error

    # Run multi-objective optimization with Optuna and save output
    ...


if __name__ == "__main__":
    KPT_3D_OUTPUT_BASEDIR = (
        get_poseforge_datadir()
        / "pose_estimation/keypoints3d/trial_20251118a/production/epoch19_step9167/"
    )
    OUTPUT_DIR = get_outputs_dir() / "neuromechfly_replay/optim/"
    MIN_XY_CONF = 0.58
    MASK_DENOISE_KERNEL_SIZE_SEC = 0.1
    MIN_DURATION_SEC = 1
    MASK_DENOISE_KERNEL_SIZE_STEPS = int(MASK_DENOISE_KERNEL_SIZE_SEC * DATA_FPS)
    WALKING_PERIOD_IDX = 21
    WALKING_PERIOD_TIMERANGE = (0.5, 2.5)
    PASSIVE_TARSUS_STIFFNESS = 10
    PASSIVE_TARSUS_DAMPING = 0.5

    invkin_dataset = KinematicsDataset(
        keypoints3d_output_dir=KPT_3D_OUTPUT_BASEDIR,
        min_xy_conf=MIN_XY_CONF,
        mask_denoise_kernel_size_sec=MASK_DENOISE_KERNEL_SIZE_SEC,
        min_duration_sec=MIN_DURATION_SEC,
        data_fps=DATA_FPS,
    )
    kinematic_snippet = invkin_dataset[WALKING_PERIOD_IDX]

    replay_manager = NeuroMechFlyReplayManager(
        sample_invkin_snippet=kinematic_snippet,
        passive_tarsus_stiffness=PASSIVE_TARSUS_STIFFNESS,
        passive_tarsus_damping=PASSIVE_TARSUS_DAMPING,
    )

    run_optimization(
        replay_manager,
        kinematic_snippet,
        OUTPUT_DIR,
        turnrate_weight=3,
        contact_force_disp_threshold=0.2,
    )
