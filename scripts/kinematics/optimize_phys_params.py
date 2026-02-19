import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import wc26.kinematics.shared_constants as const
import wc26.kinematics.nmf_replay as replay

# Load data for a walking period
walking_period_npz_file = Path(
    str(const.KINEMATIC_DATA_PATH_FMT).format(idx=const.WALKING_PERIOD_IDX)
)
joint_angles_dict, thorax_pos_rec, spotlight_trial_dir = replay.load_kinematics_data(
    walking_period_npz_file
)

# Set up dummy simulation to get dof order and timestep info, then reformat ref joint
# angles accordingly
sim, fly = replay.set_up_simulation(
    joint_stiffness=5.0,
    joint_damping=0.5,
    actuator_gain=150.0,
    actuator_dampratio=0.2,
    actuator_timeconst_nsteps=3,
    sliding_friction=3.5,
    torsional_friction=0.05,
)
dof_order_in_sim = [dof.name for dof in fly.get_actuated_jointdofs_order("position")]
sim_timestep = sim.mj_model.opt.timestep
target_angles_arr_sim, rec_match_mask = replay.make_target_joint_angles_array(
    joint_angles_dict,
    dof_order_in_sim,
    rec_fps=const.DATA_FPS,
    sim_timestep=sim_timestep,
)

# Run optimization
for decompose in [True, False]:
    for multiobjective in [True, False]:
        study_name = f"decomp{decompose}_multiobj{multiobjective}"
        out_dir = const.PHYS_PARAMS_TUNING_DIR / study_name
        out_dir.mkdir(exist_ok=True, parents=True)
        replay.optimize_sim_params(
            study_name,
            target_angles_arr_sim,
            thorax_pos_rec,
            rec_match_mask,
            params_config=const.OPTIMIZABLE_PHYS_PARAMS,
            out_dir=out_dir,
            n_trials=1000,
            n_jobs=16,
            multiobjective=multiobjective,
            decompose_traj_mismatch=decompose,
        )

print("All done.")
