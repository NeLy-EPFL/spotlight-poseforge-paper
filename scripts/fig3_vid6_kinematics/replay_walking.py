import matplotlib

matplotlib.use("Agg")

import re
import pickle

from sppaper.kinematics.shared_constants import DATA_FPS
from sppaper.common.resources import get_inputs_dir, get_outputs_dir
from sppaper.kinematics.data import KinematicsDataset
from sppaper.kinematics.nmf_sim import NeuroMechFlyReplayManager

SPOTLIGHT_TRIAL_BASEDIR = get_inputs_dir() / "spotlight_trials"
REPLAY_OUTPUT_DIR = get_outputs_dir() / "neuromechfly_replay/sim_data/"
MIN_XY_CONF = 0.58
MASK_DENOISE_KERNEL_SIZE_SEC = 0.1
MIN_DURATION_SEC = 1
MASK_DENOISE_KERNEL_SIZE_STEPS = int(MASK_DENOISE_KERNEL_SIZE_SEC * DATA_FPS)
MIN_DURATION_STEPS = int(MIN_DURATION_SEC * DATA_FPS)
PASSIVE_TARSUS_STIFFNESS = 10
PASSIVE_TARSUS_DAMPING = 0.5
LEG_ADHESION_GAIN = {"lf": 1.0, "lm": 1.0, "lh": 0.6, "rf": 1.0, "rm": 1.0, "rh": 0.6}
ACTUATOR_GAIN = 150
JOINT_DAMPING = 0.5
SLIDING_FRICTION = 2.0
LEG_ADHESION_FORCE = 1.0

REPLAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

spotlight_trial_dirs = [
    dir_
    for dir_ in sorted(SPOTLIGHT_TRIAL_BASEDIR.iterdir())
    if re.match(r"^20250613-fly1b-\d{3}$", dir_.name)
]
invkin_dataset = KinematicsDataset(
    spotlight_trial_dirs=spotlight_trial_dirs,
    min_xy_conf=MIN_XY_CONF,
    mask_denoise_kernel_size_sec=MASK_DENOISE_KERNEL_SIZE_SEC,
    min_duration_sec=MIN_DURATION_SEC,
    data_fps=DATA_FPS,
)
invkin_dataset.summary_df.write_csv(REPLAY_OUTPUT_DIR / "walking_snippets.csv")

replay_manager = NeuroMechFlyReplayManager(
    sample_invkin_snippet=invkin_dataset[0],
    passive_tarsus_stiffness=PASSIVE_TARSUS_STIFFNESS,
    passive_tarsus_damping=PASSIVE_TARSUS_DAMPING,
    leg_adhesion_gain=LEG_ADHESION_GAIN,
)
replay_instance = replay_manager.create_sim(
    actuator_gain=ACTUATOR_GAIN,
    joint_damping=JOINT_DAMPING,
    sliding_friction=SLIDING_FRICTION,
    leg_adhesion_force=LEG_ADHESION_FORCE,
)

for i, invkin_snippet in enumerate(invkin_dataset):
    # Use snippet 21 for testing and visualization for now
    if i != 50:
        continue

    print(f"Replaying snippet {i+1}/{len(invkin_dataset)}")
    sim_results = replay_instance.replay_invkin_snippet(invkin_snippet)
    my_outdir = REPLAY_OUTPUT_DIR / f"snippet{i}/"
    my_outdir.mkdir(parents=True, exist_ok=True)
    with open(my_outdir / f"sim_data.pkl", "wb") as f:
        data = {
            "snippet": invkin_snippet,
            "sim_results": sim_results,
            "replay_manager": replay_manager,
            "actuator_gain": ACTUATOR_GAIN,
            "joint_damping": JOINT_DAMPING,
            "sliding_friction": SLIDING_FRICTION,
            "leg_adhesion_force": LEG_ADHESION_FORCE,
            "sim_timestep": replay_instance.sim.mj_model.opt.timestep,
        }
        pickle.dump(data, f)
    replay_instance.sim.renderer.save_video(my_outdir)

replay_instance.sim.renderer.close()
