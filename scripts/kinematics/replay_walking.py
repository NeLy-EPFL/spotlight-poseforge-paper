import pickle

from sppaper.kinematics.shared_constants import DATA_FPS
from sppaper.common.resources import get_poseforge_datadir, get_outputs_dir
from sppaper.kinematics.data import KinematicsDataset
from sppaper.kinematics.nmf_sim import NeuroMechFlyReplayManager

KPT_3D_OUTPUT_BASEDIR = (
    get_poseforge_datadir()
    / "pose_estimation/keypoints3d/trial_20251118a/production/epoch19_step9167/"
)
REPLAY_OUTPUT_BASEDIR = get_outputs_dir() / "neuromechfly_replay/"
MIN_XY_CONF = 0.58
MASK_DENOISE_KERNEL_SIZE_SEC = 0.1
MIN_DURATION_SEC = 1
MASK_DENOISE_KERNEL_SIZE_STEPS = int(MASK_DENOISE_KERNEL_SIZE_SEC * DATA_FPS)
MIN_DURATION_STEPS = int(MIN_DURATION_SEC * DATA_FPS)
ACTUATOR_GAIN = 150
JOINT_DAMPING = 0.5
SLIDING_FRICTION = 2.0

replay_output_dir = (
    REPLAY_OUTPUT_BASEDIR
    / f"kp{ACTUATOR_GAIN}_damp{JOINT_DAMPING}_slidfric{SLIDING_FRICTION}"
)
replay_output_dir.mkdir(parents=True, exist_ok=True)

invkin_dataset = KinematicsDataset(
    keypoints3d_output_dir=KPT_3D_OUTPUT_BASEDIR,
    min_xy_conf=MIN_XY_CONF,
    mask_denoise_kernel_size_sec=MASK_DENOISE_KERNEL_SIZE_SEC,
    min_duration_sec=MIN_DURATION_SEC,
    data_fps=DATA_FPS,
)

replay_manager = NeuroMechFlyReplayManager(sample_invkin_snippet=invkin_dataset[0])
replay_instance = replay_manager.create_sim(
    actuator_gain=ACTUATOR_GAIN,
    joint_damping=JOINT_DAMPING,
    sliding_friction=SLIDING_FRICTION,
)

for i, invkin_snippet in enumerate(invkin_dataset):
    # Use snippet 21 for testing and visualization for now
    if i != 21:
        continue
    
    print(f"Replaying snippet {i+1}/{len(invkin_dataset)}")
    sim_results = replay_instance.replay_invkin_snippet(invkin_snippet)
    my_outdir = replay_output_dir / f"snippet{i}/"
    my_outdir.mkdir(parents=True, exist_ok=True)
    with open(my_outdir / f"sim_data.pkl", "wb") as f:
        data = {
            "snippet": invkin_snippet,
            "sim_results": sim_results,
            "replay_manager": replay_manager,
            "actuator_gain": ACTUATOR_GAIN,
            "joint_damping": JOINT_DAMPING,
            "sliding_friction": SLIDING_FRICTION,
        }
        pickle.dump(data, f)
    replay_instance.sim.renderer.save_video(my_outdir)
