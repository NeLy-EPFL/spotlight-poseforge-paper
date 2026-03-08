import re

from sppaper.kinematics.shared_constants import DATA_FPS
from sppaper.common.resources import get_inputs_dir, get_outputs_dir
from sppaper.kinematics.data import KinematicsDataset

SPOTLIGHT_TRIAL_BASEDIR = get_inputs_dir() / "spotlight_trials"
REPLAY_OUTPUT_DIR = get_outputs_dir() / "neuromechfly_replay/sim_data/"
MIN_XY_CONF = 0.58
MASK_DENOISE_KERNEL_SIZE_SEC = 0.1
MIN_DURATION_SEC = 1

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
