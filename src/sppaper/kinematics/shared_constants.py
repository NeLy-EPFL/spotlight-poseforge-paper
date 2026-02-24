from sppaper.common.resources import get_outputs_dir

INVKIN_OUTPUT_DIR = get_outputs_dir() / "kinematics/poseforge_output/"
DATA_FPS = 330
KIN_MEDFILTER_SIZE = 3
XYZ_RATELIM = 0.2  # 0.2 mm * 330 FPS = 66 mm/s
JOINT_ANGLE_RATELIM = 0.19  # in rad (~= 10.09 deg), 10.9 deg * 330 FPS = 10 turns/s

# # IO
# KPT_3D_OUTPUT_BASEDIR = (
#     io.get_poseforge_datadir()
#     / "pose_estimation/keypoints3d/trial_20251118a/production/epoch19_step9167/"
# )
# OUT_DATA_DIR = io.get_outputs_dir() / "neuromechfly_replay/data/"
# OUT_FIGS_DIR = io.get_outputs_dir() / "neuromechfly_replay/figures/"
# WALKING_PERIODS_DATAFRAME_PATH = OUT_DATA_DIR / "walking_periods_summary.csv"
# KINEMATIC_DATA_PATH_FMT = OUT_DATA_DIR / "walking_period_{idx:03d}.npz"
# PHYS_PARAMS_TUNING_DIR = OUT_DATA_DIR / "nmf_phys_params_tuning/"

# # Recording and data parameters
# DATA_FPS = 330
# AXIS_ORDER = ("yaw", "pitch", "roll")  # as configured for SeqIKPy

# # Filtering parameters for extracting walking periods
# MIN_XY_CONF = 0.58
# MASK_DENOISE_KERNEL_SIZE_SEC = 0.1
# MIN_DURATION_SEC = 1
# KINEMATICS_MEDFILTER_KERNEL_SIZE = 5
# MASK_DENOISE_KERNEL_SIZE_STEPS = int(MASK_DENOISE_KERNEL_SIZE_SEC * DATA_FPS)
# MIN_DURATION_STEPS = int(MIN_DURATION_SEC * DATA_FPS)

# # NeuroMechFly parameters
# ARTICULATED_JOINTS = "legs_only"
# ACTUATED_DOFS = "legs_active_only"
# NEUTRAL_POSE_FILE = io.get_flygym_assetdir() / "model/pose/neutral.yaml"
# SPAWN_HEIGHT = 0.7  # mm
# WARMUP_PERIOD_SEC = 0.05
# OPTIMIZABLE_PHYS_PARAMS = {
#     "joint_stiffness": {"lim": (1, 20), "init": 2.7},
#     "joint_damping": {"lim": (0.1, 1), "init": 0.3},
#     "actuator_gain": {"lim": (30, 300), "init": 120.0},
#     "actuator_dampratio": {"lim": (0, 1.5), "init": 0.5},
#     "actuator_timeconst_nsteps": {"lim": (0, 33), "init": 10},
#     "sliding_friction": {"lim": (0.5, 5), "init": 3.5},
#     "torsional_friction": {"lim": (0, 0.1), "init": 0.015},
# }
# PASSIVE_TARSAL_STIFFNESS = 10.0
# PASSIVE_TARSAL_DAMPING = 0.5

# # Display & visualization
# DOF_DISPLAY_NAMES = {
#     "ThC_pitch": "ThC-p",
#     "ThC_roll": "ThC-r",
#     "ThC_yaw": "ThC-y",
#     "CTr_pitch": "CTr-p",
#     "CTr_roll": "CTr-r",
#     "FTi_pitch": "FTi-p",
#     "TiTa_pitch": "TiTa-p",
# }
# AXIS_DISPLAY_NAMES = {
#     "x": "fore/aft",
#     "y": "med/lat",
#     "z": "height",
# }
# VIDEO_PLAYBACK_SPEED = 0.25
# VIDEO_OUTPUT_FPS = 25

# # Example snippet
# WALKING_PERIOD_IDX = 21  # longest in the dataset
# START_END_SEC = (1.2, 3.2)  # typical walking period for demostration (in seconds)
