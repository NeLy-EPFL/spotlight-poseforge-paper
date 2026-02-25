from sppaper.common.resources import get_outputs_dir

INVKIN_OUTPUT_DIR = get_outputs_dir() / "kinematics/poseforge_output/"
DATA_FPS = 330
KIN_MEDFILTER_SIZE = 3
XYZ_RATELIM = 0.2  # 0.2 mm * 330 FPS = 66 mm/s
JOINT_ANGLE_RATELIM = 0.19  # 10 turns/s (in radians)
VIDEO_PLAYBACK_SPEED = 0.1
VIDEO_OUTPUT_FPS = 33
