import pickle

from sppaper.kinematics.data import get_gait_info
from sppaper.common.resources import get_outputs_dir
from sppaper.kinematics.visualize import (
    plot_time_series,
    plot_trajectory,
    make_replay_video,
    plot_invkin_frame,
    plot_claw_traj_by_swing_stance,
)

KIN_FILTER_WINDOW_SIZE = 3
DOF_DISPLAY_NAMES = {
    "ThC_pitch": "ThC-p",
    "ThC_roll": "ThC-r",
    "ThC_yaw": "ThC-y",
    "CTr_pitch": "CTr-p",
    "CTr_roll": "CTr-r",
    "FTi_pitch": "FTi-p",
    "TiTa_pitch": "TiTa-p",
}
LEG_DISP_NAMES = {
    "LF": "Left front leg",
    "LM": "Left middle leg",
    "LH": "Left hind leg",
    "RF": "Right front leg",
    "RM": "Right middle leg",
    "RH": "Right hind leg",
}
AXIS_DISPLAY_NAMES = {"x": "fore/aft", "y": "med/lat", "z": "height"}

VISUALIZED_SIM_DIR = get_outputs_dir() / "neuromechfly_replay/sim_data/nmf_hparamoptim_f1metric_nsga-ii_zonly_107/snippet21/"
VISUALIZED_SIM_TIMERANGE = (0.5, 2.5)
FWDKIN_SNAPSHOT_FULLREC_FRAMEID = 1414
VIZ_OUTPUT_DIR = get_outputs_dir() / "neuromechfly_replay/nmf_hparamoptim_f1metric_nsga-ii_zonly_107/visualization/"
VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load sim results
print("Loading simulation results...")
with open(VISUALIZED_SIM_DIR / "sim_data.pkl", "rb") as f:
    data = pickle.load(f)
sim_results = data["sim_results"]
kinematic_snippet = data["snippet"]

# Plot gait diagram and claw trajectory colored by swing/stance
gait_info = get_gait_info(VISUALIZED_SIM_DIR, t_range=VISUALIZED_SIM_TIMERANGE)
fig, ax = plot_claw_traj_by_swing_stance(
    VISUALIZED_SIM_DIR, gait_info, t_range=VISUALIZED_SIM_TIMERANGE
)
fig.savefig(VIZ_OUTPUT_DIR / "claw_traj_by_swing_stance.pdf")

# Generate time series figure
print("Generating time series figure...")
fig, axes = plot_time_series(
    sim_dir=VISUALIZED_SIM_DIR,
    leg="lf",
    gait_info=gait_info,
    t_range=VISUALIZED_SIM_TIMERANGE,
)
fig.savefig(VIZ_OUTPUT_DIR / "time_series_lf.pdf")

# Generate trajectory comparison figure
print("Generating trajectory comparison figure...")
fig, axes = plot_trajectory(
    sim_dir=VISUALIZED_SIM_DIR, t_range=VISUALIZED_SIM_TIMERANGE
)
fig.savefig(VIZ_OUTPUT_DIR / "trajectory_lf.pdf", dpi=300)

# Generate a single snapshot of forward kinematics visualization for figures
print("Generating forward kinematics snapshot figure...")
fig, ax = plot_invkin_frame(VISUALIZED_SIM_DIR, FWDKIN_SNAPSHOT_FULLREC_FRAMEID)
fig.savefig(VIZ_OUTPUT_DIR / "forward_kinematics_snapshot.pdf")

# Generate kinematic replay side-by-side video
print("Generating replay video...")
make_replay_video(
    sim_dir=VISUALIZED_SIM_DIR,
    output_path=VIZ_OUTPUT_DIR / "nmf_replay_summary.mp4",
    t_range=VISUALIZED_SIM_TIMERANGE,
    final_output_playback_speed=0.2,
    coarse_frames_interval=10,
)
