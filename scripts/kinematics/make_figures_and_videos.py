import numpy as np
import polars as pl
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import cmasher
import pickle
from scipy.signal import savgol_filter, medfilt

from flygym.anatomy import LEGS

import sppaper.kinematics.shared_constants as const
import sppaper.kinematics.trajectory as traj
import sppaper.kinematics.visualize as viz
from sppaper.common.resources import get_outputs_dir

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

VISUALIZED_SIM_DIR = (
    get_outputs_dir() / "neuromechfly_replay/kp150_damp0.5_slidfric2.0/snippet21/"
)
VISUALIZED_SIM_TIMERANGE = (0.5, 2.5)
VIZ_OUTPUT_DIR = get_outputs_dir() / "neuromechfly_replay/visualization/"
VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(VISUALIZED_SIM_DIR / "sim_data.pkl", "rb") as f:
    data = pickle.load(f)
sim_results = data["sim_results"]
kinematic_snippet = data["snippet"]

# Generate time series figure
fig, axes = viz.plot_time_series(
    kinematic_snippet, sim_results, leg="lf", t_range=VISUALIZED_SIM_TIMERANGE
)
fig.savefig(VIZ_OUTPUT_DIR / "time_series_lf.pdf")

# Generate trajectory comparison figure
trajs_info = viz.align_smooth_decompose_trajectories(
    kinematic_snippet, sim_results, t_range=VISUALIZED_SIM_TIMERANGE
)
fig, axes = viz.plot_trajectory(
    kinematic_snippet, trajs_info, t_range=VISUALIZED_SIM_TIMERANGE
)
fig.savefig(VIZ_OUTPUT_DIR / "trajectory_lf.pdf", dpi=300)
