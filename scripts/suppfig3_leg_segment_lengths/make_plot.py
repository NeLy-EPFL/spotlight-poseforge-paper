from sppaper.common.plot import setup_matplotlib_params

setup_matplotlib_params()

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
from scipy.signal import medfilt

import poseforge.pose.keypoints3d.invkin as ik

from sppaper.common.resources import get_outputs_dir, get_poseforge_datadir

LEG_POSITIONS = ["F", "M", "H"]
SIDES = ["L", "R"]
LEG_SEGMENTS = ["Coxa", "Femur", "Tibia", "Tarsus"]
SEGLINK_DISPNAMES = {
    "Coxa": "coxa",
    "Femur": "trochanter-femur",
    "Tibia": "tibia",
    "Tarsus": "tarsus",
}
SIDE_DISPNAMES = {"L": "left", "R": "right"}
LEGPOS_DISPNAMES = {"F": "front", "M": "middle", "H": "hind"}
SEGLEN_LIM = (0, 1.5)
KDE_XLIM = (0, 20)
MM_TO_IN = 1 / 25.4
COLORS_BY_SIDE = {"L": "#546a76", "R": "#a23e48"}


def get_segment_lengths_over_time(poseforg_pred_path, frame_range=None):
    """Returns a dict mapping segment name to an array of segment lengths over time.

    Args:
        poseforg_pred_path: Path to the PoseForge keypoints3d.h5 output file generated
            by the production pipeline.

    Returns:
        A dict mapping segment name to an array of segment lengths over time.
        Dict keys are: {leg}_{segment}, where leg = {L,R}{F,M,H} and segment is among
            {Coxa,Femur,Tibia,Tarsus}.
    """
    with h5py.File(poseforg_pred_path, "r") as f:
        ds = f["keypoints_world_xyz"]
        rawpred_world_xyz = ds[:]  # (n_frames, n_keypoints, 3)
        if frame_range is not None:
            rawpred_world_xyz = rawpred_world_xyz[frame_range[0] : frame_range[1], ...]
        keypoint_names = list(ds.attrs["keypoints"])

    pose_data_dict = ik._world_xyz_to_seqikpy_format(rawpred_world_xyz, keypoint_names)
    seglens_over_time = ik.extract_leg_segment_lengths(pose_data_dict)

    return seglens_over_time


def plot(seglens_over_time, data_fps=330, medfilter_window=5):
    n_frames = seglens_over_time["LF_Coxa"].shape[0]
    t_grid = np.arange(n_frames) / data_fps

    figsize = (170 * MM_TO_IN, 70 * MM_TO_IN)
    fig, axes = plt.subplots(
        3,
        8,
        figsize=figsize,
        width_ratios=[5, 1, 5, 1, 5, 1, 5, 1],
        tight_layout=True,
    )
    for ax in axes.flat:
        ax.set_facecolor("none")

    for i_pos, pos in enumerate(LEG_POSITIONS):
        for i_seglink, seglink in enumerate(LEG_SEGMENTS):
            ax_ts = axes[i_pos, i_seglink * 2]
            for i_side, side in enumerate(SIDES):
                leg_name = f"{side}{pos}"
                segment_name = f"{leg_name}_{seglink}"
                length_ts = seglens_over_time[segment_name]
                length_ts_filt = medfilt(length_ts, medfilter_window)
                color = COLORS_BY_SIDE[side]
                side_dispname = SIDE_DISPNAMES[side]
                ax_ts.plot(
                    t_grid,
                    length_ts_filt,
                    label=side_dispname,
                    color=color,
                    linewidth=0.5,
                )

                # Add stat texts
                mean = np.mean(length_ts_filt)
                std = np.std(length_ts_filt)
                coef_var = std / mean
                ax_ts.text(
                    0.05 * t_grid[-1],
                    0.9 * SEGLEN_LIM[1] - 0.15 * SEGLEN_LIM[1] * i_side,
                    f"{mean:.2f}±{std:.1g} mm (CV={int(100*coef_var)}%)",
                    color=color,
                    fontsize=5,
                )

            sns.despine(ax=ax_ts)
            ax_ts.set_xlim(t_grid[0], t_grid[-1] + 1 / data_fps)
            ax_ts.set_ylim(SEGLEN_LIM)
            disp_name = f"{LEGPOS_DISPNAMES[pos]} legs, {SEGLINK_DISPNAMES[seglink]}"
            ax_ts.set_title(disp_name[0].upper() + disp_name[1:])
            if i_pos == len(LEG_POSITIONS) - 1:
                ax_ts.set_xlabel("Time (s)")
            else:
                ax_ts.set_xticklabels([])
            if i_pos == 0 and i_seglink == 0:
                ax_ts.legend(frameon=False)
            if i_seglink == 0:
                ax_ts.set_ylabel(f"Length (mm)")
            else:
                ax_ts.set_yticklabels([])

            ax_kde = axes[i_pos, i_seglink * 2 + 1]
            sns.despine(ax=ax_kde, bottom=True)
            for side in SIDES:
                leg_name = f"{side}{pos}"
                segment_name = f"{leg_name}_{seglink}"
                length_ts = seglens_over_time[segment_name]
                length_ts_filt = medfilt(length_ts, medfilter_window)
                color = COLORS_BY_SIDE[side]
                sns.kdeplot(
                    y=length_ts_filt,
                    ax=ax_kde,
                    color=color,
                    bw_adjust=0.2,
                    linewidth=0.5,
                )
                ax_kde.set_ylim(SEGLEN_LIM)
                ax_kde.set_xticks([])
                ax_kde.set_yticks([])
                ax_kde.set_xlabel("")

    return fig, axes


if __name__ == "__main__":
    keypoints3d_output_basedir = (
        get_poseforge_datadir()
        / "pose_estimation/keypoints3d/trial_20251118a/production/epoch19_step9167/"
    )
    output_path = get_outputs_dir() / "leg_segment_lengths/leg_segment_lengths.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trial = "20250613-fly1b-003"
    frame_range = (38 * 30, 82 * 30)  # 0:38 to 1:22 in PoseForge's 30 FPS visualization
    data_fps = 330
    medfilter_window = 5

    poseforge_output_path = keypoints3d_output_basedir / trial / "keypoints3d.h5"
    seglens_over_time = get_segment_lengths_over_time(
        poseforge_output_path, frame_range
    )
    fig, axes = plot(seglens_over_time, data_fps, medfilter_window)
    fig.savefig(output_path)
