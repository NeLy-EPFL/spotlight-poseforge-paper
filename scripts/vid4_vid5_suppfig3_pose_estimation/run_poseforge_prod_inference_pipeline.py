from importlib.resources import files

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
from scipy.signal import medfilt

from poseforge.production.spotlight.core import SpotlightRecordingProcessor
import poseforge.pose.keypoints3d.invkin as ik

from sppaper.common.resources import get_spotlight_trials_dir, get_outputs_dir


def process_spotlight_trial(
    spotlight_trial_dir,
    output_dir,
    model_config_path,
    output_fps,
    output_playspeed,
    segment_lengths_medfilter_window,
    frame_range=None,
):
    # Create Spotlight trial structure, with symlinks to original input data
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir_symlink = output_dir / "metadata"
    processed_dir_symlink = output_dir / "processed"
    input_metadata_dir = spotlight_trial_dir / "metadata"
    input_processed_dir = spotlight_trial_dir / "processed"
    assert input_metadata_dir.is_dir()
    assert input_processed_dir.is_dir()
    if not metadata_dir_symlink.exists():
        metadata_dir_symlink.symlink_to(input_metadata_dir.absolute())
    if not processed_dir_symlink.exists():
        processed_dir_symlink.symlink_to(input_processed_dir.absolute())

    # Run Spotlight production pipeline
    recording = SpotlightRecordingProcessor(
        output_dir, model_config_path, with_muscle=True
    )
    recording.detect_usable_frames(edge_tolerance_mm=5.0, loading_n_workers=8)
    recording.predict_keypoints3d(loading_n_workers=8)
    recording.solve_inverse_kinematics()
    recording.predict_body_segmentation(loading_n_workers=8)
    recording.visualize_keypoints3d(
        output_playspeed, output_fps, frame_range=frame_range
    )
    recording.bodyseg_predicted = True
    recording.visualize_bodyseg_predictions(
        output_playspeed, output_fps, frame_range=frame_range
    )

    # Extract leg segment lengths over time and plot them
    seglens_over_time = get_segment_lengths_over_time(
        recording.paths.keypoints3d_prediction, frame_range
    )
    fig, axes = plot_segment_lengths(
        seglens_over_time,
        data_fps=recording.experiment_param["behavior_fps"],
        medfilter_window=segment_lengths_medfilter_window,
    )
    fig.savefig(output_dir / "poseforge_output/leg_segment_lengths.pdf")


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
        rawpred_world_xyz = f["pred_world_xyz"][:]  # (n_frames, n_keypoints, 3)
        if frame_range is not None:
            rawpred_world_xyz = rawpred_world_xyz[frame_range[0] : frame_range[1], ...]
        keypoint_names = list(f.attrs["keypoint_names"])

    pose_data_dict = ik._world_xyz_to_seqikpy_format(rawpred_world_xyz, keypoint_names)
    seglens_over_time = ik.extract_leg_segment_lengths(pose_data_dict)

    return seglens_over_time


def plot_segment_lengths(seglens_over_time, data_fps=330, medfilter_window=5):
    from sppaper.common.plot import setup_matplotlib_params

    setup_matplotlib_params()

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
    SEGLEN_LIM = (0, 1.2)
    MM_TO_IN = 1 / 25.4
    COLORS_BY_SIDE = {"L": "#546a76", "R": "#a23e48"}

    n_frames = seglens_over_time["LF_Coxa"].shape[0]
    t_grid = np.arange(n_frames) / data_fps

    figsize = (180 * MM_TO_IN, 70 * MM_TO_IN)
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
                ax_ts.plot(t_grid, length_ts_filt, color=color, linewidth=0.5)

                # Add stat texts
                mean = np.mean(length_ts_filt)
                std = np.std(length_ts_filt)
                coef_var = std / mean
                side_name = SIDE_DISPNAMES[side].capitalize()
                ax_ts.text(
                    0.05 * t_grid[-1],
                    0.9 * SEGLEN_LIM[1] - 0.15 * SEGLEN_LIM[1] * i_side,
                    f"{side_name}: {mean:.2f}±{std:.1g} mm (CV={int(100*coef_var)}%)",
                    color=color,
                    fontsize=5,
                )

            sns.despine(ax=ax_ts)
            ax_ts.set_xlim(t_grid[0], t_grid[-1] + 1 / data_fps)
            ax_ts.set_ylim(SEGLEN_LIM)
            disp_name = f"{LEGPOS_DISPNAMES[pos]} legs, {SEGLINK_DISPNAMES[seglink]}"
            ax_ts.set_title(disp_name.capitalize())
            if i_pos == len(LEG_POSITIONS) - 1:
                ax_ts.set_xlabel("Time (s)")
            else:
                ax_ts.set_xticklabels([])
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
    model_config_path = files("poseforge.production.spotlight").joinpath("config.yaml")
    trial = "20250613-fly1b-003"
    # frame_range = (38 * 30, 82 * 30)  # 0:38 to 1:22 in PoseForge's 30 FPS visualization
    frame_range = (int(6 * 0.2 * 330), int(18 * 0.2 * 330))  # 0:06-0:18 at 0.2x speed
    input_spotlight_trial_dir = get_spotlight_trials_dir() / trial
    output_dir = get_outputs_dir() / "pose_estimation" / trial
    segment_lengths_medfilter_window = 5  # in frames; should be odd integer

    process_spotlight_trial(
        input_spotlight_trial_dir,
        output_dir,
        model_config_path,
        output_fps=30,
        output_playspeed=0.2,
        frame_range=frame_range,
        segment_lengths_medfilter_window=segment_lengths_medfilter_window,
    )
