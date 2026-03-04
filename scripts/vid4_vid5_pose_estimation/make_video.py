from pathlib import Path

import imageio.v3 as iio
import numpy as np

from sppaper.common.resources import get_poseforge_datadir, get_outputs_dir


def crop_and_trim_video(
    input_path,
    output_path,
    row_range=None,
    col_range=None,
    frame_range=None,
    output_fps=None,
    output_codec="libx264",
    output_quality=8,
):
    """
    Crop and trim a video spatially and temporally.

    Args:
        input_path: Path to the input video.
        output_path: Path to save the output video.
        row_range: (start, end) row indices to crop; None keeps all rows.
        col_range: (start, end) col indices to crop; None keeps all columns.
        frame_range: (start, end) frame indices to trim; None keeps all frames.
        output_fps: Output framerate; None preserves input FPS.
        output_codec: FFmpeg codec for output (default: "libx264").
        output_quality: FFmpeg quality/CRF value (default: 8).
    """
    props = iio.improps(input_path, plugin="pyav")
    meta = iio.immeta(input_path, plugin="pyav")

    fps = output_fps or meta.get("fps", 30)

    r0, r1 = row_range or (0, props.shape[1])
    c0, c1 = col_range or (0, props.shape[2])
    f0, f1 = frame_range or (0, props.shape[0])

    frames = iio.imread(input_path, plugin="pyav")
    cropped = frames[f0:f1, r0:r1, c0:c1]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        output_path,
        cropped,
        fps=fps,
        codec=output_codec,
        quality=output_quality,
    )


if __name__ == "__main__":
    keypoints3d_output_basedir = (
        get_poseforge_datadir()
        / "pose_estimation/keypoints3d/trial_20251118a/production/epoch19_step9167/"
    )
    bodyseg_output_basedir = (
        get_poseforge_datadir()
        / "pose_estimation/bodyseg/trial_20251012b/production/epoch13_step18335/"
    )
    output_dir = get_outputs_dir() / "pose_estimation/"
    output_dir.mkdir(parents=True, exist_ok=True)

    trial = "20250613-fly1b-003"
    frame_range = (38 * 30, 78 * 30)  # 0:38 to 1:18 in PoseForge's 30 FPS visualization
    output_fps = 60  # 0.2x (NMF renders at 300 FPS)

    # 3D keypoints video
    print("Processing 3D keypoints video...")
    kpt3d_input_path = keypoints3d_output_basedir / trial / "ik_comparison.mp4"
    kpt3d_output_path = output_dir / "keypoints3d.mp4"
    crop_and_trim_video(
        input_path=kpt3d_input_path,
        output_path=kpt3d_output_path,
        frame_range=frame_range,
        output_fps=output_fps,
    )

    # Body segmentation video
    print("Processing body segmentation video...")
    bodyseg_input_path = (
        bodyseg_output_basedir / f"{trial}_model_prediction_not_flipped/viz.mp4"
    )
    bodyseg_output_path = output_dir / "bodyseg.mp4"
    crop_and_trim_video(
        input_path=bodyseg_input_path,
        output_path=bodyseg_output_path,
        frame_range=frame_range,
        col_range=(120, -30),
        output_fps=output_fps,
    )
