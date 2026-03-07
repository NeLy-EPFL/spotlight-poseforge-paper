from importlib.resources import files

from poseforge.production.spotlight.core import SpotlightRecordingProcessor

from sppaper.common.resources import get_spotlight_trials_dir, get_outputs_dir


def process_spotlight_trial(
    spotlight_trial_dir,
    output_dir,
    model_config_path,
    output_fps,
    output_playspeed,
    frame_range=None,
):
    # Create Spotlight trial structure, with symlinks to original input data
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir_symlink = output_dir / "metadata"
    processed_dir_symlink = output_dir / "processed"
    input_metadata_dir = spotlight_trial_dir / "metadata"
    input_processed_dir = spotlight_trial_dir / "processed"
    print(input_metadata_dir, input_processed_dir)
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
    recording.visualize_keypoints3d(output_playspeed, output_fps)
    recording.predict_body_segmentation(loading_n_workers=8)
    recording.visualize_bodyseg_predictions(output_playspeed, output_fps)
    recording.visualize_keypoints3d(
        output_playspeed, output_fps, frame_range=frame_range
    )
    recording.visualize_bodyseg_predictions(
        output_playspeed, output_fps, frame_range=frame_range
    )


if __name__ == "__main__":
    model_config_path = files("poseforge.production.spotlight").joinpath("config.yaml")
    trial = "20250613-fly1b-003"
    # frame_range = (38 * 30, 82 * 30)  # 0:38 to 1:22 in PoseForge's 30 FPS visualization
    frame_range = (int(6 * 0.2 * 330), int(18 * 0.2 * 330))  # 0:06-0:18 at 0.2x speed
    input_spotlight_trial_dir = get_spotlight_trials_dir() / trial
    output_dir = get_outputs_dir() / "pose_estimation" / trial

    process_spotlight_trial(
        input_spotlight_trial_dir,
        output_dir,
        model_config_path,
        output_fps=30,
        output_playspeed=0.2,
        frame_range=frame_range,
    )
