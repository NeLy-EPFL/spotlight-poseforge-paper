from poseforge.util.plot import configure_matplotlib_style as set_poseforge_mpl_style

set_poseforge_mpl_style(dark_background=True)

from tempfile import mkdtemp
from shutil import copytree, copyfile, rmtree
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import h5py
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont
from tqdm import trange
from joblib import Parallel, delayed

from poseforge.neuromechfly.postprocessing import visualize_subsegment
from poseforge.pose import visualize_latent_trajectory

from sppaper.common.resources import get_poseforge_datadir, get_outputs_dir
from sppaper.common.plot import find_font_path

_FONT_PATH = find_font_path("Arial", weight="normal")

FONTSIZE = 18
FONT = ImageFont.truetype(_FONT_PATH, FONTSIZE)
FONT_SMALL = ImageFont.truetype(_FONT_PATH, FONTSIZE * 0.8)
VARIANTS_PER_ROW = 4
# NMF frames are rendered at this FPS
SOURCE_DATA_FREQ = 300
# latent traj videos are too wide - crop from the sides. Original dimension: 1024x768
LATENT_TRAJ_PANEL_XCROP = slice(200, -100)
# Spacing between NMF overview and synth data, and between synth data and latent trajs
VSPACE = 20

OUTPUT_CODEC = "libx265"
OUTPUT_QUALITY = 8
OUTPUT_PIXELFORMAT = "yuv420p"
OUTPUT_PLAYSPEED = 0.2
OUTPUT_FPS = SOURCE_DATA_FREQ * OUTPUT_PLAYSPEED


def make_nmf_overview_video(processed_subsegment_dir, output_path):

    tempdir = mkdtemp()
    copytree(processed_subsegment_dir, tempdir, dirs_exist_ok=True)
    visualize_subsegment(
        processed_subsegment_dir,
        fps=OUTPUT_FPS,
        azimuth_rotation_period=600,
        dark_background=True,
    )

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    copyfile(Path(tempdir) / "visualization.mp4", output_path)
    rmtree(tempdir)


def make_latent_trajectories_videos(
    inferred_latents_path: Path,
    output_dir: Path,
    training_stages: list[str],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(inferred_latents_path, "r") as f:
        for feature in ["h_features_pooled"]:  # ["h_features_pooled", "z_features"]:
            for stage in training_stages:
                print(f"Visualizing {feature} at stage {stage}...")
                latent_space_data = f[f"{feature}/{stage}"][:]
                visualize_latent_trajectory(
                    latent_space_data=latent_space_data,
                    source_data_freq=SOURCE_DATA_FREQ,
                    play_speed=OUTPUT_PLAYSPEED,
                    output_fps=OUTPUT_FPS,
                    video_path=output_dir / f"{feature}_{stage}.mp4",
                    headless=True,
                )


def combine_videos_with_synthetic_variants(
    nmf_overview_path, synth_output_dir, lattraj_paths, output_path, n_job=-1
):
    # Load NMF overview video
    nmf_overview_frames = iio.imread(nmf_overview_path)

    # Load synthetic variant videos
    variant_frames = {}
    variant_paths = sorted(synth_output_dir.glob("translated_*.mp4"))
    variant_frames_shape = None
    for i_variant, variant_path in enumerate(variant_paths):
        frames = iio.imread(variant_path)
        variant_frames[i_variant] = frames
        if variant_frames_shape is None:
            assert frames.shape[0] == nmf_overview_frames.shape[0]
            variant_frames_shape = frames.shape
        else:
            assert frames.shape == variant_frames_shape

    # Load latent trajectory videos
    lattraj_frames = {}
    lattraj_frames_shape = None
    for (space, stage), lattraj_path in lattraj_paths.items():
        frames = iio.imread(lattraj_path)
        frames = frames[:, LATENT_TRAJ_PANEL_XCROP]  # Crop latent trajectory videos
        lattraj_frames[(space, stage)] = frames
        if lattraj_frames_shape is None:
            assert frames.shape[0] == nmf_overview_frames.shape[0]
            lattraj_frames_shape = frames.shape
        else:
            assert frames.shape == lattraj_frames_shape

    # Compose output frames
    def process_frame(i):
        curr_nmf_overview_frame = nmf_overview_frames[i]
        curr_variant_frames = [frames[i] for _, frames in variant_frames.items()]
        curr_lattraj_frames = {k: frames[i] for k, frames in lattraj_frames.items()}
        return compose_frame(
            curr_nmf_overview_frame, curr_variant_frames, curr_lattraj_frames
        )

    nframes = nmf_overview_frames.shape[0]
    output_frames = Parallel(n_jobs=n_job)(
        delayed(process_frame)(i) for i in trange(nframes, desc="Composing frames")
    )

    # Write output video
    iio.imwrite(
        output_path,
        output_frames,
        extension=".mp4",
        fps=OUTPUT_FPS,
        codec=OUTPUT_CODEC,
        quality=OUTPUT_QUALITY,
        pixelformat=OUTPUT_PIXELFORMAT,
    )


def compose_frame(nmf_viz_frame, variant_frames, lattraj_frames):
    # ========== Figure out output frame size ==========
    # NMF overview panels
    nmf_viz_nrows_raw, nmf_viz_ncols_raw, _ = nmf_viz_frame.shape

    # Synthetic data panels
    synth_nrows, synth_ncols, _ = variant_frames[0].shape
    nvariants = len(variant_frames)
    output_ncols = VARIANTS_PER_ROW * synth_ncols
    n_rows_variants = (nvariants + VARIANTS_PER_ROW - 1) // VARIANTS_PER_ROW
    nmf_viz_nrows = int(nmf_viz_nrows_raw * (output_ncols / nmf_viz_ncols_raw))

    # Latent trajectory panels
    lattraj_ncols = output_ncols // 2
    lattraj_nrows_raw, lattraj_ncols_raw, _ = next(iter(lattraj_frames.values())).shape
    lattraj_nrows = int(lattraj_nrows_raw * (lattraj_ncols / lattraj_ncols_raw))

    output_nrows = (
        nmf_viz_nrows + synth_nrows * n_rows_variants + lattraj_nrows + 3 * VSPACE
    )

    # ========== Create empty canvas ==========
    output_frame = np.zeros((output_nrows, output_ncols, 3), dtype=np.uint8)
    output_frame[:VSPACE] = 255

    # ========== Add top part (NMF visualization) ==========
    nmf_viz_resized = resize(
        nmf_viz_frame, (nmf_viz_nrows, output_ncols), anti_aliasing=True
    )
    nmf_viz_resized = (nmf_viz_resized * 255).astype(np.uint8)
    output_frame[:nmf_viz_nrows] = nmf_viz_resized

    # ========== Add middle rows (synthetic variant videos) ==========
    for i_variant, variant_frame in enumerate(variant_frames):
        row_idx = i_variant // VARIANTS_PER_ROW
        col_idx = i_variant % VARIANTS_PER_ROW
        row_start = nmf_viz_nrows + row_idx * synth_nrows + VSPACE * 2
        row_end = row_start + synth_nrows
        col_start = col_idx * synth_ncols
        col_end = col_start + synth_ncols
        output_frame[row_start:row_end, col_start:col_end] = variant_frame

    # ========== Add bottom rows (latent trajectories) ==========
    lattraj_ybase = nmf_viz_nrows + synth_nrows * n_rows_variants + VSPACE * 3
    lattraj_topleft_corners = {
        ("h", "untrained"): (0, lattraj_ybase),
        ("h", "trained"): (lattraj_ncols, lattraj_ybase),
    }
    for (feature, stage), topleft_corner in lattraj_topleft_corners.items():
        col_start, row_start = topleft_corner
        row_end = row_start + lattraj_nrows
        col_end = col_start + lattraj_ncols
        lattraj_frame = lattraj_frames[(feature, stage)]
        lattraj_frame_resized = resize(
            lattraj_frame, (lattraj_nrows, lattraj_ncols), anti_aliasing=True
        )
        lattraj_frame_resized = (lattraj_frame_resized * 255).astype(np.uint8)
        output_frame[row_start:row_end, col_start:col_end] = lattraj_frame_resized

    # ========== Add text labels ==========
    img = Image.fromarray(output_frame)
    draw = ImageDraw.Draw(img)

    # Top label
    draw.text(
        (output_ncols // 2, int(FONTSIZE * 0.8)),
        f"Rendered from simulation at {SOURCE_DATA_FREQ} Hz, played at {OUTPUT_PLAYSPEED}x real-time",
        fill=(255, 255, 255),
        font=FONT,
        anchor="mt",
    )

    # Synthetic data
    for i_variant in range(nvariants):
        row_idx = i_variant // VARIANTS_PER_ROW
        col_idx = i_variant % VARIANTS_PER_ROW
        textpos = (
            (col_idx + 0.5) * synth_ncols,
            nmf_viz_nrows + row_idx * synth_nrows + int(FONTSIZE * -0.5),
        )
        draw.text(
            textpos,
            f"Synthetic data (variant {i_variant + 1})",
            fill=(255, 255, 255),
            font=FONT,
            anchor="mt",
        )

    for (feature, stage), topleft_corner in lattraj_topleft_corners.items():
        col_start, row_start = topleft_corner
        text_midtop_pos = (
            col_start + lattraj_ncols // 2,
            row_start + int(FONTSIZE * -0.5),
        )
        if stage == "untrained":
            stage_text = "Naive latent space (h)"
        else:
            stage_text = "Learned latent space (h)"
        draw.text(
            text_midtop_pos,
            stage_text,
            fill=(255, 255, 255),
            font=FONT,
            anchor="mt",
        )

    return np.array(img)


if __name__ == "__main__":
    subsegment = "BO_Gal4_fly5_trial005/segment_003/subsegment_002"
    synth_output_dir = (
        get_poseforge_datadir()
        / f"style_transfer/production/translated_videos/{subsegment}"
    )
    nmf_sim_dir = get_poseforge_datadir() / f"nmf_rendering/{subsegment}/"
    contrastive_encoder_output_path = Path(
        get_poseforge_datadir()
        / f"pose_estimation/contrastive_pretraining/trial_20251117a/inference/{subsegment}/contrastive_latents.h5"
    )
    contrastive_encoder_training_stages = ["untrained", "epoch009_step003055"]

    output_dir = get_outputs_dir() / "synthetic_sample_and_latent_trajectories/"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate overview video for NMF simulation & rendering
    print("Generating NMF overview video...")
    nmf_overview_video_path = Path(output_dir) / "nmf_overview.mp4"
    make_nmf_overview_video(nmf_sim_dir, output_path=nmf_overview_video_path)

    # Generate visualization for learned latent spaces
    print("Generating learned latent space visualizations...")
    make_latent_trajectories_videos(
        contrastive_encoder_output_path,
        output_dir=output_dir,
        training_stages=contrastive_encoder_training_stages,
    )
    lattraj_paths = {
        ("h", "untrained"): output_dir / "h_features_pooled_untrained.mp4",
        ("h", "trained"): output_dir / "h_features_pooled_epoch009_step003055.mp4",
    }

    # Combine the videos above with synthetic data variants
    print("Combining videos...")
    output_path = output_dir / "combined_visualization.mp4"
    combine_videos_with_synthetic_variants(
        nmf_overview_video_path, synth_output_dir, lattraj_paths, output_path
    )

    print(f"Done. Output video saved to {output_path}")
