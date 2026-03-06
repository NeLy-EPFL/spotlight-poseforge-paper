from sppaper.common.plot import setup_matplotlib_params

setup_matplotlib_params()

from pathlib import Path
from collections import deque

import av
import numpy as np
import pandas as pd
import cv2
import yaml
import matplotlib.pyplot as plt
from scipy import ndimage, signal

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from spotlight_tools.postprocessing.visualize import _determine_adaptive_muscle_vrange

from sppaper.common.plot import find_font_path
from sppaper.common.resources import get_outputs_dir, get_spotlight_trials_dir

LEGEND_FONTLARGE = ImageFont.truetype(find_font_path("Arial", weight="bold"), 30)
LEGEND_FONTSMALL = ImageFont.truetype(find_font_path("Arial"), 25)
LEGEND_TEXTCOLOR = (255, 255, 255)
OUTPUT_CODEC = "libx265"
OUTPUT_CODEC_OPTIONS = {
    "crf": "20",  # high quality
    "preset": "veryslow",  # best compression (slowest encoding speed)
    "x265-params": "log-level=error",
}


def find_centermost_object(image, thr, min_size_norm):
    """Detect centermost object by binary thresholding and connected component analysis.

    1. Binarize image (2D array) by thresholding using thr
    2. Finding connected components that are >= image.size * min_size_norm in size
    3. Find the centermost connected components

    Return the mask for the object and its centroid position (x, y).
    """
    # Step 1: Binarize
    binary = image >= thr

    # Step 2: Label connected components and filter by minimum size
    labeled, num_features = ndimage.label(binary)
    min_size = image.size * min_size_norm

    valid_labels = [
        label
        for label in range(1, num_features + 1)
        if np.sum(labeled == label) >= min_size
    ]

    if not valid_labels:
        return None, None

    # Step 3: Find the centermost component
    image_center = np.array([image.shape[1] / 2, image.shape[0] / 2])  # (x, y)

    closest_label = None
    closest_dist = float("inf")
    closest_centroid = None

    for label in valid_labels:
        component_mask = labeled == label
        cy, cx = ndimage.center_of_mass(component_mask)
        dist = np.linalg.norm(np.array([cx, cy]) - image_center)
        if dist < closest_dist:
            closest_dist = dist
            closest_label = label
            closest_centroid = (cx, cy)

    object_mask = labeled == closest_label
    return object_mask, closest_centroid


def crop_with_padding(frame, centroid, cropdim):
    """Crop a image to (cropdim x cropdim) centered at centroid, 0-pad if out of bounds.

    Args:
        frame: 2D numpy array
        centroid: (x, y) center of the crop (col, row)
        cropdim: side length of the square crop

    Returns:
        Cropped (cropdim x cropdim) array of the same dtype as frame.
    """
    cx, cy = centroid
    colmin = int(round(cx - cropdim // 2))
    rowmin = int(round(cy - cropdim // 2))
    colmax = colmin + cropdim
    rowmax = rowmin + cropdim

    H, W = frame.shape[:2]
    src_c0, src_c1 = max(colmin, 0), min(colmax, W)
    src_r0, src_r1 = max(rowmin, 0), min(rowmax, H)
    dst_c0 = src_c0 - colmin
    dst_r0 = src_r0 - rowmin
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    dst_r1 = dst_r0 + (src_r1 - src_r0)

    cropped = np.zeros((cropdim, cropdim, *frame.shape[2:]), dtype=frame.dtype)
    cropped[dst_r0:dst_r1, dst_c0:dst_c1] = frame[src_r0:src_r1, src_c0:src_c1]
    return cropped


def add_legend(
    frame: np.ndarray,
    title: str,
    fps: int,
    playback_speed: float,
    title_font=LEGEND_FONTLARGE,
    info_font=LEGEND_FONTSMALL,
    text_color=LEGEND_TEXTCOLOR,
) -> np.ndarray:
    frame_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_img)

    draw.text(
        (title_font.size, title_font.size),
        title,
        fill=text_color,
        font=title_font,
    )
    draw.text(
        (title_font.size, 2.2 * title_font.size),
        f"Recorded at {fps} Hz",
        fill=text_color,
        font=info_font,
    )
    draw.text(
        (title_font.size, 2.2 * title_font.size + 1.2 * info_font.size),
        f"Played at {playback_speed}x real-time",
        fill=text_color,
        font=info_font,
    )

    return np.array(frame_img)


def get_timing_info(spotlight_trialdir):
    """Get metadata for matching behavior and muscle frames in a Spotlight trial.

    Returns:
        - beh_fps
        - mus_fps
        - syncratio: Ratio of behavior FPS to muscle FPS (should be an integer)
        - mus_frame_update_dict: Dict mapping behavior frame IDs to muscle frame IDs
             *IF AND ONLY IF* the muscle frame should be updated at that behavior frame
    """
    mus_frames_metadata_path = (
        spotlight_trialdir / "processed/muscle_frames_metadata.csv"
    )
    timing_metadata_path = spotlight_trialdir / "metadata/dual_recording_timing.yaml"

    with open(timing_metadata_path, "r") as f:
        timing_metadata = yaml.safe_load(f)
    beh_fps = timing_metadata["behavior_camera_fps"]
    syncratio = timing_metadata["sync_ratio"]
    assert isinstance(syncratio, int) and syncratio > 0, "sync_ratio not a pos integer"
    mus_fps = int(beh_fps / syncratio)
    assert np.isclose(mus_fps, beh_fps / syncratio), "muscle FPS not a whole number"

    mus_frames_metadata_df = pd.read_csv(mus_frames_metadata_path)
    beh_keyframeids = mus_frames_metadata_df["corresponding_behavior_frame_id"].values
    assert np.all(np.diff(beh_keyframeids) == syncratio), "inconsistent syncratio"

    # Build dict: behavior frame ID at which to update muscle frame -> muscle frame ID
    beh2mus_matches = {
        int(row["corresponding_behavior_frame_id"]): int(row["muscle_frame_id"])
        for i, row in mus_frames_metadata_df.iterrows()
    }

    return beh_fps, mus_fps, beh2mus_matches


def read_muscle_frame(mus_frames_dir, musframe_id):
    musframe_path = mus_frames_dir / f"muscle_frame_{musframe_id:09d}.tif"
    musframe = cv2.imread(str(musframe_path), cv2.IMREAD_UNCHANGED)
    if musframe is None:
        raise FileNotFoundError(
            f"Muscle frame not found or unreadable: {musframe_path}"
        )
    return musframe


def plot_centroid_traj(centroid_hist, output_path):
    fig, ax = plt.subplots(figsize=(2, 2), tight_layout=True)
    centroid_hist = np.array(centroid_hist)
    ax.plot(centroid_hist[:, 0], centroid_hist[:, 1])
    ax.set_title("Fly centroid trajectory (filtered)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    fig.savefig(output_path)


def musframe_16bit_to_8bit_overview(musframe_16bit, vrange):
    vrange_span = vrange[1] - vrange[0]
    musframe_norm = (musframe_16bit.astype(np.float32) - vrange[0]) / vrange_span
    musframe_norm = np.clip(musframe_norm, 0, 1)
    musframe_8bit = (musframe_norm * 255).astype(np.uint8)
    musframe_rgb = np.zeros((*musframe_8bit.shape, 3), dtype=np.uint8)
    musframe_rgb[:, :, 1] = musframe_8bit  # green channel only
    return musframe_rgb


def make_behframe_disp(
    behframe_plot, centroid_filt, cropdim, beh_fps, output_playback_speed
):
    """Crop behavior frame and add legend overlay."""
    behframe_crop = crop_with_padding(behframe_plot, centroid_filt, cropdim)
    behframe_disp = np.stack([behframe_crop] * 3, axis=-1)  # to RGB
    behframe_disp = add_legend(
        behframe_disp,
        "Behavior recording",
        beh_fps,
        output_playback_speed,
    )
    return behframe_disp


def make_musframe_disp(
    musframes_dir,
    musframe_id,
    mus_thr,
    mus_minsize_norm,
    cropdim,
    mus_vrange,
    mus_fps,
    output_playback_speed,
):
    """Read, crop, and convert a muscle frame to an 8-bit RGB display frame."""
    musframe = read_muscle_frame(musframes_dir, musframe_id)
    _, mus_centroid = find_centermost_object(musframe, mus_thr, mus_minsize_norm)
    musframe_crop = crop_with_padding(musframe, mus_centroid, cropdim)
    musframe_disp = musframe_16bit_to_8bit_overview(musframe_crop, mus_vrange)
    musframe_disp = add_legend(
        musframe_disp,
        "Muscle imaging (GCaMP8m)",
        mus_fps,
        output_playback_speed,
    )
    return musframe_disp


def gaussian_weighted_centroid(
    behcentroids_buffer, gaus_kernel, behframe_offset, behframe_plotterid
):
    """Compute Gaussian-weighted centroid from the buffer for the current plot frame."""
    centroids = np.array([ctr for _, ctr in behcentroids_buffer])
    kernel_start = max(0, behframe_offset - behframe_plotterid)
    weights = gaus_kernel[kernel_start : kernel_start + len(centroids)]
    return np.sum(centroids * weights[:, None], axis=0) / np.sum(weights)


def make_summary_video(
    spotlight_trialdir,
    output_path,
    output_playback_speed,
    output_nbehframes=None,
    beh_thr=60,
    mus_thr=200,
    mus_vrange_quantiles=(97.0, 99.995),
    mus_vrange_samplerate=0.05,
    beh_minsize_norm=0.01,
    mus_minsize_norm=0.005,
    centroid_filtwindow=15,
    centroid_filtsigma_norm=0.5,
    cropdim=1100,
    output_codec=OUTPUT_CODEC,
    output_codec_options=OUTPUT_CODEC_OPTIONS,
):
    behvideo_path = spotlight_trialdir / "processed/fullsize_behavior_video.mkv"
    musframes_dir = spotlight_trialdir / "processed/fullsize_muscle_images"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    beh_fps, mus_fps, beh2mus_matches = get_timing_info(spotlight_trialdir)
    output_fps = int(beh_fps * output_playback_speed)

    mus_vrange = _determine_adaptive_muscle_vrange(
        muscle_vrange_quantiles=mus_vrange_quantiles,
        muscle_vrange_quantiles_sample_rate=mus_vrange_samplerate,
        muscle_image_paths=sorted(list(musframes_dir.glob("*.tif"))),
    )

    gaus_kernel_sigma = centroid_filtwindow * centroid_filtsigma_norm * 0.5  # per side
    gaus_kernel = signal.windows.gaussian(centroid_filtwindow, std=gaus_kernel_sigma)
    behframe_offset = centroid_filtwindow // 2

    behframes_buffer = deque(maxlen=centroid_filtwindow)
    behcentroids_buffer = deque(maxlen=centroid_filtwindow)
    musframe_disp_buffer = None
    filtered_centroids_hist = []

    with av.open(str(behvideo_path)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"  # enable ffmpeg multithreading for decoding

        with av.open(str(output_path), mode="w") as out_container:
            out_stream = out_container.add_stream(output_codec, rate=output_fps)
            out_stream.pix_fmt = "yuv420p"
            out_stream.thread_type = "AUTO"  # multithreaded encoding
            out_stream.options = output_codec_options
            out_stream.width = cropdim * 2
            out_stream.height = cropdim

            behframe_readerid = 0
            with tqdm(total=output_nbehframes + behframe_offset) as pbar:
                is_done = False

                for packet in in_container.demux(in_stream):
                    for frame in packet.decode():
                        # Buffer behavior frame and centroid
                        behframe_read = frame.to_ndarray(format="rgb24")[:, :, 0]
                        _, behcentroid = find_centermost_object(
                            behframe_read, beh_thr, beh_minsize_norm
                        )
                        if behcentroid is None:
                            if len(behcentroids_buffer) == 0:
                                raise ValueError("No fly detected on the first frame")
                            print(
                                f"Warning: no fly detected on frame "
                                f"{behframe_readerid}, using last known centroid"
                            )
                            behcentroid = behcentroids_buffer[-1][1]
                        behframes_buffer.append((behframe_readerid, behframe_read))
                        behcentroids_buffer.append((behframe_readerid, behcentroid))

                        # Determine which frame we're plotting
                        behframe_plotterid = behframe_readerid - behframe_offset
                        if behframe_plotterid >= output_nbehframes:
                            # Early stop
                            is_done = True
                            break
                        behframe_readerid += 1
                        pbar.update(1)
                        if behframe_plotterid < 0:
                            continue  # not enough buffered frames yet

                        # Compute filtered centroid
                        centroid_filt = gaussian_weighted_centroid(
                            behcentroids_buffer,
                            gaus_kernel,
                            behframe_offset,
                            behframe_plotterid,
                        )
                        filtered_centroids_hist.append(centroid_filt)

                        # Build behavior display frame
                        _frameid, behframe_plot = behframes_buffer[
                            -(behframe_offset + 1)
                        ]
                        assert (
                            _frameid == behframe_plotterid
                        ), "frame buffering mismatch"
                        behframe_disp = make_behframe_disp(
                            behframe_plot,
                            centroid_filt,
                            cropdim,
                            beh_fps,
                            output_playback_speed,
                        )

                        # Update muscle display frame if needed
                        musframe_to_update = beh2mus_matches.get(behframe_plotterid)
                        if musframe_to_update is not None:
                            musframe_disp_buffer = make_musframe_disp(
                                musframes_dir,
                                musframe_to_update,
                                mus_thr,
                                mus_minsize_norm,
                                cropdim,
                                mus_vrange,
                                mus_fps,
                                output_playback_speed,
                            )

                        # Composite and encode output frame
                        output_frame = np.zeros(
                            (cropdim, cropdim * 2, 3), dtype=np.uint8
                        )
                        output_frame[:, :cropdim, :] = behframe_disp
                        if musframe_disp_buffer is not None:
                            output_frame[:, cropdim:, :] = musframe_disp_buffer
                        out_frame = av.VideoFrame.from_ndarray(
                            output_frame, format="rgb24"
                        )
                        for out_packet in out_stream.encode(out_frame):
                            out_container.mux(out_packet)

                    if is_done:
                        break  # early stop

            # Flush output encoder
            for out_packet in out_stream.encode():
                out_container.mux(out_packet)

    # Plot centroid trajectory for debugging
    plot_centroid_traj(
        filtered_centroids_hist,
        output_path.parent / f"{output_path.stem}_centroid_traj.pdf",
    )


if __name__ == "__main__":
    input_path = get_spotlight_trials_dir() / "20250613-fly1b-012/"
    output_path = get_outputs_dir() / "spotlight_data_sample/sample_cropped.mp4"
    output_nbehframes = 100  # 10 * 330  # 10 sec at 330 FPS

    make_summary_video(
        input_path,
        output_path,
        output_playback_speed=0.2,
        output_nbehframes=output_nbehframes,
    )
