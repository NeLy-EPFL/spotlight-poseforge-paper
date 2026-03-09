"""
Generate a video showing muscle activity during stimulation experiments.

The video consists of:
- Top left: Raw muscle frames (normalized 10th to 98th percentile)
- Top middle: Raw muscle frames (same normalization)
- Top right: Muscle frames with segmentation overlay (only selected segments)
- Bottom: Muscle activity traces (ΔF/F₀) for selected segments

Usage:
    python make_muscle_activity_video.py --exp_folder /path/to/experiment
"""

from pathlib import Path
import argparse
import sys
from joblib import Parallel, delayed
import multiprocessing

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import cv2
import h5py
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import av

from spotlight_tools.postprocessing.muscle import (
    match_muscle_frameid_to_behavior_frameid,
)
from sppaper.common.muscle import (
    compute_muscle_activity,
    compute_delta_f_over_f,
    create_segmentation_overlay,
)

# Import configuration parameters
from figure_config import (
    # Preprocessing
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    # Video output
    OUTPUT_CODEC,
    CODEC_PRESET,
    CODEC_CRF,
    OUTPUT_FPS,
    # Normalization
    NORM_LOWER_PERCENTILE,
    NORM_UPPER_PERCENTILE,
    VMAX_SHIFT,
    # Muscle activity
    TOP_K_PIXELS,
    # Segmentation
    MORPH_KERNEL_SIZE,
    MORPH_N_ITERATIONS,
    DILATION_KERNELS,
    # Baseline
    BASELINE_WINDOW_SEC,
    # Layout
    TRACE_PANEL_HEIGHT_RATIO,
    # Overlay
    SEGMENT_BLEND_ORIGINAL,
    SEGMENT_BLEND_COLOR,
    TOP_K_HIGHLIGHT_FACTOR,
    # Colors and styling
    SEGMENT_ORDER,
    SEGMENT_COLORS,
    TRACE_LINEWIDTH,
    TRACE_BASELINE_ALPHA,
    TRACE_CURRENT_TIME_WIDTH,
    TRACE_CURRENT_TIME_ALPHA,
    STIM_PERIOD_ALPHA,
)

# Build codec options from config
CODEC_OPTIONS = {
    "crf": CODEC_CRF,
    "preset": CODEC_PRESET,
}


def create_trace_panel_fast(time_sec, traces, segment_names, current_time_idx,
                           stim_starts, stim_ends, panel_width, panel_height):
    """Create trace panel optimized for speed (creates new figure each time but faster)."""
    fig = plt.figure(figsize=(panel_width / 100, panel_height / 100), 
                     dpi=100, facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    
    n_segments = len(segment_names)
    offset_spacing = traces.max()
    
    # Plot each trace (only up to current time for efficiency)
    for j, segment_name in enumerate(segment_names):
        offset = j * offset_spacing
        color = SEGMENT_COLORS.get(segment_name, "#ffffff")
        
        # Plot the trace
        ax.plot(time_sec, traces[:, j] + offset, color=color, linewidth=TRACE_LINEWIDTH)
        
        # Add horizontal baseline
        ax.axhline(y=offset, color='gray', linestyle='--', alpha=TRACE_BASELINE_ALPHA, linewidth=0.5)
    
    # Mark current time
    current_time = time_sec[current_time_idx]
    ax.axvline(x=current_time, color='red', linestyle='-', linewidth=TRACE_CURRENT_TIME_WIDTH, alpha=TRACE_CURRENT_TIME_ALPHA)
    
    # Mark stimulation periods
    for start_time, end_time in zip(stim_starts, stim_ends):
        ax.axvspan(start_time, end_time, alpha=STIM_PERIOD_ALPHA, color='yellow')
    
    # Set y-ticks to show segment names
    ytick_positions = [j * offset_spacing for j in range(n_segments)]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(segment_names, fontsize=8, color='white')
    
    # Labels and styling
    ax.set_xlabel("Time (s)", fontsize=10, color='white')
    ax.set_ylabel("Muscle Segment", fontsize=10, color='white')
    ax.set_title(r"Muscle Activity ($\Delta$F/F$_0$)", fontsize=12, color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, color='white')
    ax.set_xlim(time_sec[0], time_sec[-1])
    
    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    
    return img[:, :, :3]  # Drop alpha channel


def prerender_all_trace_panels(time_sec, traces, segment_names, stim_start_times,
                                stim_end_times, panel_width, panel_height, n_jobs=-1):
    """Pre-render all trace panels in parallel for maximum speed."""
    print(f"Pre-rendering {len(time_sec)} trace panels in parallel...")
    
    def render_single_trace(idx):
        return create_trace_panel_fast(
            time_sec, traces, segment_names, idx,
            stim_start_times, stim_end_times,
            panel_width, panel_height
        )
    
    # Parallel rendering
    trace_panels = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(render_single_trace)(i) 
        for i in tqdm(range(len(time_sec)), desc="Rendering traces")
    )
    
    return trace_panels


def generate_video(exp_folder, segments_to_show, output_path, max_frames=None):
    """Generate muscle activity video for the experiment."""
    exp_folder = Path(exp_folder)
    processed_folder = exp_folder / "processed"
    metadata_folder = exp_folder / "metadata"
    
    # Load metadata
    print("Loading metadata...")
    import pandas as pd
    mf_metadata = pd.read_csv(processed_folder / "muscle_frames_metadata.csv")
    bf_metadata = pd.read_csv(processed_folder / "behavior_frames_metadata.csv")
    
    # Limit frames if requested
    if max_frames is not None:
        print(f"Limiting to first {max_frames} frames for profiling")
        mf_metadata = mf_metadata.iloc[:max_frames]
    duo_yaml = metadata_folder / "dual_recording_timing.yaml"
    
    # Load experiment parameters to get stimulation times
    exp_parameters = yaml.safe_load(
        (metadata_folder / "experiment_parameters.yaml").read_text()
    )
    experiment_protocol = exp_parameters["experiment_protocol"]
    
    # Parse stimulation protocol
    def parse_protocol(protocol):
        lines = protocol.split(";")
        prev_state = {}
        stim_starts = {}
        stim_ends = {}
        for line in lines:
            if len(line.strip()) < 3:
                continue
            fnumber, channel, state = line.split("/")
            channel_prev_state = prev_state.get(channel, "off")
            if state == "on" and channel_prev_state == "off":
                if channel not in stim_starts:
                    stim_starts[channel] = []
                stim_starts[channel].append(int(fnumber))
            elif state == "off" and channel_prev_state == "on":
                if channel not in stim_ends:
                    stim_ends[channel] = []
                stim_ends[channel].append(int(fnumber))
            prev_state[channel] = state
        return stim_starts, stim_ends
    
    stim_starts, stim_ends = parse_protocol(experiment_protocol)
    
    # Use first channel with stimulations
    channel = list(stim_starts.keys())[0]
    stim_start_frames = stim_starts[channel]
    stim_end_frames = stim_ends[channel]
    
    # Load segmentation data
    print("Loading segmentation data...")
    segmap_folder = exp_folder.parent / "poseforge_output/bodyseg/epoch14_step12000" / \
                   (exp_folder.name + "_model_prediction_not_flipped")
    with h5py.File(segmap_folder / "bodyseg_pred.h5", "r") as f:
        segmaps = f["pred_segmap"][:]
        frame_ids = f["frame_ids"][:]
        seg_labels = list(f["pred_segmap"].attrs["class_labels"])
    
    # Filter segments to show based on user input
    segments_to_analyze = [seg for seg in seg_labels if any(
        s.lower() in seg.lower() for s in segments_to_show
    )]
    
    # Sort segments according to canonical order (RF, LF, RM, LM, RH, LH)
    segments_to_analyze = sorted(segments_to_analyze, 
                                 key=lambda s: SEGMENT_ORDER.index(s) if s in SEGMENT_ORDER else 999)
    
    print(f"Analyzing segments: {segments_to_analyze}")
    
    # Compute muscle activity and top-k pixel masks
    muscle_activity, top_k_masks, processed_masks_dict = compute_muscle_activity(
        mf_metadata, processed_folder, segmaps, frame_ids,
        seg_labels, segments_to_analyze, duo_yaml, k=TOP_K_PIXELS,
        morph_kernel_size=MORPH_KERNEL_SIZE,
        morph_n_iterations=MORPH_N_ITERATIONS,
        dilation_kernels=DILATION_KERNELS,
        bilateral_d=BILATERAL_D,
        bilateral_sigma_color=BILATERAL_SIGMA_COLOR,
        bilateral_sigma_space=BILATERAL_SIGMA_SPACE
    )
    
    # Compute time axis
    time_sec = (mf_metadata["received_time_us"] - 
                mf_metadata["received_time_us"].iloc[0]) / 1e6
    
    # Convert stimulation frames to times
    stim_start_times = []
    stim_end_times = []
    for start_frame, end_frame in zip(stim_start_frames, stim_end_frames):
        start_time = bf_metadata[bf_metadata["behavior_frame_id"] == start_frame][
            "received_time_us"
        ].values
        end_time = bf_metadata[bf_metadata["behavior_frame_id"] == end_frame][
            "received_time_us"
        ].values
        if len(start_time) > 0 and len(end_time) > 0:
            stim_start_times.append(
                (start_time[0] - bf_metadata["received_time_us"].iloc[0]) / 1e6
            )
            stim_end_times.append(
                (end_time[0] - bf_metadata["received_time_us"].iloc[0]) / 1e6
            )
    
    # Compute ΔF/F₀ (treating all stimulations as valid)
    print("Computing ΔF/F₀...")
    # Convert baseline window to behavior frames
    behavior_fps = exp_parameters["behavior_fps"]
    baseline_window_frames = int(BASELINE_WINDOW_SEC * behavior_fps)
    
    delta_f_over_f = compute_delta_f_over_f(
        muscle_activity, mf_metadata, bf_metadata,
        stim_start_frames, stim_end_frames,
        baseline_window_frames,
        valid_stim_mask=None,  # All stimulations considered valid
        duo_yaml=duo_yaml
    )
    
    # Load all muscle frames to compute global normalization percentiles AND cache them
    print("Loading and caching all muscle frames...")
    cached_frames = {}
    all_pixel_values = []
    for i, mf_id in enumerate(tqdm(mf_metadata["muscle_frame_id"], 
                                   desc="Loading frames")):
        mf_id = int(mf_id)
        frame = cv2.imread(
            str(processed_folder / f"aligned_muscle_images/muscle_frame_{mf_id:09d}.tif"),
            cv2.IMREAD_UNCHANGED
        )
        cached_frames[mf_id] = frame  # Cache for later use
        all_pixel_values.append(frame.flatten())
    
    all_frames_sample = np.concatenate(all_pixel_values)
    print(f"Computing percentiles from {len(all_pixel_values)} frames ({all_frames_sample.size:,} total pixels)...")
    
    # Compute normalization percentiles ONCE (not per frame!)
    vmin = np.percentile(all_frames_sample, NORM_LOWER_PERCENTILE)
    vmax = np.percentile(all_frames_sample, NORM_UPPER_PERCENTILE) + VMAX_SHIFT
    print(f"Normalization range: {vmin:.1f} to {vmax:.1f} (with VMAX_SHIFT={VMAX_SHIFT})")
    
    del all_pixel_values  # Free memory
    del all_frames_sample  # Don't need this anymore
    
    # Get first frame to determine dimensions
    first_mf_id = mf_metadata.iloc[0]["muscle_frame_id"].astype(int)
    first_frame = cached_frames[first_mf_id]
    frame_height, frame_width = first_frame.shape
    
    # Get behavior frames folder
    beh_frames_folder = exp_folder.parent / "spotlight_aligned_and_cropped" / exp_folder.name / "all"
    
    # Layout parameters (3 panels on top: behavior, muscle raw, muscle overlay)
    top_panel_width = frame_width
    top_panel_height = frame_height
    trace_panel_height = int(frame_height * TRACE_PANEL_HEIGHT_RATIO)
    total_width = top_panel_width * 3  # Changed from 2 to 3
    total_height = top_panel_height + trace_panel_height
    
    # Pre-render all trace panels in parallel (this is the slow part!)
    print(f"Pre-rendering {len(time_sec)} trace panels...")
    print("Note: matplotlib rendering is slow - this may take several minutes")
    trace_panels = prerender_all_trace_panels(
        time_sec.values, delta_f_over_f, segments_to_analyze,
        stim_start_times, stim_end_times, 
        total_width, trace_panel_height,
        n_jobs=-1  # Use all available cores
    )
    
    # Initialize video writer
    print(f"Creating video: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream(OUTPUT_CODEC, rate=OUTPUT_FPS)
        stream.pix_fmt = "yuv420p"
        stream.width = total_width
        stream.height = total_height
        stream.options = CODEC_OPTIONS
        
        # Generate video frames (now much faster - using cached frames)
        for i, mf_id in enumerate(tqdm(mf_metadata["muscle_frame_id"].astype(int), 
                                       desc="Encoding video")):
            # Use cached muscle frame (already loaded)
            muscle_frame = cached_frames[mf_id].astype(float)
            
            # Normalize frame using pre-computed percentiles
            muscle_norm = np.clip((muscle_frame - vmin) / (vmax - vmin + 1e-8), 0, 1)
            muscle_norm = (muscle_norm * 255).astype(np.uint8)
            
            # Get corresponding behavior frame
            bf_frame_id = match_muscle_frameid_to_behavior_frameid(
                mf_id, dual_recording_timing_metadata_path=duo_yaml
            )
            behavior_frame_path = beh_frames_folder / f"frame_{bf_frame_id:09d}.jpg"
            
            # Load behavior frame
            if behavior_frame_path.exists():
                behavior_frame = cv2.imread(str(behavior_frame_path), cv2.IMREAD_UNCHANGED)
                if behavior_frame is not None:
                    # Convert to grayscale
                    if len(behavior_frame.shape) == 3:
                        behavior_gray = cv2.cvtColor(behavior_frame, cv2.COLOR_BGR2GRAY)
                    else:
                        behavior_gray = behavior_frame
                    
                    # Get original segmap for behavior overlay (not dilated)
                    segmap = segmaps[frame_ids == bf_frame_id][0]
                    behavior_height, behavior_width = behavior_gray.shape
                    
                    # Convert original segmap to list of masks (one per segment)
                    original_masks_behavior = []
                    for seg_id in range(len(seg_labels)):
                        mask = (segmap == seg_id).astype(np.uint8)
                        mask_resized = cv2.resize(
                            mask, (behavior_width, behavior_height),
                            interpolation=cv2.INTER_NEAREST
                        )
                        original_masks_behavior.append(mask_resized)
                    
                    # Create behavior overlay using original masks (no top-k highlighting)
                    behavior_overlay = create_segmentation_overlay(
                        behavior_gray, original_masks_behavior, seg_labels, segments_to_analyze,
                        SEGMENT_COLORS, SEGMENT_BLEND_ORIGINAL, SEGMENT_BLEND_COLOR,
                        TOP_K_HIGHLIGHT_FACTOR, None, None
                    )
                    
                    # Resize behavior overlay to match muscle frame dimensions
                    behavior_overlay_resized = cv2.resize(
                        behavior_overlay, (frame_width, frame_height),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    # Fallback: black frame
                    behavior_overlay_resized = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            else:
                # Fallback: black frame
                behavior_overlay_resized = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            # Get processed masks for this frame
            frame_processed_masks = processed_masks_dict[i]
            
            # Resize processed masks to muscle frame dimensions
            frame_processed_masks_resized = []
            for mask in frame_processed_masks:
                mask_resized = cv2.resize(
                    mask.astype(np.uint8), (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST
                )
                frame_processed_masks_resized.append(mask_resized)
            
            # Create segmentation overlay using processed masks
            muscle_overlay = create_segmentation_overlay(
                muscle_norm, frame_processed_masks_resized, seg_labels, segments_to_analyze,
                SEGMENT_COLORS, SEGMENT_BLEND_ORIGINAL, SEGMENT_BLEND_COLOR,
                TOP_K_HIGHLIGHT_FACTOR, top_k_masks, i
            )
            
            # Get pre-rendered trace panel
            trace_img = trace_panels[i]
            trace_img_resized = cv2.resize(trace_img, (total_width, trace_panel_height))
            
            # Compose final frame (3 panels on top: behavior, muscle raw, muscle overlay)
            top_row = np.hstack([
                behavior_overlay_resized,
                cv2.cvtColor(muscle_norm, cv2.COLOR_GRAY2RGB),
                muscle_overlay
            ])
            
            final_frame = np.vstack([top_row, trace_img_resized])
            
            # Add text labels
            final_frame_pil = Image.fromarray(final_frame)
            draw = ImageDraw.Draw(final_frame_pil)
            
            draw.text((20, 20), "Behavior", fill=(255, 255, 255))
            draw.text((top_panel_width + 20, 20), "Muscle Raw", fill=(255, 255, 255))
            draw.text((top_panel_width * 2 + 20, 20), "Muscle + ROI",
                     fill=(255, 255, 255))
            
            final_frame = np.array(final_frame_pil)
            
            # Write frame
            av_frame = av.VideoFrame.from_ndarray(final_frame, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        
        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)
    
    print(f"Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate muscle activity video from experiment data."
    )
    parser.add_argument(
        "--exp_folder",
        type=str,
        required=True,
        help="Path to experiment folder"
    )
    parser.add_argument(
        "--segments",
        type=str,
        nargs="+",
        default=["femur"],
        help="Segments to display (e.g., femur, tibia, LF, RF)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: exp_folder/muscle_activity_video.mp4)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (for profiling/testing)"
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = Path(args.exp_folder) / "muscle_activity_video.mp4"
    
    generate_video(args.exp_folder, args.segments, args.output, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
