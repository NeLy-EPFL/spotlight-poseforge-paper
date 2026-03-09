"""
Generate static figures showing muscle activity at specific time points during stimulation.

Creates:
- Raw muscle frames at stimulus onset and specified offsets
- Behavior frames with segmentation overlay
- Muscle frames with segmentation overlay (selected segments only)
- Activity trace plot showing full time course with markers at captured frames

Usage:
    python make_figures.py --exp_folder /path/to/experiment --segments femur tibia
"""

from pathlib import Path
import argparse
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import cv2
import h5py
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from spotlight_tools.postprocessing.muscle import (
    match_muscle_frameid_to_behavior_frameid,
)
from sppaper.common.muscle import (
    compute_muscle_activity,
    compute_delta_f_over_f,
)

# Import configuration
from figure_config import (
    # Preprocessing
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    # Static figure settings
    STIMULUS_INDEX,
    FRAME_OFFSETS,
    FIGURE_DPI,
    TRACE_FIGURE_WIDTH,
    TRACE_FIGURE_HEIGHT,
    ANNOTATION_FONT_SIZE,
    ANNOTATION_COLOR,
    ANNOTATION_POSITION,
    STIM_PERIOD_COLOR,
    # Shared settings
    NORM_LOWER_PERCENTILE,
    NORM_UPPER_PERCENTILE,
    TOP_K_PIXELS,
    MORPH_KERNEL_SIZE,
    MORPH_N_ITERATIONS,
    DILATION_KERNEL_SIZE,
    BASELINE_WINDOW_SEC,
    SEGMENT_BLEND_ORIGINAL,
    SEGMENT_BLEND_COLOR,
    TOP_K_HIGHLIGHT_FACTOR,
    SEGMENT_COLORS,
    TRACE_LINEWIDTH,
    TRACE_BASELINE_ALPHA,
    STIM_PERIOD_ALPHA,
)


def parse_stimulation_protocol(protocol):
    """Parse stimulation protocol string to extract onset/offset frames."""
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


def create_segmentation_overlay(muscle_frame, segmap, seg_labels, segments_to_show,
                               top_k_masks_dict=None, frame_idx=None):
    """Create RGB image with segmentation overlay (same as video script)."""
    # Convert grayscale to RGB if needed
    if len(muscle_frame.shape) == 2:
        muscle_rgb = cv2.cvtColor(muscle_frame, cv2.COLOR_GRAY2RGB).astype(np.float32)
    else:
        muscle_rgb = muscle_frame.astype(np.float32)
    
    # Create overlay - vectorized operations
    for segment_name in segments_to_show:
        if segment_name not in seg_labels:
            continue
        
        seg_id = seg_labels.index(segment_name)
        mask = (segmap == seg_id)
        
        if not mask.any():  # Skip if segment not present
            continue
        
        # Get color for this segment
        color_hex = SEGMENT_COLORS.get(segment_name, "#ffffff")
        color_rgb = np.array(mcolors.to_rgb(color_hex)) * 255
        
        # Vectorized blending with configurable weights
        mask_3d = mask[:, :, np.newaxis]
        muscle_rgb = np.where(mask_3d, 
                             SEGMENT_BLEND_ORIGINAL * muscle_rgb + SEGMENT_BLEND_COLOR * color_rgb,
                             muscle_rgb)
        
        # Highlight top k pixels in darker shade (using precomputed masks)
        if top_k_masks_dict is not None and frame_idx is not None:
            top_k_mask = top_k_masks_dict.get((frame_idx, segment_name))
            if top_k_mask is not None and top_k_mask.any():
                dark_color = color_rgb * TOP_K_HIGHLIGHT_FACTOR
                # Apply dark color to top k pixels
                top_k_mask_3d = top_k_mask[:, :, np.newaxis]
                muscle_rgb = np.where(top_k_mask_3d, dark_color, muscle_rgb)
    
    return muscle_rgb.astype(np.uint8)


def annotate_image(img, text, font_size=ANNOTATION_FONT_SIZE, 
                   position=ANNOTATION_POSITION, color=ANNOTATION_COLOR):
    """Add text annotation to image."""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()
    
    draw.text(position, text, fill=color, font=font)
    return np.array(img_pil)


def create_trace_figure(time_sec, traces, segment_names, stim_start_times, stim_end_times,
                       capture_times, capture_labels):
    """Create trace plot showing full time course with markers."""
    fig, ax = plt.subplots(figsize=(TRACE_FIGURE_WIDTH, TRACE_FIGURE_HEIGHT), dpi=FIGURE_DPI)
    ax.set_facecolor('white')
    
    n_segments = len(segment_names)
    offset_spacing = traces.max()
    
    # Plot traces
    for j, segment_name in enumerate(segment_names):
        offset = j * offset_spacing
        color = SEGMENT_COLORS.get(segment_name, "#000000")
        ax.plot(time_sec, traces[:, j] + offset, color=color, linewidth=TRACE_LINEWIDTH)
        ax.axhline(y=offset, color='gray', linestyle='--', alpha=TRACE_BASELINE_ALPHA, linewidth=0.5)
    
    # Mark stimulation periods (light gray instead of yellow)
    for start_time, end_time in zip(stim_start_times, stim_end_times):
        ax.axvspan(start_time, end_time, alpha=0.3, color=STIM_PERIOD_COLOR, zorder=0)
    
    # Mark captured time points with asterisks
    y_max = (n_segments - 1) * offset_spacing + traces.max() * 0.1
    for i, (capture_time, label) in enumerate(zip(capture_times, capture_labels)):
        # Add asterisk marker at the top
        #ax.plot(capture_time, y_max, marker='*', color='black', markersize=8, zorder=10)
        ax.text(capture_time, y_max, '*', color='black', fontsize=12, ha='center', va='bottom', zorder=10)
    
    # Styling
    ytick_positions = [j * offset_spacing for j in range(n_segments)]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(segment_names, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Muscle Segment", fontsize=11)
    ax.set_title(r"Muscle Activity ($\Delta$F/F$_0$)", fontsize=12, pad=10)
    ax.grid(axis='x', alpha=0.2, linewidth=0.5)
    ax.set_xlim(time_sec[0], time_sec[-1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def generate_figures(exp_folder, segments_to_show, output_folder=None):
    """Generate static figures for the experiment."""
    exp_folder = Path(exp_folder)
    processed_folder = exp_folder / "processed"
    metadata_folder = exp_folder / "metadata"
    
    if output_folder is None:
        output_folder = exp_folder / "figures"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving figures to: {output_folder}")
    
    # Load metadata
    print("Loading metadata...")
    import pandas as pd
    mf_metadata = pd.read_csv(processed_folder / "muscle_frames_metadata.csv")
    bf_metadata = pd.read_csv(processed_folder / "behavior_frames_metadata.csv")
    duo_yaml = metadata_folder / "dual_recording_timing.yaml"
    
    # Load experiment parameters to get stimulation times
    exp_parameters = yaml.safe_load(
        (metadata_folder / "experiment_parameters.yaml").read_text()
    )
    experiment_protocol = exp_parameters["experiment_protocol"]
    
    # Parse stimulation protocol (same as video script)
    stim_starts, stim_ends = parse_stimulation_protocol(experiment_protocol)
    
    # Use first channel with stimulations
    channel = list(stim_starts.keys())[0]
    stim_start_frames = stim_starts[channel]
    stim_end_frames = stim_ends[channel]
    
    # Select stimulus
    if STIMULUS_INDEX >= len(stim_start_frames):
        raise ValueError(f"Stimulus index {STIMULUS_INDEX} out of range. Only {len(stim_start_frames)} stimuli found.")
    
    stim_frame = stim_start_frames[STIMULUS_INDEX]
    print(f"Using stimulus {STIMULUS_INDEX} at behavior frame {stim_frame}")
    
    # Load segmentation data
    print("Loading segmentation data...")
    segmap_folder = exp_folder.parent / "poseforge_output/bodyseg/epoch14_step12000" / \
                   (exp_folder.name + "_model_prediction_not_flipped")
    with h5py.File(segmap_folder / "bodyseg_pred.h5", "r") as f:
        segmaps = f["pred_segmap"][:]
        frame_ids = f["frame_ids"][:]
        seg_labels = list(f["pred_segmap"].attrs["class_labels"])
    
    # Filter segments
    segments_to_analyze = [seg for seg in seg_labels if any(
        s.lower() in seg.lower() for s in segments_to_show
    )]
    print(f"Analyzing segments: {segments_to_analyze}")
    
    # Compute muscle activity
    print("Computing muscle activity...")
    muscle_activity, top_k_masks = compute_muscle_activity(
        mf_metadata, processed_folder, segmaps, frame_ids,
        seg_labels, segments_to_analyze, duo_yaml, k=TOP_K_PIXELS,
        morph_kernel_size=MORPH_KERNEL_SIZE,
        morph_n_iterations=MORPH_N_ITERATIONS,
        dilation_kernel_size=DILATION_KERNEL_SIZE,
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
        start_time = bf_metadata[bf_metadata["behavior_frame_id"] == start_frame]["received_time_us"].values
        end_time = bf_metadata[bf_metadata["behavior_frame_id"] == end_frame]["received_time_us"].values
        if len(start_time) > 0 and len(end_time) > 0:
            stim_start_times.append((start_time[0] - bf_metadata["received_time_us"].iloc[0]) / 1e6)
            stim_end_times.append((end_time[0] - bf_metadata["received_time_us"].iloc[0]) / 1e6)
    
    # Compute ΔF/F₀
    print("Computing ΔF/F₀...")
    delta_f_over_f = compute_delta_f_over_f(
        muscle_activity, time_sec.values, stim_start_times, stim_end_times,
        baseline_window_sec=BASELINE_WINDOW_SEC
    )
    
    # Load all muscle frames to compute global normalization percentiles AND cache them
    print("Loading and caching all muscle frames...")
    cached_frames = {}
    all_pixel_values = []
    for mf_id in tqdm(mf_metadata["muscle_frame_id"].astype(int), desc="Loading frames"):
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
    vmax = np.percentile(all_frames_sample, NORM_UPPER_PERCENTILE)
    print(f"Normalization range: {vmin:.1f} to {vmax:.1f}")
    
    del all_pixel_values  # Free memory
    del all_frames_sample
    
    # Find muscle frame indices corresponding to stimulus + offsets
    stim_time = stim_start_times[STIMULUS_INDEX]
    
    # Find closest muscle frame to stimulus onset
    time_diffs = np.abs(time_sec.values - stim_time)
    stim_muscle_idx = np.argmin(time_diffs)
    
    print(f"Stimulus time: {stim_time:.3f}s, closest muscle frame index: {stim_muscle_idx}")
    
    # Generate images at each offset
    capture_times = []
    capture_labels = []
 
    for offset in FRAME_OFFSETS:
        frame_idx = stim_muscle_idx + offset
        if frame_idx >= len(mf_metadata):
            print(f"Warning: Frame offset {offset} exceeds data length, skipping")
            continue
        
        mf_id = int(mf_metadata.iloc[frame_idx]["muscle_frame_id"])
        frame_time = time_sec.iloc[frame_idx]
        time_offset = frame_time - stim_time
        
        capture_times.append(frame_time)
        capture_labels.append(f"+{time_offset:.3f}s" if time_offset >= 0 else f"{time_offset:.3f}s")
        
        print(f"Processing frame {frame_idx}, offset={offset}, time={frame_time:.3f}s (stim+{time_offset:.3f}s)")
        
        # Use cached muscle frame (already loaded)
        muscle_frame = cached_frames[mf_id].astype(float)
        
        # Normalize frame using pre-computed percentiles
        muscle_norm = np.clip((muscle_frame - vmin) / (vmax - vmin + 1e-8), 0, 1)
        muscle_norm = (muscle_norm * 255).astype(np.uint8)
        
        # Save raw muscle frame
        raw_annotated = annotate_image(
            cv2.cvtColor(muscle_norm, cv2.COLOR_GRAY2RGB),
            f"Stim {time_offset:+.3f}s"
        )
        cv2.imwrite(str(output_folder / f"muscle_raw_offset{offset:03d}.png"), 
                   cv2.cvtColor(raw_annotated, cv2.COLOR_RGB2BGR))
        
        # Get corresponding segmap (same as video script)
        bf_frame_id = match_muscle_frameid_to_behavior_frameid(
            mf_id, dual_recording_timing_metadata_path=duo_yaml
        )
        segmap = segmaps[frame_ids == bf_frame_id][0]
        
        # Resize segmap to match muscle frame
        frame_height, frame_width = muscle_norm.shape
        segmap_resized = cv2.resize(
            segmap, (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Create muscle overlay (using precomputed top-k masks)
        muscle_overlay = create_segmentation_overlay(
            muscle_norm, segmap_resized, seg_labels, segments_to_analyze,
            top_k_masks_dict=top_k_masks, frame_idx=frame_idx
        )
        muscle_overlay_annotated = annotate_image(muscle_overlay, f"Stim {time_offset:+.3f}s")
        cv2.imwrite(str(output_folder / f"muscle_overlay_offset{offset:03d}.png"),
                   cv2.cvtColor(muscle_overlay_annotated, cv2.COLOR_RGB2BGR))
        
        # Load and process behavior frame for overlay
        beh_frames_folder = exp_folder.parent / "spotlight_aligned_and_cropped" / exp_folder.name / "all"
        behavior_frame_path = beh_frames_folder / f"frame_{bf_frame_id:09d}.jpg"
        
        if not behavior_frame_path.exists():
            print(f"  Warning: Behavior frame {behavior_frame_path} not found, skipping behavior overlay")
        else:
            behavior_frame = cv2.imread(str(behavior_frame_path), cv2.IMREAD_UNCHANGED)
            
            if behavior_frame is None:
                print(f"  Warning: Could not load behavior frame {behavior_frame_path}, skipping")
            else:
                # Resize segmap to match behavior frame
                behavior_height, behavior_width = behavior_frame.shape[:2] if len(behavior_frame.shape) == 3 else behavior_frame.shape
                segmap_behavior = cv2.resize(
                    segmap, (behavior_width, behavior_height),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Convert to grayscale if needed
                if len(behavior_frame.shape) == 3:
                    behavior_gray = cv2.cvtColor(behavior_frame, cv2.COLOR_BGR2GRAY)
                else:
                    behavior_gray = behavior_frame
                
                # Create behavior overlay (no top-k highlighting)
                behavior_overlay = create_segmentation_overlay(
                    behavior_gray, segmap_behavior, seg_labels, segments_to_analyze,
                    top_k_masks_dict=None, frame_idx=None
                )
                behavior_overlay_annotated = annotate_image(behavior_overlay, f"Stim {time_offset:+.3f}s")
                cv2.imwrite(str(output_folder / f"behavior_overlay_offset{offset:03d}.png"),
                           cv2.cvtColor(behavior_overlay_annotated, cv2.COLOR_RGB2BGR))
        
        print(f"  Saved images for offset {offset}")
    
    # Create trace figure
    print("Creating trace figure...")
    fig = create_trace_figure(
        time_sec.values, delta_f_over_f, segments_to_analyze,
        stim_start_times, stim_end_times,
        capture_times, capture_labels
    )
    fig.savefig(output_folder / "muscle_activity_traces.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nAll figures saved to: {output_folder}")
    print(f"Generated {len(FRAME_OFFSETS) * 3} image files + 1 trace plot")


def main():
    parser = argparse.ArgumentParser(
        description="Generate static figures showing muscle activity at specific time points."
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
        "--output_folder",
        type=str,
        default=None,
        help="Output folder for figures (default: exp_folder/figures)"
    )
    
    args = parser.parse_args()
    generate_figures(args.exp_folder, args.segments, args.output_folder)


if __name__ == "__main__":
    main()
