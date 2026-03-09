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
    create_segmentation_overlay,
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
    VMAX_SHIFT,
    TOP_K_PIXELS,
    MORPH_KERNEL_SIZE,
    MORPH_N_ITERATIONS,
    MIN_FRAGMENT_SIZE,
    MAX_FRAGMENT_DISTANCE,
    DILATION_KERNELS,
    BASELINE_WINDOW_SEC,
    SEGMENT_BLEND_ORIGINAL,
    SEGMENT_BLEND_COLOR,
    TOP_K_HIGHLIGHT_FACTOR,
    SEGMENT_ORDER,
    SEGMENT_COLORS,
    TRACE_LINEWIDTH,
    TRACE_BASELINE_ALPHA,
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
    # Use nanmax to handle NaN values
    traces_max = np.nanmax(traces)
    if np.isnan(traces_max):
        traces_max = 1.0  # Fallback if all values are NaN
    offset_spacing = traces_max
    
    # Plot traces (matplotlib will skip NaN values automatically)
    for j, segment_name in enumerate(segment_names):
        offset = j * offset_spacing
        color = SEGMENT_COLORS.get(segment_name, "#000000")
        ax.plot(time_sec, traces[:, j] + offset, color=color, linewidth=TRACE_LINEWIDTH)
        ax.axhline(y=offset, color='gray', linestyle='--', alpha=TRACE_BASELINE_ALPHA, linewidth=0.5)
    
    # Mark stimulation periods (light gray instead of yellow)
    for start_time, end_time in zip(stim_start_times, stim_end_times):
        ax.axvspan(start_time, end_time, alpha=0.3, color=STIM_PERIOD_COLOR, zorder=0)
    
    # Mark captured time points with asterisks
    y_max = (n_segments - 1) * offset_spacing + traces_max * 0.1
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
    
    # Sort segments according to canonical order (RF, LF, RM, LM, RH, LH)
    segments_to_analyze = sorted(segments_to_analyze, 
                                 key=lambda s: SEGMENT_ORDER.index(s) if s in SEGMENT_ORDER else 999)
    
    print(f"Analyzing segments: {segments_to_analyze}")
    
    # Compute muscle activity for ALL frames (we want to see the full trace)
    print("Computing muscle activity...")
    muscle_activity, top_k_masks, processed_masks_dict = compute_muscle_activity(
        mf_metadata, processed_folder, segmaps, frame_ids,
        seg_labels, segments_to_analyze, duo_yaml, k=TOP_K_PIXELS,
        morph_kernel_size=MORPH_KERNEL_SIZE,
        morph_n_iterations=MORPH_N_ITERATIONS,
        dilation_kernels=DILATION_KERNELS,
        bilateral_d=BILATERAL_D,
        bilateral_sigma_color=BILATERAL_SIGMA_COLOR,
        bilateral_sigma_space=BILATERAL_SIGMA_SPACE,
        min_fragment_size=MIN_FRAGMENT_SIZE,
        max_fragment_distance=MAX_FRAGMENT_DISTANCE
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
    vmax = np.percentile(all_frames_sample, NORM_UPPER_PERCENTILE) + VMAX_SHIFT
    print(f"Normalization range: {vmin:.1f} to {vmax:.1f} (with VMAX_SHIFT={VMAX_SHIFT})")
    
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
        
        # Get processed masks for this frame
        frame_processed_masks = processed_masks_dict[frame_idx]
        
        # Resize processed masks to muscle frame dimensions
        frame_height, frame_width = muscle_norm.shape
        frame_processed_masks_resized = []
        for mask in frame_processed_masks:
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST
            )
            frame_processed_masks_resized.append(mask_resized)
        
        # Create muscle overlay using processed masks
        muscle_overlay = create_segmentation_overlay(
            muscle_norm, frame_processed_masks_resized, seg_labels, segments_to_analyze,
            SEGMENT_COLORS, SEGMENT_BLEND_ORIGINAL, SEGMENT_BLEND_COLOR,
            TOP_K_HIGHLIGHT_FACTOR, top_k_masks, frame_idx
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
                # Get original segmap for behavior overlay (not dilated)
                segmap = segmaps[frame_ids == bf_frame_id][0]
                behavior_height, behavior_width = behavior_frame.shape[:2] if len(behavior_frame.shape) == 3 else behavior_frame.shape
                
                # Convert original segmap to list of masks (one per segment)
                original_masks_behavior = []
                for seg_id in range(len(seg_labels)):
                    mask = (segmap == seg_id).astype(np.uint8)
                    mask_resized = cv2.resize(
                        mask, (behavior_width, behavior_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                    original_masks_behavior.append(mask_resized)
                
                # Convert to grayscale if needed
                if len(behavior_frame.shape) == 3:
                    behavior_gray = cv2.cvtColor(behavior_frame, cv2.COLOR_BGR2GRAY)
                else:
                    behavior_gray = behavior_frame
                
                # Create behavior overlay using original masks (no top-k highlighting)
                behavior_overlay = create_segmentation_overlay(
                    behavior_gray, original_masks_behavior, seg_labels, segments_to_analyze,
                    SEGMENT_COLORS, SEGMENT_BLEND_ORIGINAL, SEGMENT_BLEND_COLOR,
                    TOP_K_HIGHLIGHT_FACTOR, None, None
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
    fig.savefig(output_folder / "muscle_activity_traces.pdf", bbox_inches='tight')
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
