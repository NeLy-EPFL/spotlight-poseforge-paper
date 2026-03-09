"""
Analyze all stimulations across multiple experiments and create summary figures.

For each stimulation:
- Checks if stage moved less than 1mm during a window around stimulation
- Extracts muscle activity traces in a 1s window
- Aggregates across trials or flies based on aggregation rule
- Plots individual traces and mean across all valid trials
- Saves justification for inclusion/exclusion of each trial

Usage:
    python analyze_all_stimulations.py --data_folder /path/to/data --aggregate fly --segments femur
"""

from pathlib import Path
import argparse
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import cv2
import h5py
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from spotlight_tools.postprocessing.muscle import (
    match_muscle_frameid_to_behavior_frameid,
    match_behavior_frameid_to_muscle_frameid,
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
    # Analysis settings
    NORM_LOWER_PERCENTILE,
    NORM_UPPER_PERCENTILE,
    TOP_K_PIXELS,
    MORPH_KERNEL_SIZE,
    MORPH_N_ITERATIONS,
    DILATION_KERNEL_SIZE,
    BASELINE_WINDOW_SEC,
    # Colors
    SEGMENT_COLORS,
    TRACE_LINEWIDTH,
    TRACE_BASELINE_ALPHA,
    STIM_PERIOD_ALPHA,
    STIM_PERIOD_COLOR,
    FIGURE_DPI,
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


def check_stage_movement(bf_metadata, start_frame, end_frame, max_distance_mm=1.0):
    """Check if stage moved more than max_distance_mm during stimulation.
    
    Returns:
        moved: bool, True if stage moved too much
        distance: float, integrated distance traveled
        reason: str, description of movement
    """
    stim_frames = bf_metadata[
        (bf_metadata["behavior_frame_id"] >= start_frame) & 
        (bf_metadata["behavior_frame_id"] <= end_frame)
    ]
    
    if len(stim_frames) < 2:
        return True, 0.0, "Not enough frames to assess movement"
    
    # Compute integrated distance (sum of euclidean distances between consecutive frames)
    x_pos = stim_frames["x_pos_mm_interp"].values
    y_pos = stim_frames["y_pos_mm_interp"].values
    
    dx = np.diff(x_pos)
    dy = np.diff(y_pos)
    distances = np.sqrt(dx**2 + dy**2)
    total_distance = np.sum(distances)
    
    moved_too_much = total_distance > max_distance_mm
    
    if moved_too_much:
        reason = f"Stage moved {total_distance:.3f}mm (>{max_distance_mm}mm threshold)"
    else:
        reason = f"Stage moved {total_distance:.3f}mm (<={max_distance_mm}mm threshold)"
    
    return moved_too_much, total_distance, reason


def extract_window_around_stimulation(muscle_activity, time_sec, stim_time, window_sec=1.0):
    """Extract a time window of muscle activity around stimulation onset.
    
    Returns:
        window_activity: (n_frames_window, n_segments) array
        window_time: time relative to stimulation onset
        indices: original indices in full trace
    """
    # Find indices within window
    time_diff = time_sec - stim_time
    window_mask = np.abs(time_diff) <= window_sec
    indices = np.where(window_mask)[0]
    
    if len(indices) == 0:
        return None, None, None
    
    window_activity = muscle_activity[indices, :]
    window_time = time_diff[indices]
    
    return window_activity, window_time, indices


def analyze_experiment(exp_folder, segments_to_show, output_folder=None, 
                      max_movement_mm=1.0, window_sec=1.0):
    """Analyze all stimulations in an experiment."""
    exp_folder = Path(exp_folder)
    processed_folder = exp_folder / "processed"
    metadata_folder = exp_folder / "metadata"
    
    if output_folder is None:
        output_folder = exp_folder / "stimulation_analysis"
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing experiment: {exp_folder.name}")
    print(f"Saving results to: {output_folder}")
    
    # Load metadata
    print("Loading metadata...")
    bf_metadata = pd.read_csv(processed_folder / "behavior_frames_metadata.csv")
    mf_metadata = pd.read_csv(processed_folder / "muscle_frames_metadata.csv")
    duo_yaml = metadata_folder / "dual_recording_timing.yaml"
    
    # Load experiment parameters
    exp_parameters = yaml.safe_load(
        (metadata_folder / "experiment_parameters.yaml").read_text()
    )
    experiment_protocol = exp_parameters["experiment_protocol"]
    
    # Parse stimulation protocol
    stim_starts, stim_ends = parse_stimulation_protocol(experiment_protocol)
    
    # Use first channel with stimulations
    channel = list(stim_starts.keys())[0]
    stim_start_frames = stim_starts[channel]
    stim_end_frames = stim_ends[channel]
    
    print(f"Found {len(stim_start_frames)} stimulations on channel {channel}")
    
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
    
    # Compute muscle activity for all frames
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
    
    # Analyze each stimulation
    print("\nAnalyzing individual stimulations...")
    valid_stimulations = []
    justifications = []
    
    for i, (start_frame, end_frame, stim_time) in enumerate(
        zip(stim_start_frames, stim_end_frames, stim_start_times)
    ):
        # Check stage movement
        moved_too_much, distance, reason = check_stage_movement(
            bf_metadata, start_frame, end_frame, max_movement_mm
        )
        
        # Extend window for movement check
        window_start_frame = start_frame - int(window_sec * 30)  # Assuming ~30 fps
        window_end_frame = end_frame + int(window_sec * 30)
        moved_in_window, window_distance, window_reason = check_stage_movement(
            bf_metadata, window_start_frame, window_end_frame, max_movement_mm
        )
        
        included = not moved_in_window
        if included:
            justification = f"Stim {i+1}: INCLUDED, {window_distance:.3f}mm\n"
        else:
            justification = f"Stim {i+1}: EXCLUDED, {window_distance:.3f}mm
        justification += f"  In {window_sec}s window: {window_reason}\n"
        
        justifications.append(justification)
        
        if included:
            # Extract window around this stimulation
            window_activity, window_time, indices = extract_window_around_stimulation(
                delta_f_over_f, time_sec.values, stim_time, window_sec
            )
            
            if window_activity is not None:
                valid_stimulations.append({
                    'exp_name': exp_folder.name,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'stim_time': stim_time,
                    'stim_duration': stim_end_times[i] - stim_start_times[i] if i < len(stim_end_times) else 0,
                    'window_activity': window_activity,
                    'window_time': window_time,
                    'distance': window_distance,
                })
            
        justifications.append("")  # Empty line between stimulations
    
    return valid_stimulations, justifications, segments_to_analyze


def find_experiment_folders(data_folder):
    """Find all experiment folders in data_folder."""
    data_folder = Path(data_folder)
    exp_folders = []
    
    # Look for folders with "processed" subfolder
    for item in data_folder.iterdir():
        if item.is_dir():
            if (item / "processed").exists():
                exp_folders.append(item)
    
    return sorted(exp_folders)


def aggregate_and_plot(all_valid_stims, segments_to_analyze, output_folder, 
                      window_sec, aggregate_by='fly'):
    """Create aggregated summary figures."""
    if len(all_valid_stims) == 0:
        print("WARNING: No valid stimulations found!")
        return
    
    # Group by experiment or fly
    if aggregate_by == 'trial':
        # Group by individual experiment
        groups = {}
        for stim in all_valid_stims:
            exp_name = stim['exp_name']
            if exp_name not in groups:
                groups[exp_name] = []
            groups[exp_name].append(stim)
    else:  # aggregate_by == 'fly'
        # Aggregate all
        groups = {'all_flies': all_valid_stims}
    
    # Create figure for each group
    for group_name, group_stims in groups.items():
        print(f"\nCreating figure for {group_name} with {len(group_stims)} stimulations...")
        
        n_segments = len(segments_to_analyze)
        # More square aspect ratio
        fig_height = 3 * n_segments
        fig_width = 6
        fig, axes = plt.subplots(n_segments, 1, figsize=(fig_width, fig_height), 
                                sharex=True, squeeze=False)
        
        # Get average stim duration
        avg_stim_dur = np.mean([s['stim_duration'] for s in group_stims])
        
        for seg_idx, segment_name in enumerate(segments_to_analyze):
            ax = axes[seg_idx, 0]
            color = SEGMENT_COLORS.get(segment_name, "#000000")
            
            # Plot individual traces
            common_time = np.linspace(-window_sec, window_sec, 100)
            interpolated_traces = []
            
            for stim in group_stims:
                window_activity = stim['window_activity']
                window_time = stim['window_time']
                trace = window_activity[:, seg_idx]
                
                # Sort by time
                sort_idx = np.argsort(window_time)
                window_time_sorted = window_time[sort_idx]
                trace_sorted = trace[sort_idx]
                
                # Interpolate to common time base
                interp_trace = np.interp(common_time, window_time_sorted, trace_sorted)
                interpolated_traces.append(interp_trace)
                
                # Plot individual trace
                ax.plot(common_time, interp_trace, color=color, alpha=0.2, linewidth=0.6)
            
            # Compute and plot mean trace
            mean_trace = np.mean(interpolated_traces, axis=0)
            sem_trace = np.std(interpolated_traces, axis=0) / np.sqrt(len(interpolated_traces))
            
            # Plot mean ± SEM
            ax.plot(common_time, mean_trace, color=color, linewidth=2.5)
            ax.fill_between(common_time, mean_trace - sem_trace, mean_trace + sem_trace,
                            color=color, alpha=0.25)
            
            # Mark stimulation period
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
            ax.axvspan(0, avg_stim_dur, alpha=0.3, color=STIM_PERIOD_COLOR, zorder=0)
            
            # Styling
            ax.set_ylabel(segment_name, fontsize=10, color=color, fontweight='bold')
            ax.grid(axis='both', alpha=0.2, linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if seg_idx == 0:
                title = f"n={len(group_stims)} trials"
                if aggregate_by == 'trial':
                    title = f"{group_name}: {title}"
                ax.set_title(title, fontsize=11, pad=10)
        
        axes[-1, 0].set_xlabel("Time relative to stimulation (s)", fontsize=10)
        fig.text(0.04, 0.5, r"$\Delta$F/F$_0$", va='center', rotation='vertical', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        safe_name = group_name.replace('/', '_')
        fig_path = output_folder / f"{safe_name}_{len(group_stims)}trials.png"
        fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved figure to: {fig_path}")
across experiments and create summary figures."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path to data folder containing experiment subfolders"
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=['fly', 'trial'],
        default='fly',
        help="Aggregation rule: 'fly' (all flies together) or 'trial' (separate per experiment)"
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
        help="Output folder for results (default: data_folder/muscle_analysis_<aggregate>)"
    )
    parser.add_argument(
        "--max_movement",
        type=float,
        default=1.0,
        help="Maximum allowed stage movement in mm (default: 1.0)"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=1.0,
        help="Time window around stimulation in seconds (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    analyze_all_experiments(
        args.data_folder,
        args.segments,
        args.output_folder,
        args.max_movement,
        args.window,
        args.aggregate
            all_valid_stims.extend(valid_stims)
            all_justifications.append(f"\n{exp_folder.name}:\n")
            all_justifications.extend(justifications)
            
            if segments_to_analyze is None:
                segments_to_analyze = segments
                
        except Exception as e:
            print(f"ERROR processing {exp_folder.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined justifications
    justification_file = output_folder / "all_stimulations_summary.txt"
    with open(justification_file, 'w') as f:
        f.write(f"Data folder: {data_folder}\n")
        f.write(f"Aggregation: {aggregate_by}\n")
        f.write(f"Max movement threshold: {max_movement_mm}mm\n")
        f.write(f"Analysis window: ±{window_sec}s\n")
        f.write(f"Total experiments: {len(exp_folders)}\n")
        f.write(f"Total valid stimulations: {len(all_valid_stims)}\n")
        f.write(f"Segments: {', '.join(segments_to_analyze)}\n\n")
        f.write("="*80 + "\n")
        f.writelines(all_justifications)
    
    print(f"\n\nSaved combined justifications to: {justification_file}")
    
    # Save CSV with all valid stimulations
    csv_path = output_folder / "all_valid_stimulations.csv"
    stim_data = []
    for stim in all_valid_stims:
        stim_data.append({
            'experiment': stim['exp_name'],
            'stimulation_index': stim['index'],
            'start_frame': stim['start_frame'],
            'end_frame': stim['end_frame'],
            'stim_time_sec': stim['stim_time'],
            'stage_movement_mm': stim['distance'],
        })
    pd.DataFrame(stim_data).to_csv(csv_path, index=False)
    print(f"Saved all stimulation data to: {csv_path}")
    
    # Create aggregated figures
    aggregate_and_plot(all_valid_stims, segments_to_analyze, output_folder,
                      window_sec, aggregate_by)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Total valid stimulations: {len(all_valid_stims)}")
    print(f"Results saved to: {output_folder}")
    print("="*80
    print("\nAnalysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze all stimulations in an experiment and create summary figures."
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
        help="Output folder for results (default: exp_folder/stimulation_analysis)"
    )
    parser.add_argument(
        "--max_movement",
        type=float,
        default=1.0,
        help="Maximum allowed stage movement in mm (default: 1.0)"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=1.0,
        help="Time window around stimulation in seconds (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    analyze_experiment(
        args.exp_folder, 
        args.segments, 
        args.output_folder,
        args.max_movement,
        args.window
    )


if __name__ == "__main__":
    main()
