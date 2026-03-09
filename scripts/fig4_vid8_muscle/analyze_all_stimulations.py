"""
Analyze all stimulations across multiple experiments and create summary figures.

For each stimulation:
- Checks if stage moved less than threshold during analysis window
- Analysis window: [stim_start - window_sec, stim_end + window_sec]
- Extracts muscle activity traces in the analysis window
- Aggregates across trials or flies based on aggregation rule
- Plots individual traces and mean across all valid trials
- Saves justification for inclusion/exclusion of each trial

Usage:
    python analyze_all_stimulations.py --data_folder /path/to/data --aggregate fly --segments femur --window 0.5
"""

from pathlib import Path
import argparse
import sys
import glob
import zipfile

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
    get_behavior_muscle_sync_ratio,
)
from sppaper.common.muscle import (
    compute_muscle_activity_for_frames,
    compute_delta_f_over_f,
)

# Import configuration
from figure_config import (
    # Preprocessing
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    # Analysis settings
    TOP_K_PIXELS,
    MORPH_KERNEL_SIZE,
    MORPH_N_ITERATIONS,
    MIN_FRAGMENT_SIZE,
    MAX_FRAGMENT_DISTANCE,
    DILATION_KERNELS,
    BASELINE_WINDOW_SEC,
    # Colors
    SEGMENT_ORDER,
    SEGMENT_COLORS,
    STIM_PERIOD_COLOR,
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
    """Check if stage moved more than max_distance_mm during specified window.
    
    Args:
        bf_metadata: Behavior frames metadata
        start_frame: Start behavior frame ID
        end_frame: End behavior frame ID
        max_distance_mm: Maximum allowed movement
    
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


def ensure_metadata_unzipped(exp_folder):
    """Unzip metadata.zip if it exists and metadata folder is empty or missing files."""
    exp_folder = Path(exp_folder)
    metadata_folder = exp_folder / "metadata"
    metadata_zip = exp_folder / "metadata.zip"
    
    # Check if we need to unzip
    needs_unzip = False
    if metadata_zip.exists():
        if not metadata_folder.exists():
            needs_unzip = True
        elif not (metadata_folder / "experiment_parameters.yaml").exists():
            needs_unzip = True
    
    if needs_unzip:
        print(f"  Unzipping metadata for {exp_folder.name}...")
        with zipfile.ZipFile(metadata_zip, 'r') as zip_ref:
            zip_ref.extractall(exp_folder)
        print(f"  Metadata extracted successfully")


def load_experiment_data(exp_folder, segments_to_show):
    """Load all necessary data for an experiment.
    
    Returns:
        Dictionary with all loaded data
    """
    exp_folder = Path(exp_folder)
    
    # Ensure metadata is unzipped if necessary
    ensure_metadata_unzipped(exp_folder)
    
    processed_folder = exp_folder / "processed"
    metadata_folder = exp_folder / "metadata"
    
    # Load metadata
    bf_metadata = pd.read_csv(processed_folder / "behavior_frames_metadata.csv")
    mf_metadata = pd.read_csv(processed_folder / "muscle_frames_metadata.csv")
    duo_yaml = metadata_folder / "dual_recording_timing.yaml"
    
    # Load experiment parameters
    exp_parameters = yaml.safe_load(
        (metadata_folder / "experiment_parameters.yaml").read_text()
    )
    experiment_protocol = exp_parameters["experiment_protocol"]

    # Get the fps
    behavior_fps = exp_parameters["behavior_fps"]
    muscle_fps = behavior_fps / exp_parameters["muscle_sync_ratio"]
    
    # Parse stimulation protocol
    stim_starts, stim_ends = parse_stimulation_protocol(experiment_protocol)
    
    # Load segmentation data
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
    
    return {
        'exp_folder': exp_folder,
        'processed_folder': processed_folder,
        'metadata_folder': metadata_folder,
        'bf_metadata': bf_metadata,
        'mf_metadata': mf_metadata,
        'duo_yaml': duo_yaml,
        'behavior_fps': behavior_fps,
        'muscle_fps': muscle_fps,
        'stim_starts': stim_starts,
        'stim_ends': stim_ends,
        'segmaps': segmaps,
        'frame_ids': frame_ids,
        'seg_labels': seg_labels,
        'segments_to_analyze': segments_to_analyze,
    }


def extract_window_around_stimulation(muscle_activity, mf_metadata, time_sec,
                                       stim_start_time, 
                                       window_start_mf, window_end_mf):
    """Extract a window of muscle activity around stimulation using muscle frame indices.
    
    Args:
        muscle_activity: Full muscle activity array (n_muscle_frames, n_segments)
        mf_metadata: Muscle frames metadata
        time_sec: Time axis for all muscle frames (relative to recording start)
        stim_start_time: Stimulation start time in seconds (for relative time computation)
        window_start_mf: Window start in muscle frame ID
        window_end_mf: Window end in muscle frame ID
    
    Returns:
        window_activity: (n_frames_window, n_segments) array
        window_time: time relative to stimulation onset (in seconds)
        window_mf_ids: muscle frame IDs in the window
    """
    # Find muscle frames within window
    window_mask = (
        (mf_metadata["muscle_frame_id"] >= window_start_mf) & 
        (mf_metadata["muscle_frame_id"] <= window_end_mf)
    )
    window_mf_ids = mf_metadata.loc[window_mask, "muscle_frame_id"].values
    
    if len(window_mf_ids) == 0:
        return None, None, None
    
    # Get corresponding indices in the muscle_activity array
    indices = mf_metadata.loc[window_mask].index.values
    
    window_activity = muscle_activity[indices, :]
    
    # Compute time relative to stim start
    window_time = time_sec[indices] - stim_start_time
    
    return window_activity, window_time, window_mf_ids


def analyze_experiment(exp_folder, segments_to_show, output_folder,
                      max_movement_mm, window_sec):
    """Analyze all stimulations in an experiment."""

    exp_folder = Path(exp_folder)
    
    # Only create output folder if explicitly provided (not when called from analyze_all_experiments)
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Analyzing experiment: {exp_folder.name}")
        print(f"Saving results to: {output_folder}")
    else:
        print(f"Analyzing experiment: {exp_folder.name}")
    
    # Load all experiment data using common function
    print("Loading experiment data...")
    data = load_experiment_data(exp_folder, segments_to_show)
    
    bf_metadata = data['bf_metadata']
    mf_metadata = data['mf_metadata']
    duo_yaml = data['duo_yaml']
    muscle_fps = data['muscle_fps']
    behavior_fps = data['behavior_fps']
    stim_starts = data['stim_starts']
    stim_ends = data['stim_ends']
    segmaps = data['segmaps']
    frame_ids = data['frame_ids']
    seg_labels = data['seg_labels']
    segments_to_analyze = data['segments_to_analyze']
    processed_folder = data['processed_folder']
    
    print(f"Muscle imaging FPS: {muscle_fps:.2f} Hz")
    
    # Use first channel with stimulations
    channel = list(stim_starts.keys())[0]
    stim_start_frames = stim_starts[channel]
    stim_end_frames = stim_ends[channel]
    
    print(f"Found {len(stim_start_frames)} stimulations on channel {channel}")
    print(f"Analyzing segments: {segments_to_analyze}")
    
    # Convert window from seconds to behavior frames
    analysis_window_frames  = int(BASELINE_WINDOW_SEC * behavior_fps)
    display_window_frames = int(window_sec * behavior_fps)
    muscle_activity_window_frames = max(analysis_window_frames, display_window_frames)

    # Find all muscle frame indices needed (union of windows around all stimulations)
    muscle_frame_indices_set = set()
    
    for start_frame, end_frame in zip(stim_start_frames, stim_end_frames):
        muscle_frame_start = match_behavior_frameid_to_muscle_frameid(
            start_frame - muscle_activity_window_frames, method="floor", dual_recording_timing_metadata_path=duo_yaml)
        muscle_frame_end = match_behavior_frameid_to_muscle_frameid(
            end_frame + muscle_activity_window_frames, method="floor", dual_recording_timing_metadata_path=duo_yaml) + 1
        muscle_frame_indices_set.update(range(muscle_frame_start, muscle_frame_end))

    # Convert to sorted list
    frame_indices_list = sorted(list(muscle_frame_indices_set))
    print(f"Computing muscle activity for {len(frame_indices_list)} muscle frames " +
          f"(out of {len(mf_metadata)} total, {100*len(frame_indices_list)/len(mf_metadata):.1f}%)")
    
    # Compute muscle activity only for required frames
    print("Computing muscle activity...")
    # Note: segmap_frame_ids is the segmentation lookup array (from h5 file)
    # frame_indices_list controls which muscle frames are actually processed
    muscle_activity_subset, top_k_masks, processed_masks_dict = compute_muscle_activity_for_frames(
        mf_metadata, processed_folder, 
        segmaps, frame_ids,  # frame_ids = segmentation lookup array
        seg_labels, segments_to_analyze, duo_yaml, 
        frame_indices_list,  # Only process these frame indices (subset)
        k=TOP_K_PIXELS,
        morph_kernel_size=MORPH_KERNEL_SIZE,
        morph_n_iterations=MORPH_N_ITERATIONS,
        dilation_kernels=DILATION_KERNELS,
        bilateral_d=BILATERAL_D,
        bilateral_sigma_color=BILATERAL_SIGMA_COLOR,
        bilateral_sigma_space=BILATERAL_SIGMA_SPACE,
        min_fragment_size=MIN_FRAGMENT_SIZE,
        max_fragment_distance=MAX_FRAGMENT_DISTANCE
    )
        
    # Create full-sized array with NaNs, then fill in computed values
    muscle_activity = np.full((len(mf_metadata), len(segments_to_analyze)), np.nan, dtype=np.float32)
    muscle_activity[frame_indices_list, :] = muscle_activity_subset

    # Check which stimulations are valid (stage didn't move too much)
    print("\nChecking stage movement...")
    valid_stim_mask = np.zeros(len(stim_start_frames), dtype=bool)
    stage_distances = []
    justifications = []
    
    for i, (start_frame, end_frame) in enumerate(zip(stim_start_frames, stim_end_frames)):
        # Check stage movement in display window
        window_start_frame = start_frame - display_window_frames
        window_end_frame = end_frame + display_window_frames
        moved_in_window, window_distance, window_reason = check_stage_movement(
            bf_metadata, window_start_frame, window_end_frame, max_movement_mm
        )
        
        valid_stim_mask[i] = not moved_in_window
        stage_distances.append(window_distance)
        
        status = "INCLUDED" if valid_stim_mask[i] else "EXCLUDED"
        justification = f"Stim {i+1}: {status}, {window_distance:.3f}mm\n"
        justification += f"  Window [{-window_sec:.1f}s to +{window_sec:.1f}s]: {window_reason}\n"
        justifications.append(justification)
        justifications.append("")  # Empty line
    
    print(f"Valid stimulations: {np.sum(valid_stim_mask)}/{len(valid_stim_mask)}")
    
    # Compute ΔF/F₀ using BASELINE_WINDOW_SEC (converted to frames)
    print(f"Computing ΔF/F₀ (baseline: {BASELINE_WINDOW_SEC}s = {analysis_window_frames} frames)...")
    
    delta_f_over_f = compute_delta_f_over_f(
        muscle_activity, mf_metadata, bf_metadata,
        stim_start_frames, stim_end_frames,
        analysis_window_frames,
        valid_stim_mask=valid_stim_mask,
        duo_yaml=duo_yaml
    )
    
    # Compute time axis for display
    time_sec = (mf_metadata["received_time_us"] - 
                mf_metadata["received_time_us"].iloc[0]) / 1e6
    
    # Convert stim frames to times for display
    stim_start_times = []
    stim_end_times = []
    for start_frame, end_frame in zip(stim_start_frames, stim_end_frames):
        start_time = bf_metadata[bf_metadata["behavior_frame_id"] == start_frame]["received_time_us"].values
        end_time = bf_metadata[bf_metadata["behavior_frame_id"] == end_frame]["received_time_us"].values
        if len(start_time) > 0 and len(end_time) > 0:
            stim_start_times.append((start_time[0] - bf_metadata["received_time_us"].iloc[0]) / 1e6)
            stim_end_times.append((end_time[0] - bf_metadata["received_time_us"].iloc[0]) / 1e6)
    
    # Extract display windows for valid stimulations
    print(f"\nExtracting {window_sec}s windows for valid stimulations...")
    valid_stimulations = []
    
    for i, (start_frame, end_frame) in enumerate(zip(stim_start_frames, stim_end_frames)):
        if valid_stim_mask[i]:
            # Define window in behavior frames
            window_start_bf = start_frame - display_window_frames
            window_end_bf = end_frame + display_window_frames
            
            # Convert to muscle frames (floor + 1 for end)
            window_start_mf = match_behavior_frameid_to_muscle_frameid(
                window_start_bf, method="floor", dual_recording_timing_metadata_path=duo_yaml
            )
            window_end_mf = match_behavior_frameid_to_muscle_frameid(
                window_end_bf, method="floor", dual_recording_timing_metadata_path=duo_yaml
            ) + 1
            
            # Extract window
            stim_start_time = stim_start_times[i]
            window_activity, window_time, window_mf_ids = extract_window_around_stimulation(
                delta_f_over_f, mf_metadata, time_sec.values,
                stim_start_time, window_start_mf, window_end_mf
            )
            
            if window_activity is not None:
                stim_end_time = stim_end_times[i] if i < len(stim_end_times) else stim_start_time + 0.1
                
                valid_stimulations.append({
                    'exp_name': exp_folder.name,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'stim_time': stim_start_time,
                    'stim_duration': stim_end_time - stim_start_time,
                    'window_activity': window_activity,
                    'window_time': window_time,
                    'distance': stage_distances[i],
                })
    
    return valid_stimulations, justifications, segments_to_analyze


def find_experiment_folders(data_folder_pattern):
    """Find all experiment folders matching the pattern.
    
    Args:
        data_folder_pattern: Can be a direct path or a glob pattern
    
    Returns:
        List of experiment folder paths that contain a 'processed' subfolder
    """
    # Use glob to expand the pattern
    matching_paths = glob.glob(str(data_folder_pattern))
    
    exp_folders = []
    for path in matching_paths:
        path = Path(path)
        if path.is_dir() and (path / "processed").exists():
            exp_folders.append(path)
    
    return sorted(exp_folders)


def aggregate_and_plot(all_valid_stims, segments_to_analyze, output_folder, 
                      window_sec, aggregate_by='all', max_movement_mm=1.0):
    """Create aggregated summary figures.
    
    Args:
        aggregate_by: 'all' = all flies together, 'fly' = separate per fly, 'trial' = separate per experiment
        max_movement_mm: Maximum allowed stage movement threshold
    """
    if len(all_valid_stims) == 0:
        print("WARNING: No valid stimulations found!")
        return
    
    # Group stimulations based on aggregation mode
    if aggregate_by == 'all':
        # All flies on the same graph
        groups = {'all_flies': all_valid_stims}
    elif aggregate_by == 'fly':
        # Group by fly (extract fly identifier from experiment name)
        groups = {}
        for stim in all_valid_stims:
            exp_name = stim['exp_name']
            # Extract fly name (e.g., "fly001" from "fly001_trial010_...")
            fly_name = exp_name.split('_trial')[0] if '_trial' in exp_name else exp_name.split('_')[0]
            if fly_name not in groups:
                groups[fly_name] = []
            groups[fly_name].append(stim)
    else:  # aggregate_by == 'trial'
        # Group by individual experiment/trial
        groups = {}
        for stim in all_valid_stims:
            exp_name = stim['exp_name']
            if exp_name not in groups:
                groups[exp_name] = []
            groups[exp_name].append(stim)
    
    # Create figure for each group
    for group_name, group_stims in groups.items():
        if len(group_stims) == 0:
            print(f"\nSkipping {group_name}: no valid stimulations")
            continue
            
        print(f"\nCreating figure for {group_name} with {len(group_stims)} stimulations...")
        
        n_segments = len(segments_to_analyze)
        # More square aspect ratio
        fig_height = 3 * n_segments
        fig_width = 6
        fig, axes = plt.subplots(n_segments, 1, figsize=(fig_width, fig_height), 
                                sharex=True, squeeze=False)
        
        # Get average stim duration (safe from division by zero since we checked len > 0)
        avg_stim_dur = np.mean([s['stim_duration'] for s in group_stims])
        
        for seg_idx, segment_name in enumerate(segments_to_analyze):
            ax = axes[seg_idx, 0]
            color = SEGMENT_COLORS.get(segment_name, "#000000")
            
            # Determine the common time range: [-window_sec, stim_duration + window_sec]
            time_end = avg_stim_dur + window_sec
            common_time = np.linspace(-window_sec, time_end, 100)
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
            if len(interpolated_traces) > 0:
                mean_trace = np.mean(interpolated_traces, axis=0)
                
                # Compute SEM only if we have more than 1 trace
                if len(interpolated_traces) > 1:
                    sem_trace = np.std(interpolated_traces, axis=0) / np.sqrt(len(interpolated_traces))
                else:
                    sem_trace = np.zeros_like(mean_trace)
                
                # Plot mean ± SEM
                ax.plot(common_time, mean_trace, color=color, linewidth=2.5)
                if len(interpolated_traces) > 1:
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
                if aggregate_by != 'all':
                    title = f"{group_name}: {title}"
                ax.set_title(title, fontsize=11, pad=10)
        
        # Set x-axis limits to match window definition
        time_end = avg_stim_dur + window_sec
        for ax_row in axes:
            ax_row[0].set_xlim(-window_sec, time_end)
        
        axes[-1, 0].set_xlabel("Time relative to stimulation onset (s)", fontsize=10)
        fig.text(0.04, 0.5, r"$\Delta$F/F$_0$", va='center', rotation='vertical', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure with hyperparameters in filename
        safe_name = group_name.replace('/', '_')
        fig_path = output_folder / f"{safe_name}_n{len(group_stims)}_win{window_sec}s_maxmov{max_movement_mm}mm_base{BASELINE_WINDOW_SEC}s.pdf"
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved figure to: {fig_path}")


def analyze_all_experiments(data_folder_pattern, segments_to_show, output_folder,
                           max_movement_mm, window_sec, aggregate_by):
    """Analyze all experiments matching the data folder pattern."""
    
    # Find all experiment folders
    exp_folders = find_experiment_folders(data_folder_pattern)
    
    if len(exp_folders) == 0:
        print(f"ERROR: No experiment folders found matching pattern: {data_folder_pattern}")
        return
    
    # Determine parent folder from first experiment for output
    parent_folder = exp_folders[0].parent
    data_folder_suffix = data_folder_pattern.name if isinstance(data_folder_pattern, Path) else Path(data_folder_pattern).name
    
    # Set output folder based on aggregation mode
    if output_folder is None:
        if aggregate_by == 'all':
            # All flies: save to parent folder
            output_folder = parent_folder / f"muscle_analysis_all_{data_folder_suffix}"
        elif aggregate_by == 'fly':
            # Per fly: save to parent folder with nice naming
            output_folder = parent_folder / f"muscle_analysis_per_fly_{data_folder_suffix}"
        else:  # aggregate_by == 'trial'
            # Per trial: will save to individual trial folders later
            output_folder = None
    
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing data folder pattern: {data_folder_pattern}")
    print(f"Aggregation rule: {aggregate_by}")
    if output_folder:
        print(f"Saving results to: {output_folder}")
    print(f"\nFound {len(exp_folders)} experiment folders:")
    for exp in exp_folders:
        print(f"  - {exp.name}")
    
    # Analyze each experiment
    all_valid_stims = []
    all_justifications = []
    segments_to_analyze = None
    
    for exp_folder in exp_folders:
        print(f"\n{'='*80}")
        print(f"Processing: {exp_folder.name}")
        print(f"{'='*80}")
        
        try:
            valid_stims, justifications, segments = analyze_experiment(
                exp_folder, segments_to_show, output_folder=None,
                max_movement_mm=max_movement_mm, window_sec=window_sec
            )
            
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
    
    # For per-trial aggregation, save results to individual trial folders
    if aggregate_by == 'trial':
        for exp_folder in exp_folders:
            trial_output = exp_folder / "stimulation_analysis"
            trial_output.mkdir(parents=True, exist_ok=True)
            
            # Filter stimulations for this trial
            trial_stims = [s for s in all_valid_stims if s['exp_name'] == exp_folder.name]
            
            if len(trial_stims) > 0:
                # Save justifications for this trial
                trial_just = [j for j in all_justifications if exp_folder.name in j]
                justification_file = trial_output / "stimulation_summary.txt"
                with open(justification_file, 'w') as f:
                    f.write(f"Experiment: {exp_folder.name}\n")
                    f.write(f"Max movement threshold: {max_movement_mm}mm\n")
                    f.write(f"Analysis window: [{-window_sec:.1f}s to +{window_sec:.1f}s] relative to stim start/end\n")
                    f.write(f"Total valid stimulations: {len(trial_stims)}\n")
                    if segments_to_analyze:
                        f.write(f"Segments: {', '.join(segments_to_analyze)}\n\n")
                    f.write("="*80 + "\n")
                    f.writelines(trial_just)
                
                # Create plot for this trial
                aggregate_and_plot(trial_stims, segments_to_analyze, trial_output,
                                 window_sec, aggregate_by='trial', max_movement_mm=max_movement_mm)
                print(f"Saved trial analysis to: {trial_output}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print(f"Total valid stimulations: {len(all_valid_stims)}")
        print(f"Results saved to individual trial folders")
        print("="*80)
        return
    
    # For 'all' and 'fly' aggregation, save to central output folder
    # Save combined justifications
    justification_file = output_folder / "all_stimulations_summary.txt"
    with open(justification_file, 'w') as f:
        f.write(f"Data folder pattern: {data_folder_pattern}\n")
        f.write(f"Aggregation: {aggregate_by}\n")
        f.write(f"Max movement threshold: {max_movement_mm}mm\n")
        f.write(f"Analysis window: [{-window_sec:.1f}s to +{window_sec:.1f}s] relative to stim start/end\n")
        f.write(f"Total experiments: {len(exp_folders)}\n")
        f.write(f"Total valid stimulations: {len(all_valid_stims)}\n")
        if segments_to_analyze:
            f.write(f"Segments: {', '.join(segments_to_analyze)}\n\n")
        else:
            f.write(f"Segments: None\n\n")
        f.write("="*80 + "\n")
        f.writelines(all_justifications)
    
    print(f"\n\nSaved combined justifications to: {justification_file}")
    
    # Save CSV with all valid stimulations
    csv_path = output_folder / "all_valid_stimulations.csv"
    stim_data = []
    for idx, stim in enumerate(all_valid_stims):
        stim_data.append({
            'experiment': stim['exp_name'],
            'stimulation_index': idx,
            'start_frame': stim['start_frame'],
            'end_frame': stim['end_frame'],
            'stim_time_sec': stim['stim_time'],
            'stim_duration_sec': stim['stim_duration'],
            'stage_movement_mm': stim['distance'],
        })
    pd.DataFrame(stim_data).to_csv(csv_path, index=False)
    print(f"Saved all stimulation data to: {csv_path}")
    
    # Create aggregated figures
    aggregate_and_plot(all_valid_stims, segments_to_analyze, output_folder,
                      window_sec, aggregate_by, max_movement_mm)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Total valid stimulations: {len(all_valid_stims)}")
    print(f"Results saved to: {output_folder}")
    print("="*80)
    print("\nAnalysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze all stimulations in experiment(s) and create summary figures."
    )
    parser.add_argument(
        "--exp_folder",
        type=str,
        help="Path to single experiment folder (for single experiment analysis)"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        help="Path to data folder containing multiple experiments (for multi-experiment analysis)"
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
        help="Output folder for results"
    )
    parser.add_argument(
        "--max_movement",
        type=float,
        default=0.5,
        help="Maximum allowed stage movement in mm (default: 0.5)"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.5,
        help="Time window (in seconds) before stim start and after stim end for analysis (default: 1.0)"
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="all",
        choices=["all", "fly", "trial"],
        help="Aggregation mode: 'all' = all flies together, 'fly' = separate per fly, 'trial' = separate per experiment (default: all)"
    )
    
    args = parser.parse_args()
    
    # Check that either exp_folder or data_folder is provided
    if args.exp_folder and args.data_folder:
        parser.error("Provide either --exp_folder or --data_folder, not both")
    if not args.exp_folder and not args.data_folder:
        parser.error("Must provide either --exp_folder or --data_folder")
    
    if args.exp_folder:
        # Single experiment analysis
        # Create output folder if not specified
        if args.output_folder is None:
            output_folder = Path(args.exp_folder) / "stimulation_analysis"
        else:
            output_folder = Path(args.output_folder)
        
        valid_stims, justifications, segments = analyze_experiment(
            args.exp_folder, 
            args.segments, 
            output_folder,  # Pass the output folder, not None
            args.max_movement,
            args.window
        )
        
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Save justifications
        justification_file = output_folder / "stimulation_summary.txt"
        with open(justification_file, 'w') as f:
            f.write(f"Experiment: {Path(args.exp_folder).name}\n")
            f.write(f"Max movement threshold: {args.max_movement}mm\n")
            f.write(f"Analysis window: [{-args.window:.1f}s to +{args.window:.1f}s] relative to stim start/end\n")
            f.write(f"Total valid stimulations: {len(valid_stims)}\n")
            f.write(f"Segments: {', '.join(segments)}\n\n")
            f.write("="*80 + "\n")
            f.writelines(justifications)
        print(f"Saved justifications to: {justification_file}")
        
        # Create plot for this experiment
        if len(valid_stims) > 0:
            aggregate_and_plot(valid_stims, segments, output_folder, 
                             args.window, aggregate_by='trial', max_movement_mm=args.max_movement)
            print(f"\nAnalysis complete! Results saved to: {output_folder}")
        else:
            print("\nNo valid stimulations found!")
    else:
        # Multi-experiment analysis
        analyze_all_experiments(
            args.data_folder,
            args.segments,
            args.output_folder,
            args.max_movement,
            args.window,
            args.aggregate
        )


if __name__ == "__main__":
    main()
