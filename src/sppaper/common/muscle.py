"""
Common utilities for muscle activity analysis.
"""

import cv2
import numpy as np
from tqdm import tqdm

from spotlight_tools.postprocessing.muscle import (
    match_muscle_frameid_to_behavior_frameid,
)
from poseforge.spotlight.muscle_segmentation import _denoise_masks, _dilate_masks


def mean_of_k_top(img, k=20):
    """Compute mean of top k pixel values."""
    flat = img.flatten()
    if len(flat) < k:
        return flat.mean()
    top_k = np.partition(flat, -k)[-k:]
    return top_k.mean()


def compute_muscle_activity(mf_metadata, processed_folder, segmaps, frame_ids, 
                           seg_labels, segments_to_analyze, duo_yaml, k=20,
                           morph_kernel_size=2, morph_n_iterations=1, 
                           dilation_kernel_size=3, bilateral_d=9, 
                           bilateral_sigma_color=75, bilateral_sigma_space=75):
    """Extract muscle activity for all frames and selected segments.
    
    Args:
        mf_metadata: DataFrame with muscle frame metadata
        processed_folder: Path to processed data folder
        segmaps: Array of segmentation maps
        frame_ids: Array of frame IDs corresponding to segmaps
        seg_labels: List of segment labels
        segments_to_analyze: List of segment names to analyze
        duo_yaml: Path to dual recording timing YAML file
        k: Number of top pixels to use for activity computation
        morph_kernel_size: Size of morphological kernel for denoising
        morph_n_iterations: Number of morphological iterations
        dilation_kernel_size: Size of dilation kernel
        bilateral_d: Diameter of pixel neighborhood for bilateral filter (0 = compute from sigmaSpace)
        bilateral_sigma_color: Filter sigma in color space (larger = more colors mixed)
        bilateral_sigma_space: Filter sigma in coordinate space (larger = more distant pixels influence)
    
    Returns:
        muscle_activity: (n_frames, n_segments) array of activity values
        top_k_masks: dict mapping (frame_idx, segment_name) to boolean mask of top k pixels
    """
    n_frames = len(mf_metadata)
    n_segments = len(segments_to_analyze)
    
    muscle_activity = np.zeros((n_frames, n_segments), dtype=np.float32)
    top_k_masks = {}  # Store top-k masks for visualization
    segment_ids = [seg_labels.index(seg) for seg in segments_to_analyze]
    
    # Compute activity for each frame
    for i, mf_id in enumerate(tqdm(mf_metadata["muscle_frame_id"], 
                                   desc="Computing muscle activity")):
        muscle_frame = cv2.imread(
            str(processed_folder / f"aligned_muscle_images/muscle_frame_{mf_id:09d}.tif"),
            cv2.IMREAD_UNCHANGED
        )
        
        # Apply bilateral filter if requested (preserves edges while smoothing)
        if bilateral_d > 0:
            # Convert to float32 for bilateral filter (only supports uint8 and float32)
            if muscle_frame.dtype == np.uint16:
                muscle_frame_float = muscle_frame.astype(np.float32)
                muscle_frame_float = cv2.bilateralFilter(muscle_frame_float, bilateral_d, 
                                                         bilateral_sigma_color, bilateral_sigma_space)
                muscle_frame = muscle_frame_float.astype(np.uint16)
            else:
                muscle_frame = cv2.bilateralFilter(muscle_frame, bilateral_d, 
                                                   bilateral_sigma_color, bilateral_sigma_space)
        
        # Get corresponding segmap
        bf_frame_id = match_muscle_frameid_to_behavior_frameid(
            mf_id, dual_recording_timing_metadata_path=duo_yaml
        )
        segmap = segmaps[frame_ids == bf_frame_id][0]
        
        # Process segmentation masks
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        denoised_masks = _denoise_masks(segmap, seg_labels, morph_kernel, 
                                       morph_n_iterations)
        
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
        )
        final_masks = _dilate_masks(denoised_masks, seg_labels, dilation_kernel)
        
        # Extract activity for each segment
        for j, seg_id in enumerate(segment_ids):
            muscle_mask = final_masks[seg_id].astype(np.uint8)
            muscle_mask_resized = cv2.resize(
                muscle_mask, 
                (muscle_frame.shape[1], muscle_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Get pixels in this segment
            segment_pixels = muscle_frame[muscle_mask_resized == 1]
            muscle_activity[i, j] = mean_of_k_top(segment_pixels, k=k)
            
            # Find and store top-k pixel mask
            if len(segment_pixels) >= k:
                threshold = np.partition(segment_pixels.flatten(), -k)[-k]
                top_k_mask = (muscle_frame >= threshold) & (muscle_mask_resized == 1)
                segment_name = segments_to_analyze[j]
                top_k_masks[(i, segment_name)] = top_k_mask
    
    return muscle_activity, top_k_masks


def compute_delta_f_over_f(muscle_activity, time_sec, stim_start_times, 
                           stim_end_times, baseline_window_sec=1.0):
    """Compute ΔF/F₀ using baseline from periods around stimulation.
    
    F0 is computed as the minimum of the top-k pixel values in the baseline windows
    (specified seconds before and after each stimulation).
    
    Args:
        muscle_activity: (n_frames, n_segments) array of activity values
        time_sec: Array of time values in seconds
        stim_start_times: List of stimulation start times in seconds
        stim_end_times: List of stimulation end times in seconds
        baseline_window_sec: Duration of baseline window in seconds
    
    Returns:
        delta_f_over_f: (n_frames, n_segments) array of ΔF/F₀ values
    """
    delta_f_over_f = np.zeros_like(muscle_activity)
    n_segments = muscle_activity.shape[1]
    
    for j in range(n_segments):
        baseline_values = []
        
        # Collect baseline values around each stimulation
        for start_time, end_time in zip(stim_start_times, stim_end_times):
            # Find frames within baseline windows
            pre_stim_mask = (time_sec >= start_time - baseline_window_sec) & \
                           (time_sec < start_time)
            post_stim_mask = (time_sec > end_time) & \
                            (time_sec <= end_time + baseline_window_sec)
            
            baseline_values.extend(muscle_activity[pre_stim_mask, j].tolist())
            baseline_values.extend(muscle_activity[post_stim_mask, j].tolist())
        
        # F0 is the MINIMUM of baseline values (top-k pixels in baseline windows)
        if len(baseline_values) > 0:
            f0 = np.min(baseline_values)
        else:
            # Fallback: use minimum of entire trace
            f0 = np.min(muscle_activity[:, j])
        
        # Avoid division by zero
        f0 = max(f0, 1e-6)
        
        # Compute ΔF/F₀
        delta_f_over_f[:, j] = (muscle_activity[:, j] - f0) / f0
    
    return delta_f_over_f
