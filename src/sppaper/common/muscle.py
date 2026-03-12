"""
Common utilities for muscle activity analysis.
"""

import cv2
import numpy as np
import matplotlib.colors as mcolors
from tqdm import tqdm

from spotlight_tools.postprocessing.muscle import (
    match_muscle_frameid_to_behavior_frameid,
)
from poseforge.spotlight.muscle_segmentation import _denoise_masks, _dilate_masks


def mean_of_k_top(img, k):
    """Compute mean of top k pixel values."""
    flat = img.flatten()
    if len(flat) < k:
        return flat.mean()
    top_k = np.partition(flat, -k)[-k:]
    return top_k.mean()


def create_segmentation_overlay(muscle_frame, processed_masks, seg_labels, segments_to_show,
                               segment_colors, segment_blend_original,
                               segment_blend_color, top_k_highlight_factor,
                               top_k_masks_dict=None, frame_idx=None):
    """Create RGB image with segmentation overlay using processed masks.
    
    Args:
        muscle_frame: Grayscale or RGB muscle frame
        processed_masks: List of processed (denoised and dilated) masks, one per segment
        seg_labels: List of all segment labels
        segments_to_show: List of segment names to visualize
        segment_colors: Dict mapping segment names to hex color strings
        segment_blend_original: Weight of original image in overlay
        segment_blend_color: Weight of segment color in overlay
        top_k_highlight_factor: Brightness factor for top-k pixels
        top_k_masks_dict: Optional dict mapping (frame_idx, segment_name) to boolean mask
        frame_idx: Frame index for looking up top-k masks
    
    Returns:
        RGB image with segmentation overlay
    """
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
        # Use processed mask instead of raw segmap
        mask = processed_masks[seg_id].astype(bool)
        
        if not mask.any():  # Skip if segment not present
            continue
        
        # Get color for this segment
        color_hex = segment_colors[segment_name]  # Will raise KeyError if not in config
        color_rgb = np.array(mcolors.to_rgb(color_hex)) * 255
        
        # Vectorized blending with configurable weights
        mask_3d = mask[:, :, np.newaxis]
        muscle_rgb = np.where(mask_3d, 
                             segment_blend_original * muscle_rgb + segment_blend_color * color_rgb,
                             muscle_rgb)
        
        # Highlight top k pixels in darker shade (using precomputed masks)
        if top_k_masks_dict is not None and frame_idx is not None:
            top_k_mask = top_k_masks_dict.get((frame_idx, segment_name))
            if top_k_mask is not None and top_k_mask.any():
                dark_color = color_rgb * top_k_highlight_factor
                # Apply dark color to top k pixels
                top_k_mask_3d = top_k_mask[:, :, np.newaxis]
                muscle_rgb = np.where(top_k_mask_3d, dark_color, muscle_rgb)
    
    return muscle_rgb.astype(np.uint8)


def compute_muscle_activity(mf_metadata, processed_folder, segmaps, segmap_frame_ids, 
                           seg_labels, segments_to_analyze, duo_yaml, k,
                           morph_kernel_size, morph_n_iterations, 
                           dilation_kernels, bilateral_d, 
                           bilateral_sigma_color, bilateral_sigma_space,
                           min_fragment_size=50, max_fragment_distance=100):
    """Extract muscle activity for all frames and selected segments.
    
    Args:
        mf_metadata: DataFrame with muscle frame metadata
        processed_folder: Path to processed data folder
        segmaps: Array of segmentation maps
        segmap_frame_ids: Array of frame IDs corresponding to segmaps (from h5 file)
        seg_labels: List of segment labels
        segments_to_analyze: List of segment names to analyze
        duo_yaml: Path to dual recording timing YAML file
        k: Number of top pixels to use for activity computation
        morph_kernel_size: Size of morphological kernel for denoising
        morph_n_iterations: Number of morphological iterations
        dilation_kernels: Dict mapping segment names to {'direction': str, 'size': int}
        bilateral_d: Diameter of pixel neighborhood for bilateral filter
        bilateral_sigma_color: Filter sigma in color space
        bilateral_sigma_space: Filter sigma in coordinate space
        min_fragment_size: Minimum size in pixels to keep a fragment
        max_fragment_distance: Maximum distance in pixels between fragments to merge
    
    Returns:
        muscle_activity: (n_frames, n_segments) array of activity values
        top_k_masks: dict mapping (frame_idx, segment_name) to boolean mask of top k pixels
        processed_masks_dict: dict mapping frame_idx to list of processed masks
    """
    # Simply call compute_muscle_activity_for_frames with all indices
    frame_indices = list(range(len(mf_metadata)))
    return compute_muscle_activity_for_frames(
        mf_metadata, processed_folder, segmaps, segmap_frame_ids,
        seg_labels, segments_to_analyze, duo_yaml, frame_indices, k,
        morph_kernel_size, morph_n_iterations, dilation_kernels,
        bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
        min_fragment_size, max_fragment_distance
    )

def create_asymmetric_dilation_kernel(direction, size):
    """Create asymmetric directional dilation kernel.
    
    Args:
        direction: 'lower_left', 'lower_right', or 'uniform'
        size: Size of the kernel extension (can be 0 for no dilation)
    
    Returns:
        kernel: Asymmetric kernel (empty if size=0)
        anchor: Anchor point for dilation
    """
    # Handle zero-size kernel (no dilation)
    if size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    half_size = size // 2
    if direction == 'lower_left':
        # Extend toward lower left (for RF legs)
        # Fill upper-right quadrant
        kernel[half_size:, :] = 0  # Lower half
        kernel[:, :half_size] = 0  # Left half
        
    elif direction == 'lower_right':
        # Extend toward lower right (for LF legs)
        # Fill upper-left quadrant
        kernel[half_size:, :] = 0  # Lower half
        kernel[:, half_size+1:] = 0  # Right half
    elif direction == 'left_narrow':
        kernel = np.zeros((size, size), dtype=np.uint8)
        kernel[half_size, half_size:] = 1  # Narrow extension to the right        
    elif direction == 'right_narrow':
        kernel = np.zeros((size, size), dtype=np.uint8)
        kernel[half_size, :half_size+1] = 1  # Narrow extension to the left
    elif direction == 'uniform':
        # Standard circular dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    return kernel


def _denoise_masks_with_fragment_merging(
    segmap: np.ndarray,
    class_labels: list,
    morph_kernel: np.ndarray,
    n_iterations: int,
    min_fragment_size: int,
    max_fragment_distance: int,
) -> np.ndarray:
    """Apply morphological denoising to all class masks with fragment merging.

    After morphological operations, instead of keeping only the largest component,
    this function preserves and merges valid fragments:
    - Keeps fragments >= min_fragment_size pixels
    - Merges fragments within max_fragment_distance pixels using convex hull
    - Falls back to largest component if fragments are too far apart

    Args:
        segmap: Segmentation map with class labels, shape (H, W)
        class_labels: List of class names corresponding to segmap values
        morph_kernel: Structuring element for morphological operations
        n_iterations: Number of iterations for opening and closing
        min_fragment_size: Minimum size in pixels to keep a fragment
        max_fragment_distance: Maximum distance in pixels between fragments to merge

    Returns:
        Array of denoised binary masks, shape (len(class_labels), H, W)
    """
    denoised_masks = []

    for i, class_label in enumerate(class_labels):
        mask = (segmap == i).astype(np.uint8)

        if class_label.lower() == "background":
            denoised_mask = mask.astype(bool)
        else:
            # Morphological opening and closing
            opened = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, morph_kernel, iterations=n_iterations
            )
            closed = cv2.morphologyEx(
                opened, cv2.MORPH_CLOSE, morph_kernel, iterations=n_iterations
            )

            # Find connected components with stats
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
                closed, connectivity=8
            )
            
            if num_labels <= 1:
                denoised_mask = np.zeros_like(mask, dtype=bool)
            else:
                # Filter components by size (skip label 0 which is background)
                valid_components = []
                for label_id in range(1, num_labels):
                    area = stats[label_id, cv2.CC_STAT_AREA]
                    if area >= min_fragment_size:
                        valid_components.append(label_id)
                
                if len(valid_components) == 0:
                    # No valid components, return empty mask
                    denoised_mask = np.zeros_like(mask, dtype=bool)
                elif len(valid_components) == 1:
                    # Only one valid component, use it
                    denoised_mask = labels_im == valid_components[0]
                else:
                    # Multiple valid components - build distance graph and find largest cluster
                    component_centroids = [centroids[label_id] for label_id in valid_components]
                    n_components = len(valid_components)
                    
                    # Build adjacency matrix: components are connected if within max_fragment_distance
                    adjacency = np.zeros((n_components, n_components), dtype=bool)
                    for j in range(n_components):
                        for k in range(j + 1, n_components):
                            dist = np.linalg.norm(component_centroids[j] - component_centroids[k])
                            if dist <= max_fragment_distance:
                                adjacency[j, k] = True
                                adjacency[k, j] = True
                    
                    # Find connected components in the distance graph using BFS
                    visited = np.zeros(n_components, dtype=bool)
                    clusters = []
                    
                    for start_idx in range(n_components):
                        if visited[start_idx]:
                            continue
                        
                        # BFS to find all connected components
                        cluster = []
                        queue = [start_idx]
                        visited[start_idx] = True
                        
                        while queue:
                            current = queue.pop(0)
                            cluster.append(current)
                            
                            # Add all unvisited neighbors
                            for neighbor_idx in range(n_components):
                                if adjacency[current, neighbor_idx] and not visited[neighbor_idx]:
                                    visited[neighbor_idx] = True
                                    queue.append(neighbor_idx)
                        
                        clusters.append(cluster)
                    
                    # Select the largest cluster (by total area)
                    largest_cluster = max(clusters, key=lambda c: sum(stats[valid_components[idx], cv2.CC_STAT_AREA] for idx in c))
                    components_to_merge = [valid_components[idx] for idx in largest_cluster]
                    
                    # Create combined mask of selected components
                    combined_mask = np.zeros_like(mask)
                    for label_id in components_to_merge:
                        combined_mask |= (labels_im == label_id).astype(np.uint8)
                    
                    # Find contours and compute convex hull
                    contours, _ = cv2.findContours(
                        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if len(contours) > 0:
                        # Merge all contours and compute convex hull
                        all_points = np.vstack(contours)
                        hull = cv2.convexHull(all_points)
                        
                        # Create mask from convex hull
                        merged_mask = np.zeros_like(mask)
                        cv2.fillPoly(merged_mask, [hull], 1)
                        denoised_mask = merged_mask.astype(bool)
                    else:
                        denoised_mask = combined_mask.astype(bool)

        denoised_masks.append(denoised_mask)

    return np.stack(denoised_masks, axis=0)


def denoise_and_dilate_masks_with_config(segmap, seg_labels, morph_kernel_size, 
                                         morph_n_iterations, dilation_kernels,
                                         min_fragment_size, max_fragment_distance):
    """Denoise and dilate masks using kernel configuration dictionary.
    
    After denoising, valid mask fragments are preserved and merged if they meet criteria:
    - Fragment size >= min_fragment_size pixels
    - Distance between fragments <= max_fragment_distance pixels
    
    Args:
        segmap: Segmentation map
        seg_labels: List of segment labels
        morph_kernel_size: Size for denoising morphological operations
        morph_n_iterations: Number of denoising iterations
        dilation_kernels: Dict mapping segment names to {'direction': str, 'size': int}
        min_fragment_size: Minimum size in pixels to keep a fragment (default: 50)
        max_fragment_distance: Maximum distance in pixels between fragments to merge (default: 100)
    
    Returns:
        List of dilated masks
    """
    # Denoise with fragment merging
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    denoised_masks = _denoise_masks_with_fragment_merging(
        segmap, seg_labels, morph_kernel, morph_n_iterations,
        min_fragment_size=min_fragment_size, max_fragment_distance=max_fragment_distance
    )
    
    # Get default kernel config from the dictionary
    default_kernel_config = dilation_kernels['default']
    
    # Apply segment-specific dilation
    final_masks = []
    for i, seg_label in enumerate(seg_labels):
        mask = denoised_masks[i].astype(np.uint8)
        
        # Get kernel config for this segment (or use default from config)
        kernel_config = dilation_kernels.get(seg_label, default_kernel_config)
        direction = kernel_config['direction']
        size = kernel_config['size']
                # Skip dilation if size is 0
        if size == 0:
            final_masks.append(mask)
            continue
                # Create and apply kernel
        kernel = create_asymmetric_dilation_kernel(direction, size)
        # do not dilate if kernel is empty (size=0 or direction results in empty kernel)
        if kernel.sum() > 0:
            # Note: iterations=1 is appropriate for dilation (single pass)
            dilated = cv2.dilate(mask, kernel, iterations=1)
        else:
            dilated = mask
        
        final_masks.append(dilated)
    
    return final_masks


def compute_muscle_activity_for_frames(mf_metadata, processed_folder, segmaps, segmap_frame_ids,
                                      seg_labels, segments_to_analyze, duo_yaml,
                                      frame_indices, k,
                                      morph_kernel_size, morph_n_iterations,
                                      dilation_kernels, bilateral_d,
                                      bilateral_sigma_color, bilateral_sigma_space,
                                      min_fragment_size, max_fragment_distance):
    """Extract muscle activity for specific frame indices only.
    
    More efficient than compute_muscle_activity when you only need a subset of frames.
    
    Args:
        mf_metadata: DataFrame with muscle frame metadata
        processed_folder: Path to processed data folder
        segmaps: Array of segmentation maps
        segmap_frame_ids: Array of frame IDs corresponding to segmaps (from h5 file, used for lookup)
        seg_labels: List of segment labels
        segments_to_analyze: List of segment names to analyze
        duo_yaml: Path to dual recording timing YAML file
        frame_indices: List or array of frame indices to process (indices into mf_metadata)
        k: Number of top pixels to use for activity computation
        morph_kernel_size: Size of morphological kernel for denoising
        morph_n_iterations: Number of morphological iterations
        dilation_kernels: Dict mapping segment names to {'direction': str, 'size': int}
        bilateral_d: Diameter of pixel neighborhood for bilateral filter
        bilateral_sigma_color: Filter sigma in color space
        bilateral_sigma_space: Filter sigma in coordinate space
        min_fragment_size: Minimum size in pixels to keep a fragment
        max_fragment_distance: Maximum distance in pixels between fragments to merge
    
    Returns:
        muscle_activity: (len(frame_indices), n_segments) array of activity values
        top_k_masks: dict mapping (idx_in_output, segment_name) to boolean mask
        processed_masks_dict: dict mapping frame_idx to list of processed masks
    """
    n_frames = len(frame_indices)
    n_segments = len(segments_to_analyze)
    
    muscle_activity = np.ones((n_frames, n_segments), dtype=np.float32)*np.nan
    top_k_masks = {}
    processed_masks_dict = {}  # Store processed masks for each frame
    segment_ids = [seg_labels.index(seg) for seg in segments_to_analyze]
    
    # Process only requested frames (frame_indices controls which frames are actually processed)
    for out_idx, frame_idx in enumerate(tqdm(frame_indices, 
                                             desc="Computing muscle activity")):        
        muscle_frame_raw = cv2.imread(
            str(processed_folder / f"aligned_muscle_images/muscle_frame_{frame_idx:09d}.tif"),
            cv2.IMREAD_UNCHANGED
        )
        
        # Apply bilateral filter if configured
        if bilateral_d is not None and bilateral_d > 0:
            if muscle_frame_raw.dtype == np.uint16:
                muscle_frame_float = muscle_frame_raw.astype(np.float32)
                muscle_frame_float = cv2.bilateralFilter(muscle_frame_float, bilateral_d,
                                                         bilateral_sigma_color, bilateral_sigma_space)
                muscle_frame = muscle_frame_float.astype(np.uint16)
            else:
                muscle_frame = cv2.bilateralFilter(muscle_frame_raw, bilateral_d,
                                                   bilateral_sigma_color, bilateral_sigma_space)
        
        # Get corresponding segmap (segmap_frame_ids is lookup array, not list of frames to process)
        bf_frame_id = match_muscle_frameid_to_behavior_frameid(
            frame_idx, dual_recording_timing_metadata_path=duo_yaml
        )
        segmap = segmaps[segmap_frame_ids == bf_frame_id][0]
        
        # Denoise and dilate masks using configured kernels
        final_masks = denoise_and_dilate_masks_with_config(
            segmap, seg_labels, morph_kernel_size, morph_n_iterations,
            dilation_kernels, min_fragment_size, max_fragment_distance
        )
        
        # Store processed masks for this frame (for overlay visualization)
        processed_masks_dict[out_idx] = final_masks
        
        # Extract activity for each segment
        for j, seg_id in enumerate(segment_ids):
            muscle_mask = final_masks[seg_id].astype(np.uint8)
            # Note: cv2.INTER_NEAREST is appropriate for masks (no interpolation)
            muscle_mask_resized = cv2.resize(
                muscle_mask,
                (muscle_frame.shape[1], muscle_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Get pixels in this segment
            segment_pixels = muscle_frame_raw[muscle_mask_resized == 1]
            
            # Compute top-k mean and find threshold in one pass
            if len(segment_pixels) >= k:
                # Compute partition once and reuse for both mean and threshold
                flat_pixels = segment_pixels.flatten()
                top_k_values = np.partition(flat_pixels, -k)[-k:]
                muscle_activity[out_idx, j] = top_k_values.mean()
                if top_k_values.mean() <= 0:
                    print(f"Warning: Mean of top-k pixels is non-positive for frame {frame_idx}, segment {segments_to_analyze[j]}")
                    print(f"Top-k values: {top_k_values}")
                
                # Store top-k pixel mask using the threshold
                threshold = top_k_values.min()  # Minimum of top-k is the threshold
                top_k_mask = (muscle_frame >= threshold) & (muscle_mask_resized == 1)
                segment_name = segments_to_analyze[j]
                top_k_masks[(out_idx, segment_name)] = top_k_mask
            else:
                # Not enough pixels, just use mean of all
                muscle_activity[out_idx, j] = segment_pixels.mean() if len(segment_pixels) > 0 else np.nan
    
    return muscle_activity, top_k_masks, processed_masks_dict


def compute_delta_f_over_f(muscle_activity, mf_metadata, bf_metadata,
                           stim_start_frames, stim_end_frames, 
                           baseline_window_frames, valid_stim_mask=None,
                           duo_yaml=None):
    """Compute ΔF/F₀ using baseline from periods around stimulation.
    
    F0 is computed as the minimum of the muscle activity values in the baseline windows
    (specified frames before and after each stimulation).
    
    Args:
        muscle_activity: (n_frames, n_segments) array of activity values
        mf_metadata: Muscle frames metadata DataFrame
        bf_metadata: Behavior frames metadata DataFrame  
        stim_start_frames: List of stimulation start frames (behavior frame IDs)
        stim_end_frames: List of stimulation end frames (behavior frame IDs)
        baseline_window_frames: Duration of baseline window in behavior frames
        valid_stim_mask: Boolean array indicating which stimulations are valid.
                        If None, all stimulations are considered valid. Default: None
        duo_yaml: Path to dual recording timing metadata YAML file
    
    Returns:
        delta_f_over_f: (n_frames, n_segments) array of ΔF/F₀ values
    """
    from spotlight_tools.postprocessing.muscle import match_behavior_frameid_to_muscle_frameid
    
    delta_f_over_f = np.zeros_like(muscle_activity)
    n_segments = muscle_activity.shape[1]
    
    # If no mask provided, consider all stimulations valid
    if valid_stim_mask is None:
        valid_stim_mask = np.ones(len(stim_start_frames), dtype=bool)
    
    for j in range(n_segments):
        baseline_values = []
        
        # Collect baseline values around each VALID stimulation only
        for i, (start_frame, end_frame) in enumerate(zip(stim_start_frames, stim_end_frames)):
            # Skip if stimulation is not valid (stage moved too much)
            if not valid_stim_mask[i]:
                continue
            
            # Define baseline windows in behavior frames
            pre_stim_start_bf = start_frame - baseline_window_frames
            pre_stim_end_bf = start_frame
            post_stim_start_bf = end_frame
            post_stim_end_bf = end_frame + baseline_window_frames
            
            # Convert to muscle frames
            pre_stim_start_mf = match_behavior_frameid_to_muscle_frameid(
                pre_stim_start_bf, method="floor", dual_recording_timing_metadata_path=duo_yaml
            )
            pre_stim_end_mf = match_behavior_frameid_to_muscle_frameid(
                pre_stim_end_bf, method="floor", dual_recording_timing_metadata_path=duo_yaml
            )
            post_stim_start_mf = match_behavior_frameid_to_muscle_frameid(
                post_stim_start_bf, method="floor", dual_recording_timing_metadata_path=duo_yaml
            )
            post_stim_end_mf = match_behavior_frameid_to_muscle_frameid(
                post_stim_end_bf, method="floor", dual_recording_timing_metadata_path=duo_yaml
            ) + 1
            
            # Create masks for baseline periods
            pre_stim_mask = (
                (mf_metadata["muscle_frame_id"] >= pre_stim_start_mf) & 
                (mf_metadata["muscle_frame_id"] < pre_stim_end_mf)
            )
            post_stim_mask = (
                (mf_metadata["muscle_frame_id"] >= post_stim_start_mf) & 
                (mf_metadata["muscle_frame_id"] < post_stim_end_mf)
            )
            
            # Collect baseline values (excluding NaNs)
            pre_vals = muscle_activity[pre_stim_mask.values, j]
            post_vals = muscle_activity[post_stim_mask.values, j]
            baseline_values.extend(pre_vals.tolist())
            baseline_values.extend(post_vals.tolist())
                
        # F0 is the MINIMUM of baseline values
        if len(baseline_values) > 0:
            f0 = np.nanmin(baseline_values)
        else:
            # Fallback: use minimum of entire trace (excluding NaNs)
            valid_vals = muscle_activity[:, j]
            if len(valid_vals) > 0:
                f0 = np.nanmin(valid_vals)
            else:
                f0 = 1.0  # Last resort fallback
        
        assert f0 > 0, f"F0 is zero or negative for segment {j}, cannot compute ΔF/F₀"

        # Compute ΔF/F₀
        delta_f_over_f[:, j] = (muscle_activity[:, j] - f0) / f0
        
    return delta_f_over_f
