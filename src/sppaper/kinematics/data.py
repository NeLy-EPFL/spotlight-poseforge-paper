import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import h5py
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from spotlight_tools.calibration.mapper import SpotlightPositionMapper
from poseforge.neuromechfly.constants import dof_name_lookup_canonical_to_nmf

import sppaper.common.filter as filter
import sppaper.kinematics.trajectory as traj
from sppaper.common.resources import get_spotlight_trials_dir

POSE_WORKING_DIM = 256
RAW_INPUT_CROP_DIM = 900
SWING_SPEED_THR = 25  # mm/s
SWING_SPEED_SG_WINDOW = 7
SWING_MASK_MEDFILTER_WINDOW = 5


class KinematicsDataset:
    def __init__(
        self,
        *,
        spotlight_trial_dirs,
        min_xy_conf,
        mask_denoise_kernel_size_sec,
        min_duration_sec,
        data_fps,
    ):
        self.summary_df, self.data_by_idx = _load_poseforge_output(
            spotlight_trial_dirs=spotlight_trial_dirs,
            min_xy_conf=min_xy_conf,
            mask_denoise_kernel_size_sec=mask_denoise_kernel_size_sec,
            min_duration_sec=min_duration_sec,
            data_fps=data_fps,
        )
        self._snippets = []
        for idx, data in self.data_by_idx.items():
            joints_order = [
                f"{leg}{dof}"
                for leg in data["legs_order"]
                for dof in data["dofs_order_per_leg"]
            ]
            joint_angles_arr = data["joint_angles"]
            # Flatten leg and dof-per-leg into a single dim
            joint_angles_arr = joint_angles_arr.reshape(joint_angles_arr.shape[0], -1)
            metadata = {
                "data_fps": data["fps"],
                "keypoints_order": data["keypoints_order"],
                "legs_order": data["legs_order"],
                "joints_order": joints_order,
                "keypoints_order_per_leg": data["keypoints_order_per_leg"],
                "dofs_order_per_leg": data["dofs_order_per_leg"],
            }
            snippet = KinematicsSnippet(
                exp_trial=data["trial"],
                start_idx=data["start_idx"],
                end_idx=data["end_idx"],
                joint_angles=joint_angles_arr,
                fwdkin_world_xyz=data["fwdkin_world_xyz"],
                cam_xy=data["cam_xy"],
                metadata=metadata,
            )
            self._snippets.append(snippet)

    def __len__(self):
        return len(self._snippets)

    def __getitem__(self, idx) -> "KinematicsSnippet":
        return self._snippets[idx]


@dataclass
class KinematicsSnippet:
    exp_trial: str
    start_idx: int
    end_idx: int
    joint_angles: np.ndarray
    fwdkin_world_xyz: np.ndarray
    cam_xy: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.data_fps = self.metadata["data_fps"]
        self.duration_s = (self.end_idx - self.start_idx) / self.data_fps
        self.exp_trial_dir = get_spotlight_trials_dir() / self.exp_trial

        # Load thorax position in pixel coords in raw Spotlight recording
        transform_file = (
            self.exp_trial_dir / "processed/behavior_alignment_transforms.h5"
        )
        with h5py.File(transform_file, "r") as f:
            coarse_kpts = list(f["keypoints_xy_pre_alignment"].attrs["keypoint_names"])
            thorax_idx = coarse_kpts.index("thorax")
            neck_idx = coarse_kpts.index("neck")
            kpts_xy = f["keypoints_xy_pre_alignment"][self.start_idx : self.end_idx]
            self.thorax_pos_px = kpts_xy[:, thorax_idx, :]
            self.neck_pos_px = kpts_xy[:, neck_idx, :]
            self.crop_transmats = f["transform_matrices"][self.start_idx : self.end_idx]

        # Set up mapper to convert from (translation stage pos, pixel coords) to
        # physical coordinates in the arena
        self.spotlight_coords_mapper = SpotlightPositionMapper(
            self.exp_trial_dir / "metadata/calibration_parameters_behavior.yaml"
        )

        # Load translation stage positions at the time each frame was recorded
        beh_frame_metadata_file = (
            self.exp_trial_dir / "processed/behavior_frames_metadata.csv"
        )
        frame_metadata = pl.read_csv(beh_frame_metadata_file)
        frame_metadata = frame_metadata[self.start_idx : self.end_idx]
        cols = ["x_pos_mm_interp", "y_pos_mm_interp"]
        self.stage_pos_mm = frame_metadata.select(cols).to_numpy()

        # Load camera calibration parameters and map
        # (translation stage pos, pixel corrds) to physical coordinates in the arena
        self.thorax_pos_mm = self.spotlight_coords_mapper.stage_and_pixel_to_physical(
            self.stage_pos_mm, self.thorax_pos_px
        )
        self.neck_pos_mm = self.spotlight_coords_mapper.stage_and_pixel_to_physical(
            self.stage_pos_mm, self.neck_pos_px
        )

    def get_subselection(self, start_sec, end_sec) -> "KinematicsSnippet":
        start_idx = int(start_sec * self.data_fps)
        end_idx = int(end_sec * self.data_fps)
        if start_idx < 0 or end_idx > len(self):
            raise ValueError("Subselection out of bounds")
        if start_idx >= end_idx:
            raise ValueError("start_sec must be less than end_sec")
        return KinematicsSnippet(
            exp_trial=self.exp_trial,
            start_idx=self.start_idx + start_idx,
            end_idx=self.start_idx + end_idx,
            joint_angles=self.joint_angles[start_idx:end_idx],
            fwdkin_world_xyz=self.fwdkin_world_xyz[start_idx:end_idx],
            cam_xy=self.cam_xy[start_idx:end_idx],
            metadata=self.metadata,
        )

    def __len__(self):
        return self.joint_angles.shape[0]

    def get_filtered_joint_angles(self, medkernel_size=None, ratelim=None):
        ts = self.joint_angles
        if ratelim is not None:
            ts = filter.ratelim_filter_over_time(ts, ratelim)
        if medkernel_size is not None:
            ts = filter.median_filter_over_time(ts, medkernel_size)
        return ts

    def get_filtered_fwdkin_world_xyz(self, medkernel_size=None, ratelim=None):
        filtered = self.fwdkin_world_xyz.copy()
        if ratelim is not None:
            filtered = filter.ratelim_filter_over_time(filtered, ratelim)
        if medkernel_size is not None:
            filtered = filter.median_filter_over_time(filtered, medkernel_size)
        return filtered

    def get_interpolated_joint_angles(
        self, nsteps_out, medkernel_size=None, ratelim=None
    ):
        filtered = self.get_filtered_joint_angles(
            medkernel_size=medkernel_size, ratelim=ratelim
        )
        nsteps_in = self.joint_angles.shape[0]
        in_idxs = np.arange(nsteps_in)
        out_idxs = np.arange(nsteps_out) * (nsteps_in / nsteps_out)
        f = interp1d(in_idxs, filtered, axis=0, kind="linear", fill_value="extrapolate")
        interp_angles = f(out_idxs)

        # Make a mask of shape (n_sim_steps,) indicating which simulation steps roughly
        # correspond to frames in the recording
        matching_frameids_in_rec = np.linspace(
            0, nsteps_in, nsteps_out, endpoint=False, dtype=np.int32
        )
        # Find out for which simulation steps the matching frame ID changes
        rec_match_mask = np.diff(matching_frameids_in_rec, prepend=-1) != 0

        return interp_angles, rec_match_mask


def _load_poseforge_output(
    *,
    spotlight_trial_dirs,
    min_xy_conf,
    mask_denoise_kernel_size_sec,
    min_duration_sec,
    data_fps,
):
    mask_denoise_kernel_size_steps = int(mask_denoise_kernel_size_sec * data_fps)
    min_duration_steps = int(min_duration_sec * data_fps)

    summary_df_rows = []
    data_by_idx = {}
    for trial_dir in spotlight_trial_dirs:
        poseforge_output_dir = trial_dir / "poseforge_output/"
        if not poseforge_output_dir.exists():
            raise FileNotFoundError(
                f"PoseForge output not found for trial {trial_dir.stem}. "
                "Run PoseForge production pipeline on this trial first; this would "
                f"generate a directory {poseforge_output_dir} with pose estimates."
            )
        with h5py.File(poseforge_output_dir / "keypoints3d_prediction.h5", "r") as f:
            raw_world_xyz = f["pred_world_xyz"][:]
            cam_xy = f["pred_xy"][:]
            conf_xy = f["conf_xy"][:]
            all_keypoints_order = list(f.attrs["keypoint_names"])
        with h5py.File(poseforge_output_dir / "inverse_kinematics_output.h5", "r") as f:
            # For keypoint positions, drop antannae and expand 30 to 6*5 (by leg)
            fwdkin_world_xyz = f["fwdkin_world_xyz"][:, :30, :].reshape(-1, 6, 5, 3)
            joint_angles = f["joint_angles"][:]
            legs_order = list(f["fwdkin_world_xyz"].attrs["legs"])
            keypoints_order_per_leg = list(
                kpt
                for kpt in f["fwdkin_world_xyz"].attrs["keypoint_names"]
                if kpt.startswith(legs_order[0])
            )
            # Temporary hack: change the following line to
            # `f["joint_angles"].attrs["dof_names_per_leg"]`
            # after fixing https://github.com/NeLy-EPFL/poseforge/issues/47
            dofs_order_per_leg = list(dof_name_lookup_canonical_to_nmf.keys())
            assert list(f["joint_angles"].attrs["legs"]) == legs_order

        confmask = conf_xy.mean(axis=1) > min_xy_conf
        confmask_denoised = filter.boolean_majority_filter(
            confmask, mask_denoise_kernel_size_steps
        )
        start_end_ids = filter.boolean_true_runs(confmask_denoised)
        for start, end in start_end_ids:
            if end - start < min_duration_steps:
                continue
            metadata_cols = {
                "idx": len(summary_df_rows),
                "trial": trial_dir.stem,
                "start_idx": np.int32(start),
                "end_idx": np.int32(end),
                "duration_s": np.float32((end - start) / data_fps),
            }
            summary_df_rows.append(metadata_cols)
            data_by_idx[len(summary_df_rows) - 1] = {
                "world_xyz": raw_world_xyz[start:end].astype(np.float32),
                "cam_xy": cam_xy[start:end].astype(np.float32),
                "fwdkin_world_xyz": fwdkin_world_xyz[start:end].astype(np.float32),
                "joint_angles": joint_angles[start:end].astype(np.float32),
                "mask_denoise_kernel_size_sec": mask_denoise_kernel_size_sec,
                "min_xy_conf": min_xy_conf,
                "fps": data_fps,
                "keypoints_order": all_keypoints_order,
                "legs_order": legs_order,
                "keypoints_order_per_leg": keypoints_order_per_leg,
                "dofs_order_per_leg": dofs_order_per_leg,
                **metadata_cols,
            }

    summary_df = pl.DataFrame(summary_df_rows)
    return summary_df, data_by_idx


def align_smooth_decompose_trajs(
    kinematic_snippet: KinematicsSnippet,
    sim_results: dict,
    t_range=None,
    posvelxy_sg_window_sec=0.5,
    linspeed_sg_window_sec=0.5,
    sg_window_turnrate_sec=1.0,
):
    if t_range is not None:
        start_idx_before = kinematic_snippet.start_idx
        kinematic_snippet = kinematic_snippet.get_subselection(*t_range)
        steps_offset = kinematic_snippet.start_idx - start_idx_before
    else:
        steps_offset = 0
    slice_ = slice(steps_offset, steps_offset + len(kinematic_snippet))

    dt = 1 / kinematic_snippet.data_fps
    basetraj_sim = sim_results["thorax_pos_inputmatched"][slice_]
    align_info = traj.align_traj(kinematic_snippet.thorax_pos_mm, basetraj_sim)
    basetraj_rec = align_info["traj_aligned"]
    basetraj_sim_filtered, basevelxy_sim = traj.get_denoised_traj_and_vel(
        basetraj_sim, dt, sg_window_sec=posvelxy_sg_window_sec
    )
    basetraj_rec_filtered, basevelxy_rec = traj.get_denoised_traj_and_vel(
        basetraj_rec, dt, sg_window_sec=posvelxy_sg_window_sec
    )
    origin_offset = basetraj_sim[0].copy()
    basetraj_sim -= origin_offset
    basetraj_sim_filtered -= origin_offset
    basetraj_rec -= origin_offset
    basetraj_rec_filtered -= origin_offset

    baselinspeed_sim, baseheading_sim, baseturnrate_sim = traj.get_egocentric_vel(
        basetraj_sim,
        dt,
        linspeed_sg_window_sec=linspeed_sg_window_sec,
        turnrate_sg_window_sec=sg_window_turnrate_sec,
    )
    baselinspeed_rec, baseheading_rec, baseturnrate_rec = traj.get_egocentric_vel(
        basetraj_rec,
        dt,
        linspeed_sg_window_sec=linspeed_sg_window_sec,
        turnrate_sg_window_sec=sg_window_turnrate_sec,
    )

    return {
        "basetraj_sim": basetraj_sim,
        "basetraj_sim_filtered": basetraj_sim_filtered,
        "basevelxy_sim": basevelxy_sim,
        "baselinspeed_sim": baselinspeed_sim,
        "baseheading_sim": baseheading_sim,
        "baseturnrate_sim": baseturnrate_sim,
        "basetraj_rec": basetraj_rec,
        "basetraj_rec_filtered": basetraj_rec_filtered,
        "basevelxy_rec": basevelxy_rec,
        "baselinspeed_rec": baselinspeed_rec,
        "baseheading_rec": baseheading_rec,
        "baseturnrate_rec": baseturnrate_rec,
        "steps_offset": steps_offset,
        "slice": slice_,
        "origin_offset": origin_offset,
        "rec_traj_alignment_transform": align_info,
    }


def undo_poseforge_input_transform(
    xy_posttransform: np.ndarray, transform_matrices: np.ndarray
) -> np.ndarray:
    """
    Reverses affine transforms applied to 2D point coordinates.

    Args:
        xy_posttransform:   (n_frames, ..., 2) — transformed coordinates
        transform_matrices: (n_frames, 2, 3)   — affine matrices that were applied

    Returns:
        xy_pretransform: (n_frames, ..., 2) — recovered original coordinates
    """
    n_frames = xy_posttransform.shape[0]
    original_shape = xy_posttransform.shape

    # Flatten all middle dims into a single n_points dimension
    xy_flat = xy_posttransform.reshape(n_frames, -1, 2)  # (n_frames, n_points, 2)

    # Promote each 2x3 matrix to a full 3x3 affine matrix by appending [0, 0, 1]
    bottom_row = np.tile(
        np.array([[[0.0, 0.0, 1.0]]]), (n_frames, 1, 1)
    )  # (n_frames, 1, 3)
    M_3x3 = np.concatenate([transform_matrices, bottom_row], axis=1)  # (n_frames, 3, 3)

    # Invert all matrices at once
    M_inv = np.linalg.inv(M_3x3)  # (n_frames, 3, 3)

    # Convert points to homogeneous coordinates: (n_frames, n_points, 3)
    ones = np.ones((*xy_flat.shape[:2], 1))
    xy_homogeneous = np.concatenate([xy_flat, ones], axis=-1)

    # Apply inverse transform: einsum over (frame, point, coord)
    xy_pretransform = np.einsum("fij,fpj->fpi", M_inv, xy_homogeneous)

    # Drop the homogeneous coordinate and restore the original shape
    return xy_pretransform[..., :2].reshape(original_shape)


def get_coords_arena_mm(kinematic_snippet):
    slice_ = slice(kinematic_snippet.start_idx, kinematic_snippet.end_idx)

    poseforge_output_dir = kinematic_snippet.exp_trial_dir / "poseforge_output"
    with h5py.File(poseforge_output_dir / "inverse_kinematics_output.h5", "r") as f:
        raw_world_xyz = f["rawpred_world_xyz"][slice_, :30, :].reshape(-1, 6, 5, 3)
        fwdkin_world_xyz = f["fwdkin_world_xyz"][slice_, :30, :].reshape(-1, 6, 5, 3)
    with h5py.File(poseforge_output_dir / "keypoints3d_prediction.h5", "r") as f:
        cam_xy_px = f["pred_xy"][slice_, :30, :].reshape(-1, 6, 5, 2)
    assert raw_world_xyz.shape[0] == len(kinematic_snippet)
    assert cam_xy_px.shape[0] == len(kinematic_snippet)
    assert fwdkin_world_xyz.shape[0] == len(kinematic_snippet)

    # Image was downsampled before given as input to neural network. Scale output px
    # coords back up to original input image scale
    cam_xy_px *= RAW_INPUT_CROP_DIM / POSE_WORKING_DIM
    # ... then undo alignment and cropping
    crop_transmats = kinematic_snippet.crop_transmats
    cam_xy_px_pretransform = undo_poseforge_input_transform(cam_xy_px, crop_transmats)

    mapper = kinematic_snippet.spotlight_coords_mapper
    stage_pos_mm_expanded = np.tile(
        kinematic_snippet.stage_pos_mm[:, None, None, :], (1, 6, 5, 1)
    )
    xypos_arena = mapper.stage_and_pixel_to_physical(
        stage_pos_mm_expanded, cam_xy_px_pretransform
    )

    return xypos_arena


def get_gait_info(
    sim_dir: Path,
    t_range: tuple[float, float] | None = None,
    swing_speed_thr=SWING_SPEED_THR,
    swing_speed_sg_window=SWING_SPEED_SG_WINDOW,
    swing_mask_medfilter_window=SWING_MASK_MEDFILTER_WINDOW,
):
    with open(sim_dir / "sim_data.pkl", "rb") as f:
        data = pickle.load(f)
        kinematic_snippet = data["snippet"]
    if t_range is not None:
        kinematic_snippet = kinematic_snippet.get_subselection(*t_range)

    kpts_xypos_arena = get_coords_arena_mm(kinematic_snippet)
    # kpts_xypos_arena: (n_frames, 6 legs, 5 kpts per leg, {x,y})
    claw_xypos_arena = kpts_xypos_arena[:, :, -1, :]

    dt = 1 / kinematic_snippet.data_fps
    claw_xyvel_arena = savgol_filter(
        claw_xypos_arena, swing_speed_sg_window, deriv=1, polyorder=2, delta=dt, axis=0
    )
    claw_speed = np.linalg.norm(claw_xyvel_arena, axis=-1)
    swing_mask = claw_speed > swing_speed_thr

    swing_mask = filter.median_filter_over_time(swing_mask, swing_mask_medfilter_window)

    return {
        "swing_mask": swing_mask,
        "claw_xypos_arena": claw_xypos_arena,
        "claw_speed": claw_speed,
    }
