from dataclasses import dataclass

import numpy as np
import polars as pl
import h5py
from scipy.interpolate import interp1d

from spotlight_tools.calibration.mapper import SpotlightPositionMapper

import poseforge.neuromechfly.constants as nmfconst

import sppaper.common.filter as filter
import sppaper.kinematics.trajectory as traj
from sppaper.common.resources import get_spotlight_trials_dir


class KinematicsDataset:
    def __init__(
        self,
        *,
        poseforge_output_dirs,
        min_xy_conf,
        mask_denoise_kernel_size_sec,
        min_duration_sec,
        data_fps,
    ):
        self.summary_df, self.data_by_idx = _load_poseforge_output(
            poseforge_output_dirs=poseforge_output_dirs,
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
                camera_xy=data["camera_xy"],
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
    camera_xy: np.ndarray
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
        stage_pos_mm = frame_metadata.select(
            "x_pos_mm_interp", "y_pos_mm_interp"
        ).to_numpy()

        # Load camera calibration parameters and map
        # (translation stage pos, pixel corrds) to physical coordinates in the arena
        self.thorax_pos_mm = self.spotlight_coords_mapper.stage_and_pixel_to_physical(
            stage_pos_mm, self.thorax_pos_px
        )
        self.neck_pos_mm = self.spotlight_coords_mapper.stage_and_pixel_to_physical(
            stage_pos_mm, self.neck_pos_px
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
    poseforge_output_dirs,
    min_xy_conf,
    mask_denoise_kernel_size_sec,
    min_duration_sec,
    data_fps,
):
    mask_denoise_kernel_size_steps = int(mask_denoise_kernel_size_sec * data_fps)
    min_duration_steps = int(min_duration_sec * data_fps)

    summary_df_rows = []
    data_by_idx = {}
    for poseforge_output_dir in sorted(poseforge_output_dirs):
        with h5py.File(poseforge_output_dir / "inverse_kinematics_output.h5", "r") as f:
            fwdkin_world_xyz = f["fwdkin_world_xyz"][:]
            joint_angles = f["joint_angles"][:]
            kpts_order_per_leg = list(f["fwdkin_world_xyz"].attrs["keypoint_names"])
            dofs_order_per_leg = list(nmfconst.dof_name_lookup_canonical_to_nmf.keys())
            legs_order = list(f["fwdkin_world_xyz"].attrs["legs"])
        
        with h5py.File(poseforge_output_dir / "keypoints3d_prediction.h5", "r") as f:
            keypoints_camera_xy = f["pred_xy"][:]
            raw_world_xyz = f["pred_world_xyz"][:]
            conf_xy = f["conf_xy"][:]
            all_keypoints_order = list(f.attrs["keypoint_names"])

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
                "trial": poseforge_output_dir.parent.stem,
                "start_idx": np.int32(start),
                "end_idx": np.int32(end),
                "duration_s": np.float32((end - start) / data_fps),
            }
            summary_df_rows.append(metadata_cols)
            data_by_idx[len(summary_df_rows) - 1] = {
                "world_xyz": raw_world_xyz[start:end].astype(np.float32),
                "camera_xy": keypoints_camera_xy[start:end].astype(np.float32),
                "fwdkin_world_xyz": fwdkin_world_xyz[start:end].astype(np.float32),
                "joint_angles": joint_angles[start:end].astype(np.float32),
                "mask_denoise_kernel_size_sec": mask_denoise_kernel_size_sec,
                "min_xy_conf": min_xy_conf,
                "fps": data_fps,
                "keypoints_order": all_keypoints_order,
                "legs_order": legs_order,
                "keypoints_order_per_leg": kpts_order_per_leg,
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
