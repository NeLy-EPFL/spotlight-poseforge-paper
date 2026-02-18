from pathlib import Path

import numpy as np
import polars as pl
import h5py

from wc26.kinematics import config
from wc26.common.filter import boolean_majority_filter, boolean_true_runs

if __name__ == "__main__":
    # Extract walking kinematics from keypoints3d model output and subsequent
    # inverse kinematics via SeqIKPy
    rows = []
    data_by_idx = {}
    for trial_dir in sorted(config.KPT_3D_OUTPUT_BASEDIR.iterdir()):
        with h5py.File(trial_dir / "keypoints3d.h5", "r") as f:
            raw_world_xyz = f["keypoints_world_xyz"][:]
            conf_xy = f["keypoints_camera_xy_conf"][:]
            all_keypoints_order = list(f["keypoints_world_xyz"].attrs["keypoints"])
        with h5py.File(trial_dir / "inverse_kinematics.h5", "r") as f:
            fwdkin_world_xyz = f["fwdkin_world_xyz"][:]
            joint_angles = f["joint_angles"][:]
            keypoints_order_per_leg = list(
                f["fwdkin_world_xyz"].attrs["keypoint_names_per_leg"]
            )
            legs_order = list(f["fwdkin_world_xyz"].attrs["legs"])
            dofs_order_per_leg = list(f["joint_angles"].attrs["dof_names_per_leg"])
            assert list(f["joint_angles"].attrs["legs"]) == legs_order

        confmask = conf_xy.mean(axis=1) > config.MIN_XY_CONF
        confmask_denoised = boolean_majority_filter(
            confmask, config.MASK_DENOISE_KERNEL_SIZE_STEPS
        )
        start_end_ids = boolean_true_runs(confmask_denoised)
        for start, end in start_end_ids:
            if end - start < config.MIN_DURATION_STEPS:
                continue
            idx = len(rows)
            metadata_cols = {
                "idx": len(rows),
                "trial": trial_dir.stem,
                "start_idx": np.int32(start),
                "end_idx": np.int32(end),
                "duration_s": np.float32((end - start) / config.DATA_FPS),
            }
            rows.append(metadata_cols)
            data_by_idx[len(rows) - 1] = {
                "world_xyz": raw_world_xyz[start:end].astype(np.float32),
                "fwdkin_world_xyz": fwdkin_world_xyz[start:end].astype(np.float32),
                "joint_angles": joint_angles[start:end].astype(np.float32),
                "denoise_kernel_size_sec": config.MASK_DENOISE_KERNEL_SIZE_SEC,
                "min_xy_conf": config.MIN_XY_CONF,
                "fps": config.DATA_FPS,
                "keypoints_order": all_keypoints_order,
                "legs_order": legs_order,
                "keypoints_order_per_leg": keypoints_order_per_leg,
                "dofs_order_per_leg": dofs_order_per_leg,
                **metadata_cols,
            }

    # Save extracted walking periods and kinematics data
    walking_periods_df = pl.DataFrame(rows)
    config.WALKING_PERIODS_DATAFRAME_PATH.parent.mkdir(exist_ok=True, parents=True)
    walking_periods_df.write_csv(config.WALKING_PERIODS_DATAFRAME_PATH)

    for idx, data in data_by_idx.items():
        path = Path(str(config.KINEMATIC_DATA_PATH_FMT).format(idx=idx))
        path.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(path, **data)
