import numpy as np
from tqdm import trange

import mujoco
from scipy.spatial.transform import Rotation as R

from flygym import assets_dir as flygym_assets_dir
from flygym import Simulation
from flygym.compose import (
    Fly,
    FlatGroundWorld,
    KinematicPose,
    ContactParams,
    ActuatorType,
)
from flygym.anatomy import (
    Skeleton,
    BodySegment,
    JointPreset,
    ActuatedDOFPreset,
    AxisOrder,
    RotationAxis,
)
from flygym.utils.math import Rotation3D

from sppaper.kinematics.data import KinematicsSnippet

# Constants
PASSIVE_TARSUS_STIFFNESS = 10
PASSIVE_TARSUS_DAMPING = 0.5
ARTICULATED_JOINTS = JointPreset.LEGS_ONLY
ACTUATED_DOFS = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
AXIS_ORDER = AxisOrder.YAW_PITCH_ROLL
ACTUATOR_TYPE = ActuatorType.POSITION
NEUTRAL_POSE_FILE = flygym_assets_dir / "model/pose/neutral.yaml"
SPAWN_POSITION = (0, 0, 0.7)  # mm
SPAWN_ROTATION = Rotation3D("quat", (1, 0, 0, 0))
VIDEO_PLAYBACK_SPEED = 0.1
VIDEO_OUTPUT_FPS = 33
CAM_RES = 1440


class NeuroMechFlyReplayManager:
    def __init__(self, sample_invkin_snippet: KinematicsSnippet):
        self.dof_idxmap_rec2sim, self.dof_mirror_mask = (
            self._get_dof_mapping_and_mirror_mask(sample_invkin_snippet)
        )

    def create_sim(
        self,
        actuator_gain: float,
        joint_damping: float,
        sliding_friction: float,
    ) -> "NeuroMechFlyReplayInstance":
        sim, fly = self._setup_sim_and_fly(
            actuator_gain,
            joint_damping,
            sliding_friction,
        )
        return NeuroMechFlyReplayInstance(
            sim, fly, self.dof_idxmap_rec2sim, self.dof_mirror_mask
        )

    def _setup_sim_and_fly(
        self,
        actuator_gain: float,
        joint_damping: float,
        sliding_friction: float,
    ) -> tuple[Simulation, Fly]:
        fly = Fly(simplify_claw=True)
        skeleton = Skeleton(axis_order=AXIS_ORDER, joint_preset=ARTICULATED_JOINTS)
        neutral_pose = KinematicPose(path=NEUTRAL_POSE_FILE)
        fly.add_joints(
            skeleton,
            neutral_pose=neutral_pose,
            damping=joint_damping,
        )
        n_tarsus_overrides = 0
        for dof, mjcf_joint in fly.jointdof_to_mjcfjoint.items():
            if dof.child.link in ("tarsus2", "tarsus3", "tarsus4", "tarsus5"):
                mjcf_joint.stiffness = PASSIVE_TARSUS_STIFFNESS
                mjcf_joint.damping = PASSIVE_TARSUS_DAMPING
                n_tarsus_overrides += 1
        assert n_tarsus_overrides == 4 * 6, "error overriding tarsus joint params"

        # Add position actuators and set them to the neutral pose
        actuated_dofs_list = fly.skeleton.get_actuated_dofs_from_preset(ACTUATED_DOFS)
        fly.add_actuators(
            actuated_dofs_list,
            actuator_type=ACTUATOR_TYPE,
            kp=actuator_gain,
            neutral_input=neutral_pose,
        )

        # Add visuals
        fly.colorize()
        tracking_cam = fly.add_tracking_camera(fovy=50)
        fly.mjcf_root.visual.get_children("global").offwidth = CAM_RES
        fly.mjcf_root.visual.get_children("global").offheight = CAM_RES

        # Create a world and spawn the fly
        world = FlatGroundWorld()
        world.ground_geom.rgba = (1, 1, 1, 0.75)
        contact_params = ContactParams(sliding_friction=sliding_friction)
        world.add_fly(
            fly, SPAWN_POSITION, SPAWN_ROTATION, ground_contact_params=contact_params
        )
        bottom_cam = world.mjcf_root.worldbody.add(
            "camera",
            name="bottom_cam",
            pos=(0, 0, -100),
            quat=(0, 1, 0, 0),
            fovy=5,
            mode="trackcom",
            target="nmf/c_thorax",
        )

        # Create a simulation and set up the renderer
        sim = Simulation(world)
        sim.set_renderer(
            [tracking_cam, bottom_cam],
            playback_speed=VIDEO_PLAYBACK_SPEED,
            output_fps=VIDEO_OUTPUT_FPS,
            camera_res=[CAM_RES, CAM_RES],
        )

        return sim, fly

    def _get_dof_mapping_and_mirror_mask(self, sample_invkin_snippet):
        """Match DoFs between recording and simulation based on their conventions.

        Args:
            sample_invkin_snippet:
                An example snippet from the dataset (just to get DoF ordering info).

        Returns:
            dof_idxmap_rec2sim:
                Array mapping simulation DoF indices to recording DoF indices.
            dof_mirror_mask:
                Boolean array indicating which DoFs need sign flip due to convention
                differences between FlyGym and SeqIKPy. FlyGym uses an "anatomical"
                convention for roll and yaw axis (for example, positive roll is always
                outward regardless of left vs. right), while SeqIKPy uses the geometric
                convention  corresponding to global axes. Therefore, for roll and yaw
                axes on the right side, we need a sign flip. Important: this mask is in
                simulation DoF order, so it should be applied *after* column reordering.
        """
        # Make a dummy simulation to get DoF ordering info - physics params don't matter
        ref_sim, ref_fly = self._setup_sim_and_fly(0, 0, 0)
        actuated_dofs_order = ref_fly.get_actuated_jointdofs_order(ACTUATOR_TYPE)
        n_actuators = len(actuated_dofs_order)
        dof_idxmap_rec2sim = np.full(n_actuators, dtype=np.int32, fill_value=-1)
        dof_mirror_mask = np.zeros(n_actuators, dtype=bool)
        for i, dof in enumerate(actuated_dofs_order):
            # Find corresponding dof in the recording based on joint name matching, and
            # save the index mapping
            flygym_dofname = dof.name
            seqikpy_dofname = joint_name_flygym2seqikpy(flygym_dofname)
            rec_dof_idx = sample_invkin_snippet.metadata["joints_order"].index(
                seqikpy_dofname
            )
            dof_idxmap_rec2sim[i] = rec_dof_idx
            if dof.child.pos[0] == "r" and (
                dof.axis in (RotationAxis.ROLL, RotationAxis.YAW)
            ):
                dof_mirror_mask[i] = True
        assert np.all(dof_idxmap_rec2sim >= 0), "error in dof name matching"
        expected_nflips = 3 * 3  # 3 legs * (thorax-coxa roll+yaw, coxa-trochanter roll)
        assert dof_mirror_mask.sum() == expected_nflips, "error in mirror mask"

        return dof_idxmap_rec2sim, dof_mirror_mask


class NeuroMechFlyReplayInstance:
    def __init__(
        self,
        sim: Simulation,
        fly: Fly,
        dof_idxmap_rec2sim: np.ndarray,
        dof_mirror_mask: np.ndarray,
    ) -> None:
        self.sim = sim
        self.fly = fly
        self.dof_idxmap_rec2sim = dof_idxmap_rec2sim
        self.dof_mirror_mask = dof_mirror_mask
        self.bodysegs_order = fly.get_bodysegs_order()
        self.actuated_dofs_order = fly.get_actuated_jointdofs_order(ACTUATOR_TYPE)

    def replay_invkin_snippet(
        self,
        kinematics_snippet: KinematicsSnippet,
        preinterp_medkernel_size: int | None = None,
        preinterp_ratelim: float | None = None,
    ):
        """Replay a kinematic snippet in the simulation.

        Args:
            kinematics_snippet: The kinematic data to replay.
            preinterp_medkernel_size: Optional median filter kernel size to apply
                before interpolation.
            preinterp_ratelim: Optional rate limit to apply before interpolation.

        Returns:
            sim_results: Dictionary containing simulation results.
        """
        # Interpolate recorded joint angles to match simulation timestep
        nsteps_sim = int(kinematics_snippet.duration_s / self.sim.mj_model.opt.timestep)
        joint_angles_arr, ctrl_update_mask = (
            kinematics_snippet.get_interpolated_joint_angles(
                nsteps_out=nsteps_sim,
                medkernel_size=preinterp_medkernel_size,
                ratelim=preinterp_ratelim,
            )
        )

        # Reorder and mirror columns to match simulation DoF ordering and conventions
        joint_angles_arr = joint_angles_arr[:, self.dof_idxmap_rec2sim]
        joint_angles_arr[:, self.dof_mirror_mask] *= -1

        # Initialize simulation and set up some buffers
        self.sim.reset()
        fly_name = self.fly.name
        self.sim.set_actuator_inputs(fly_name, ACTUATOR_TYPE, joint_angles_arr[0, :])
        self.sim.warmup()
        body_pos_hist = np.full(
            (nsteps_sim, len(self.bodysegs_order), 3), np.nan, dtype=np.float32
        )
        body_rot_hist = np.full(
            (nsteps_sim, len(self.bodysegs_order), 4), np.nan, dtype=np.float32
        )
        actuator_forces_hist = np.full(
            (nsteps_sim, len(self.actuated_dofs_order)), np.nan, dtype=np.float32
        )
        contact_active_hist = np.full((nsteps_sim, 6), False, dtype=bool)
        contact_forces_hist = np.full((nsteps_sim, 6, 3), np.nan, dtype=np.float32)
        contact_torques_hist = np.full((nsteps_sim, 6, 3), np.nan, dtype=np.float32)
        contact_pos_hist = np.full((nsteps_sim, 6, 3), np.nan, dtype=np.float32)
        contact_normals_hist = np.full((nsteps_sim, 6, 3), np.nan, dtype=np.float32)
        contact_tangents_hist = np.full((nsteps_sim, 6, 3), np.nan, dtype=np.float32)

        # Run simulation loop
        for step in trange(nsteps_sim):
            target_angles = joint_angles_arr[step, :]
            self.sim.set_actuator_inputs(fly_name, ACTUATOR_TYPE, target_angles)
            self.sim.step()

            body_pos_hist[step, :, :] = self.sim.get_body_positions(fly_name).copy()
            body_rot_hist[step, :, :] = self.sim.get_body_rotations(fly_name).copy()
            actuator_forces_hist[step, :] = self.sim.get_actuator_forces(
                fly_name, ACTUATOR_TYPE
            ).copy()
            ground_contact_info = self.sim.get_ground_contact_info(fly_name)
            contact_active_hist[step, :] = ground_contact_info[0] > 0
            contact_forces_hist[step, :, :] = ground_contact_info[1]
            contact_torques_hist[step, :, :] = ground_contact_info[2]
            contact_pos_hist[step, :, :] = ground_contact_info[3]
            contact_normals_hist[step, :, :] = ground_contact_info[4]
            contact_tangents_hist[step, :, :] = ground_contact_info[5]

            self.sim.render_as_needed()

        sim_results = self._postprocess_replay_results(
            body_pos_hist,
            body_rot_hist,
            actuator_forces_hist,
            contact_active_hist,
            contact_forces_hist,
            contact_torques_hist,
            contact_pos_hist,
            contact_normals_hist,
            contact_tangents_hist,
            ctrl_update_mask,
        )

        return sim_results

    def _postprocess_replay_results(
        self,
        body_pos_hist,
        body_rot_hist,
        actuator_forces_hist,
        contact_active_hist,
        contact_forces_hist,
        contact_torques_hist,
        contact_pos_hist,
        contact_normals_hist,
        contact_tangents_hist,
        ctrl_update_mask,
    ):
        # Base trajectory
        body_pos_hist = body_pos_hist.copy()
        thorax_idx = self.bodysegs_order.index(BodySegment("c_thorax"))
        thorax_pos_inputmatched = body_pos_hist[ctrl_update_mask, thorax_idx, :2]
        thorax_pos_inputmatched -= thorax_pos_inputmatched[0, :]
        thorax_rot_inputmatched = R.from_quat(
            body_rot_hist[ctrl_update_mask, thorax_idx, :], scalar_first=True
        )
        forward_vec_inputmatched = thorax_rot_inputmatched.apply(np.array([1, 0, 0]))
        heading_inputmatched = np.arctan2(
            forward_vec_inputmatched[:, 1], forward_vec_inputmatched[:, 0]
        )

        # Forces applied by actuators
        actuator_forces_hist = np.stack(actuator_forces_hist, axis=0)

        # Contact info
        contact_forces_hist[~contact_active_hist, :] = np.nan
        contact_torques_hist[~contact_active_hist, :] = np.nan
        contact_normals_hist[~contact_active_hist, :] = np.nan
        contact_tangents_hist[~contact_active_hist, :] = np.nan
        contact_forces_mag_hist = np.linalg.norm(contact_forces_hist, axis=-1)
        contact_forces_world_hist = _vec_local2global(
            contact_forces_hist, contact_normals_hist, contact_tangents_hist
        )
        contact_torques_mag_hist = np.linalg.norm(contact_torques_hist, axis=-1)
        contact_torques_world_hist = _vec_local2global(
            contact_torques_hist, contact_normals_hist, contact_tangents_hist
        )
        contact_hist = {
            "active_mask": contact_active_hist,
            "forces_magnitude": contact_forces_mag_hist,
            "forces_world": contact_forces_world_hist,
            "torques_magnitude": contact_torques_mag_hist,
            "torques_world": contact_torques_world_hist,
            "pos": contact_pos_hist,
        }

        sim_results = {
            "body_positions": body_pos_hist,
            "body_rotations": body_rot_hist,
            "bodyseg_order": self.bodysegs_order,
            "actuator_forces": actuator_forces_hist,
            "actuated_dofs_order": self.actuated_dofs_order,
            "ctrl_update_mask": ctrl_update_mask,
            "thorax_pos_inputmatched": thorax_pos_inputmatched,
            "heading_inputmatched": heading_inputmatched,
            "ground_contacts": contact_hist,
            "sim_timestep": self.sim.mj_model.opt.timestep,
        }
        return sim_results


def joint_name_seqikpy2flygym(seqikpy_name):
    side = seqikpy_name[0]
    legpos = seqikpy_name[1]
    if side not in "LR" or legpos not in "FMH":
        raise ValueError(f"Unexpected SeqIKPy joint name {seqikpy_name}")

    flygym_leg = f"{side}{legpos}".lower()
    seqikpy_anatomical_joint = seqikpy_name.split("_")[0][2:]
    if seqikpy_anatomical_joint == "ThC":
        parent = "c_thorax"
        child = f"{flygym_leg}_coxa"
    elif seqikpy_anatomical_joint == "CTr":
        parent = f"{flygym_leg}_coxa"
        child = f"{flygym_leg}_trochanterfemur"
    elif seqikpy_anatomical_joint == "FTi":
        parent = f"{flygym_leg}_trochanterfemur"
        child = f"{flygym_leg}_tibia"
    elif seqikpy_anatomical_joint == "TiTa":
        parent = f"{flygym_leg}_tibia"
        child = f"{flygym_leg}_tarsus1"
    else:
        raise ValueError(f"Unknown SeqIKPy anatomical joint {seqikpy_anatomical_joint}")

    axis = seqikpy_name.split("_")[1]
    if axis not in ("pitch", "roll", "yaw"):
        raise ValueError(f"Unexpected axis {axis} in SeqIKPy joint name {seqikpy_name}")

    flygym_joint_name = f"{parent}-{child}-{axis}"
    return flygym_joint_name


def joint_name_flygym2seqikpy(flygym_name):
    parent, child, axis = flygym_name.split("-")
    leg = child[:2].upper()
    if parent == "c_thorax" and child.endswith("_coxa"):
        anatomical_joint = "ThC"
    elif parent.endswith("_coxa") and child.endswith("_trochanterfemur"):
        anatomical_joint = "CTr"
    elif parent.endswith("_trochanterfemur") and child.endswith("_tibia"):
        anatomical_joint = "FTi"
    elif parent.endswith("_tibia") and child.endswith("_tarsus1"):
        anatomical_joint = "TiTa"
    else:
        raise ValueError(f"Unexpected flygym joint name {flygym_name}")

    seqikpy_name = f"{leg}{anatomical_joint}_{axis}"
    assert (
        joint_name_seqikpy2flygym(seqikpy_name) == flygym_name
    ), "error in joint name conversion"
    return seqikpy_name


def _vec_local2global(vec_local, frameaxis1, frameaxis2):
    input_dim = vec_local.shape
    if input_dim[-1] != 3:
        raise ValueError(f"Last dimension of vec_local must be 3, got {input_dim[-1]}")
    if not frameaxis1.shape == frameaxis2.shape == input_dim:
        raise ValueError("vec_local, frameaxis1, and frameaxis2 have different shapes")
    vec_local = vec_local.reshape(-1, 3)
    frameaxis1 = frameaxis1.reshape(-1, 3)
    frameaxis2 = frameaxis2.reshape(-1, 3)
    vec_world = np.full_like(vec_local, np.nan)

    nanmask = np.isnan(vec_local).any(axis=-1)
    vec_local = vec_local[~nanmask, :]
    frameaxis1 = frameaxis1[~nanmask, :]
    frameaxis2 = frameaxis2[~nanmask, :]

    if not np.allclose(np.sum(frameaxis1 * frameaxis2, axis=1), 0):
        raise ValueError("frameaxis1 and frameaxis2 are not orthogonal")

    # frameaxis* are all (N, 3)
    frameaxis1 = frameaxis1 / np.linalg.norm(frameaxis1, axis=1, keepdims=True)
    frameaxis2 = frameaxis2 / np.linalg.norm(frameaxis2, axis=1, keepdims=True)
    frameaxis3 = np.cross(frameaxis1, frameaxis2)
    # rotation matrix is stacked to (N, 3, 3)
    rot_matrices = np.stack([frameaxis1, frameaxis2, frameaxis3], axis=2)
    vec_world_nonan = np.einsum("nij,nj->ni", rot_matrices, vec_local)

    vec_world[~nanmask, :] = vec_world_nonan
    return vec_world.reshape(input_dim)
