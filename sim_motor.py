import math

import numpy as np
import mujoco


class SimMotor:
    """MuJoCo stand-in for the biped's joint motors.

    Units match the moteus convention used across these projects:
      - position in revolutions (MuJoCo internally uses radians)
      - velocity in rev/s
      - torque in Nm

    Unlike the single-/dual-motor balance_bot wrapper this drives an arbitrary
    list of joints, so get_pos/get_vel/set_torque all work on numpy arrays in
    the JOINT_NAMES order. The pelvis freejoint stands in for an IMU + mocap:
    helpers expose its height, attitude, and velocities for the RL observation.
    """

    def __init__(self, model_path, joint_names, actuator_names=None,
                 root_joint_name="root"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        if actuator_names is None:
            actuator_names = joint_names
        self.joint_names = list(joint_names)
        self.n = len(self.joint_names)

        self.actuator_ids = []
        for name in actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"actuator '{name}' not found in {model_path}")
            self.actuator_ids.append(aid)
        self.actuator_ids = np.array(self.actuator_ids)

        qpos_addrs, qvel_addrs = [], []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint '{name}' not found in {model_path}")
            qpos_addrs.append(self.model.jnt_qposadr[jid])
            qvel_addrs.append(self.model.jnt_dofadr[jid])
        self.qpos_addrs = np.array(qpos_addrs)
        self.qvel_addrs = np.array(qvel_addrs)

        # Pelvis freejoint -- a sim stand-in for the floating-base IMU/mocap.
        self._root_qpos = None
        self._root_qvel = None
        if root_joint_name is not None:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
            if jid >= 0:
                self._root_qpos = self.model.jnt_qposadr[jid]
                self._root_qvel = self.model.jnt_dofadr[jid]

        # Optional "home" keyframe -- the model's nominal initial state (pose +
        # base height). When present, reset_state initializes to it so the sim
        # starts in the nominal stance instead of the bare model default.
        self._home_key = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self._home_key < 0:
            self._home_key = None

        self._torque = np.zeros(self.n)

    @property
    def sim_dt(self):
        return self.model.opt.timestep

    def reset_state(self, pose=None, vel=None):
        """Reset the sim to the nominal initial state (the 'home' keyframe if the
        model defines one, else the bare default). `pose`/`vel` are optional joint
        overrides (rev, rev/s) in JOINT_NAMES order applied on top; the freejoint
        base keeps the keyframe's pose/height."""
        if self._home_key is not None:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        if pose is not None:
            self.data.qpos[self.qpos_addrs] = np.asarray(pose) * 2.0 * math.pi
        if vel is not None:
            self.data.qvel[self.qvel_addrs] = np.asarray(vel) * 2.0 * math.pi
        self._torque = np.zeros(self.n)
        mujoco.mj_forward(self.model, self.data)

    def set_torque(self, torque):
        self._torque = np.asarray(torque, dtype=float)

    def step_n(self, n):
        """Advance the sim by n physics steps, holding the current torque."""
        for _ in range(n):
            self.data.ctrl[self.actuator_ids] = self._torque
            mujoco.mj_step(self.model, self.data)

    def get_pos(self):
        return self.data.qpos[self.qpos_addrs] / (2.0 * math.pi)

    def get_vel(self):
        return self.data.qvel[self.qvel_addrs] / (2.0 * math.pi)

    def stop(self):
        self._torque = np.zeros(self.n)
        self.data.ctrl[self.actuator_ids] = 0.0

    # ---- pelvis / floating-base helpers (sim stand-in for an IMU) ----------

    def _require_root(self):
        if self._root_qpos is None:
            raise RuntimeError(
                "no pelvis freejoint found -- pass root_joint_name=... or add a "
                "<freejoint> to the pelvis")

    def get_pelvis_height(self):
        self._require_root()
        return float(self.data.qpos[self._root_qpos + 2])

    def get_pelvis_quat(self):
        """Pelvis orientation quaternion (w, x, y, z)."""
        self._require_root()
        a = self._root_qpos
        return self.data.qpos[a + 3:a + 7].copy()

    def get_pelvis_rpy(self):
        """Pelvis attitude as Euler angles (roll, pitch, yaw) in rad."""
        qw, qx, qy, qz = self.get_pelvis_quat()
        roll = math.atan2(2.0 * (qw * qx + qy * qz),
                          1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))
        return (roll, pitch, yaw)

    def get_pelvis_lin_vel(self):
        """Pelvis linear velocity (world frame, m/s)."""
        self._require_root()
        a = self._root_qvel
        return self.data.qvel[a:a + 3].copy()

    def get_pelvis_ang_vel(self):
        """Pelvis angular velocity (body frame, rad/s)."""
        self._require_root()
        a = self._root_qvel
        return self.data.qvel[a + 3:a + 6].copy()

    def projected_gravity(self):
        """Gravity direction expressed in the pelvis frame (unit vector).

        Reads (0, 0, -1) when perfectly upright and tilts as the pelvis leans --
        an orientation cue an onboard IMU can produce directly, so it ports to
        hardware without needing absolute yaw."""
        qw, qx, qy, qz = self.get_pelvis_quat()
        # world -> body rotation of the down vector (0, 0, -1)
        gx = -2.0 * (qx * qz - qw * qy)
        gy = -2.0 * (qy * qz + qw * qx)
        gz = -(1.0 - 2.0 * (qx * qx + qy * qy))
        return np.array([gx, gy, gz])
