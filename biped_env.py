import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim_motor import SimMotor
from controller import PIDController


# Joint vector order -- must match the <actuator> order in biped.xml.
JOINT_NAMES = (
    "left_hip_roll", "left_hip_pitch", "left_knee",
    "right_hip_roll", "right_hip_pitch", "right_knee",
)
N_JOINTS = len(JOINT_NAMES)

# Nominal standing pose (rev), one entry per JOINT_NAMES. The knees are slightly
# bent into a small crouch; each hip flexes by half the knee angle so the foot
# stays directly under the hip (with point feet and no ankle, bending only the
# knee would swing the foot up off the floor). The pelvis spawn height in
# biped.xml is set to match so the feet start resting on the ground.
_KNEE_BEND = -0.30            # rad; negative = bent (knee range is [-2.4, 0])
_HIP_FLEX = -_KNEE_BEND / 2   # rad; keeps each foot under its hip
_RAD2REV = 1.0 / (2.0 * math.pi)
NOMINAL_POSE = np.array([
    0.0, _HIP_FLEX * _RAD2REV, _KNEE_BEND * _RAD2REV,   # left:  roll, pitch, knee
    0.0, _HIP_FLEX * _RAD2REV, _KNEE_BEND * _RAD2REV,   # right: roll, pitch, knee
], dtype=float)

# Pelvis height (m) of the nominal stance -- matches the biped.xml "home"
# keyframe and the pelvis spawn position, so the feet rest on the floor.
NOMINAL_HEIGHT = 0.7428

# The RL action is a pose offset (rev) added to NOMINAL_POSE: a in [-1, 1] maps
# to +/- ACTION_SCALE rev per joint. Keeps the policy exploring near a stance.
ACTION_SCALE = 0.20

# Inner-loop PID gains (shared by the env and the deployed RLController so the
# cascade behaves the same in training and at run time). rev / rev/s / Nm.
INNER_KP = 100.0
INNER_KD = 10.0
INNER_KI = 0.0
INNER_INTEGRAL_LIMIT = 1.0
MAX_TORQUE = 40.0

# Control rates. The inner PID and outer (RL) loops run at independent rates;
# physics runs at SIM_HZ, chosen as a common multiple of both so each loop lands
# on an exact whole number of physics steps. 100 Hz and 30 Hz are not
# integer-related, so the number of inner ticks per outer step varies (4, 3, 3,
# ...) and averages INNER_HZ / OUTER_HZ -- the faithful behavior of two
# asynchronous loops, not an artifact.
INNER_HZ = 100   # inner PID loop rate
OUTER_HZ = 50    # outer (RL) loop rate -- one env step is one outer tick
SIM_HZ = 200     # MuJoCo physics rate (must be a common multiple of the above)
assert SIM_HZ % INNER_HZ == 0 and SIM_HZ % OUTER_HZ == 0, \
    "SIM_HZ must be an integer multiple of both INNER_HZ and OUTER_HZ"
SIM_DT = 1.0 / SIM_HZ
INNER_DT = 1.0 / INNER_HZ
OUTER_DT = 1.0 / OUTER_HZ
SIM_PER_INNER = SIM_HZ // INNER_HZ   # physics steps between inner PID updates (3)
SIM_PER_OUTER = SIM_HZ // OUTER_HZ   # physics steps per outer / env step (10)

# Termination thresholds.
MIN_HEIGHT = 0.50   # pelvis height (m) below which we count it as fallen
MAX_TILT = 0.7      # |roll| or |pitch| (rad) beyond which we count it as fallen


def read_pelvis(motor):
    """Bundle the pelvis floating-base measurements used in the observation."""
    return {
        "proj_grav": motor.projected_gravity(),
        "height": motor.get_pelvis_height(),
        "lin_vel": motor.get_pelvis_lin_vel(),
        "ang_vel": motor.get_pelvis_ang_vel(),
    }


def build_obs(pos, vel, pelvis, last_action):
    """Flat observation shared by the env and the deployed RLController.

    Layout: proj_grav(3), height(1), lin_vel(3), ang_vel(3),
    joint_pos(N_JOINTS), joint_vel(N_JOINTS), last_action(N_JOINTS).
    """
    return np.concatenate([
        pelvis["proj_grav"],
        [pelvis["height"]],
        pelvis["lin_vel"],
        pelvis["ang_vel"],
        np.asarray(pos, dtype=float),
        np.asarray(vel, dtype=float),
        np.asarray(last_action, dtype=float),
    ]).astype(np.float32)


OBS_DIM = 3 + 1 + 3 + 3 + N_JOINTS + N_JOINTS + N_JOINTS


class BipedEnv(gym.Env):
    """Gymnasium env wrapping the MuJoCo biped for RL over an inner PID.

    The policy is the outer loop, queried at OUTER_HZ: each step its action sets
    a pose offset, and an inner PID tracks NOMINAL_POSE + offset at INNER_HZ
    (cascade control), with physics at SIM_HZ. The task is to walk forward while
    staying upright.

    Action: N_JOINTS-vector in [-1, 1], scaled by ACTION_SCALE (rev) into pose
    offsets. Reward: forward velocity + alive bonus + uprightness - control
    effort - action jerk. Episode terminates on a fall (low pelvis / large tilt).
    """

    metadata = {"render_modes": ["human"], "render_fps": OUTER_HZ}

    def __init__(self, model_path="biped.xml", max_steps=300, render_mode=None):
        super().__init__()
        self.dt = OUTER_DT
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.motor = SimMotor(model_path, JOINT_NAMES)
        # Drive physics at SIM_HZ so the inner/outer rates land on exact step
        # counts regardless of the timestep written in the XML.
        self.motor.model.opt.timestep = SIM_DT

        # The model must initialize to the nominal stance; guard against the XML
        # "home" keyframe drifting from NOMINAL_POSE / NOMINAL_HEIGHT.
        self.motor.reset_state()
        assert np.allclose(self.motor.get_pos(), NOMINAL_POSE, atol=1e-3) and \
            abs(self.motor.get_pelvis_height() - NOMINAL_HEIGHT) < 1e-3, \
            "biped.xml 'home' keyframe does not match NOMINAL_POSE/NOMINAL_HEIGHT"

        self.inner = PIDController(
            target=NOMINAL_POSE.copy(), kp=INNER_KP, kd=INNER_KD, ki=INNER_KI,
            dt=INNER_DT, integral_limit=INNER_INTEGRAL_LIMIT,
            max_torque=MAX_TORQUE, n_joints=N_JOINTS,
        )

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)

        self._step_count = 0
        self._sim_tick = 0
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float32)
        self._viewer = None

    def _obs(self):
        return build_obs(self.motor.get_pos(), self.motor.get_vel(),
                         read_pelvis(self.motor), self._prev_action)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Start from the standing pose with a small random perturbation.
        pose = NOMINAL_POSE + self.np_random.uniform(-0.02, 0.02, size=N_JOINTS)
        vel = self.np_random.uniform(-0.05, 0.05, size=N_JOINTS)
        self.motor.reset_state(pose=pose, vel=vel)
        self.inner.reset()
        self._step_count = 0
        self._sim_tick = 0
        self._prev_action = np.zeros(N_JOINTS, dtype=np.float32)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # Outer loop (OUTER_HZ): one env step = one outer tick. This action sets
        # the inner PID's pose reference.
        self.inner.target = NOMINAL_POSE + action * ACTION_SCALE

        # Advance one outer period of physics. The inner PID recomputes its
        # torque every SIM_PER_INNER steps (INNER_HZ) and holds it in between;
        # physics steps at SIM_HZ. The global _sim_tick keeps inner ticks evenly
        # spaced across outer-step boundaries (so they stay at exactly INNER_HZ).
        torque = np.zeros(N_JOINTS)
        for _ in range(SIM_PER_OUTER):
            if self._sim_tick % SIM_PER_INNER == 0:
                torque = self.inner.compute(self.motor.get_pos(),
                                            self.motor.get_vel())
                self.motor.set_torque(torque)
            self.motor.step_n(1)
            self._sim_tick += 1

        pelvis = read_pelvis(self.motor)
        roll, pitch, _ = self.motor.get_pelvis_rpy()
        fwd_vel = float(pelvis["lin_vel"][0])
        height = pelvis["height"]

        d_action = action - self._prev_action
        self._prev_action = action

        # Curriculum: learn to BALANCE first (stay upright at nominal height,
        # without flailing) with only a small forward-speed incentive; a walking
        # gait can then emerge. Point feet have no support polygon, so the
        # upright term must carry a smooth gradient -- it rewards how vertical the
        # pelvis is (proj_grav_z = -1 upright, -> 0 tipped over).
        upright = -float(pelvis["proj_grav"][2])     # +1 upright, -> 0 on its side
        height_err = height - NOMINAL_HEIGHT
        ang_vel_sq = float(np.sum(pelvis["ang_vel"] ** 2))
        reward = (
            0.5                                      # alive bonus
            + 1.0 * upright                          # stay upright (smooth)
            - 3.0 * height_err ** 2                  # hold nominal height
            - 0.02 * ang_vel_sq                      # damp body rotation
            + 0.1 * fwd_vel                          # small forward incentive
            - 0.001 * float(np.sum(torque ** 2))     # control effort
            - 0.01 * float(np.sum(d_action ** 2))    # action smoothness
        )

        fallen = (height < MIN_HEIGHT or abs(roll) > MAX_TILT
                  or abs(pitch) > MAX_TILT)
        self._step_count += 1
        terminated = bool(fallen)
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return
        import mujoco.viewer
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(
                self.motor.model, self.motor.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
