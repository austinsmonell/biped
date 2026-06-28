import numpy as np


class PIDController:
    """Per-joint position PID with derivative-on-measurement and integral clamp.

    The biped's inner loop: drives every joint to a target pose and holds it.
    `target`, `kp`, `kd`, `ki` may be scalars (shared by all joints) or arrays
    (one per joint); everything broadcasts over the joint vector. I/O is in the
    motor convention (rev, rev/s, Nm), matching SimMotor.
    """

    def __init__(self, target, kp, kd, ki, dt, integral_limit, max_torque,
                 n_joints=None):
        self.target = np.asarray(target, dtype=float)
        n = n_joints if n_joints is not None else self.target.size
        self.kp = np.broadcast_to(np.asarray(kp, float), (n,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, float), (n,)).copy()
        self.ki = np.broadcast_to(np.asarray(ki, float), (n,)).copy()
        self.dt = dt
        self.integral_limit = integral_limit
        self.max_torque = max_torque
        self.integral = np.zeros(n)

    def compute(self, pos, vel, pelvis=None):
        """torque (Nm) for each joint. `pelvis` is accepted but ignored so the
        inner PID is interchangeable with the RL controller in the sim loop."""
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        err = self.target - pos
        self.integral = np.clip(self.integral + err * self.dt,
                                -self.integral_limit, self.integral_limit)
        u = self.kp * err + self.ki * self.integral - self.kd * vel
        return np.clip(u, -self.max_torque, self.max_torque)

    def update_outer(self, pos, vel, pelvis=None):
        """No-op: a standalone PID has no outer loop, its target is fixed.
        Present so the run loop can drive a PID and an RL cascade identically."""
        pass

    def reset(self):
        self.integral = np.zeros_like(self.integral)

    @property
    def target_pos(self):
        return self.target


class RLController:
    """RL outer loop over an inner PID (cascade), matching the biped_env setup.

    Scheduling lives in the run loop (main.run_in_sim): update_outer() is called
    at the outer rate (OUTER_HZ) to query the PPO policy and set the inner PID's
    pose reference; compute() is called at the inner rate (INNER_HZ) to track it.
    Same (update_outer, compute) interface as PIDController, so the loop drives
    either one identically. Train the policy with train_ppo.py.
    """

    def __init__(self, inner_pid, model_path):
        import os
        import pickle
        from stable_baselines3 import PPO
        # Imported here to keep the inner-loop PID import-light and avoid a
        # circular import (biped_env imports PIDController from this module).
        from biped_env import NOMINAL_POSE, ACTION_SCALE, build_obs

        self.inner = inner_pid
        self.model = PPO.load(model_path)
        self._nominal = NOMINAL_POSE
        self._action_scale = ACTION_SCALE
        self._build_obs = build_obs
        self._last_action = np.zeros(NOMINAL_POSE.size, dtype=np.float32)

        # Load the VecNormalize stats saved by train_ppo so the policy sees the
        # same whitened observations it trained on. Without this the policy gets
        # raw obs and behaves nonsensically. (Reward normalization is irrelevant
        # at deployment, so only obs_rms is used.)
        self._obs_rms = None
        stats = model_path + "_vecnormalize.pkl"
        if os.path.exists(stats):
            with open(stats, "rb") as f:
                vn = pickle.load(f)
            self._obs_rms = vn.obs_rms
            self._clip_obs = vn.clip_obs
            self._epsilon = vn.epsilon
        else:
            print(f"RLController: no {stats} found -- using raw (unnormalized) "
                  "observations; retrain with train_ppo.py to generate stats")

    def _normalize(self, obs):
        if self._obs_rms is None:
            return obs
        norm = (obs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + self._epsilon)
        return np.clip(norm, -self._clip_obs, self._clip_obs).astype(np.float32)

    def update_outer(self, pos, vel, pelvis):
        """Outer loop: query the policy and set the inner PID's pose target."""
        obs = self._normalize(self._build_obs(pos, vel, pelvis, self._last_action))
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        self._last_action = action
        self.inner.target = self._nominal + action * self._action_scale

    def compute(self, pos, vel, pelvis=None):
        """Inner loop: PID tracks the current reference."""
        return self.inner.compute(pos, vel)

    @property
    def last_action(self):
        """Most recent policy action (pose offset in [-1, 1] per joint)."""
        return self._last_action

    @property
    def target_pos(self):
        return self.inner.target
