import asyncio
import math
import time

import numpy as np

from biped_env import (
    JOINT_NAMES, NOMINAL_POSE, N_JOINTS,
    INNER_KP, INNER_KD, INNER_KI, INNER_INTEGRAL_LIMIT, MAX_TORQUE,
    SIM_DT, INNER_DT, INNER_HZ, OUTER_HZ, SIM_HZ, SIM_PER_INNER, SIM_PER_OUTER,
    read_pelvis,
)


# Either flag, or both. If both are True the sim runs first for sim_duration
# seconds, then the (placeholder) hardware loop starts.
run_hardware = False
run_sim = True
sim_duration = 10.0   # seconds; only used when both flags are True


# Shared controller parameters. Loop rates (inner PID, outer RL, physics) are
# configured in biped_env: INNER_HZ, OUTER_HZ, SIM_HZ.
controller_type = "rl"   # "pid" (inner loop only) or "rl" (RL outer + PID inner)
target_pose = NOMINAL_POSE  # joint pose the inner PID holds (rev), per JOINT_NAMES

# Inner-loop PID gains. Default to the biped_env values so an RL cascade matches
# its training; override here to tune the standalone "pid" stand/hold behavior.
pos_kp = INNER_KP
pos_kd = INNER_KD
pos_ki = INNER_KI
integral_limit = INNER_INTEGRAL_LIMIT
max_torque = MAX_TORQUE

# RL: saved PPO policy (no .zip extension), trained by train_ppo.py.
rl_model_path = "ppo_biped"

print_period = 0.1   # console print period (s)


def make_inner_pid(target):
    from controller import PIDController
    return PIDController(
        target=np.array(target, dtype=float), kp=pos_kp, kd=pos_kd, ki=pos_ki,
        dt=INNER_DT, integral_limit=integral_limit, max_torque=max_torque,
        n_joints=N_JOINTS,
    )


def make_controller():
    """Build the configured controller. Both expose the same (update_outer,
    compute) interface so run_in_sim drives either one identically."""
    if controller_type == "pid":
        return make_inner_pid(target_pose)
    if controller_type == "rl":
        from controller import RLController
        # The inner PID runs at INNER_HZ in both training and deployment.
        inner = make_inner_pid(NOMINAL_POSE)
        return RLController(inner, rl_model_path)
    raise ValueError(f"unknown controller_type: {controller_type!r}")


async def run_in_sim(controller, duration=None):
    import mujoco.viewer
    from sim_motor import SimMotor

    motor = SimMotor("biped.xml", JOINT_NAMES)
    # Drive physics at SIM_HZ so the inner/outer loops land on exact step counts.
    motor.model.opt.timestep = SIM_DT
    motor.reset_state(pose=NOMINAL_POSE)
    print_every = max(1, int(round(print_period / SIM_DT)))

    with mujoco.viewer.launch_passive(motor.model, motor.data) as viewer:
        viewer.cam.lookat[2] += 0.5
        sim_start = motor.data.time
        wall_start = time.perf_counter()
        torque_cmd = np.zeros(N_JOINTS)

        try:
            sim_tick = 0
            while True:
                if duration is not None and (motor.data.time - sim_start) >= duration:
                    break

                pos = motor.get_pos()
                vel = motor.get_vel()
                pelvis = read_pelvis(motor)

                # Outer loop (OUTER_HZ): refresh the controller's reference.
                if sim_tick % SIM_PER_OUTER == 0:
                    controller.update_outer(pos, vel, pelvis)
                # Inner loop (INNER_HZ): recompute the tracking torque.
                if sim_tick % SIM_PER_INNER == 0:
                    torque_cmd = controller.compute(pos, vel, pelvis)
                    motor.set_torque(torque_cmd)

                motor.step_n(1)   # one physics step at SIM_HZ
                sim_tick += 1
                viewer.sync()

                if sim_tick % print_every == 0:
                    roll, pitch, _ = motor.get_pelvis_rpy()
                    msg = (f"time={motor.data.time - sim_start:+.3f} s  "
                           f"height={pelvis['height']:+.3f} m  "
                           f"roll={roll:+.3f} pitch={pitch:+.3f} rad  "
                           f"fwd_vel={pelvis['lin_vel'][0]:+.3f} m/s  "
                           f"|trq|max={np.max(np.abs(torque_cmd)):+.2f} Nm")
                    # For the RL controller, also show the latest policy action
                    # (pose offset in [-1, 1] per joint, in JOINT_NAMES order).
                    action = getattr(controller, "last_action", None)
                    if action is not None:
                        msg += "  action=[" + " ".join(
                            f"{a:+.2f}" for a in action) + "]"
                    print(msg)

                # Synchronize wall clock to sim clock.
                sim_elapsed = motor.data.time - sim_start
                wall_elapsed = time.perf_counter() - wall_start
                lag = sim_elapsed - wall_elapsed  # positive means sim is ahead
                if lag > 0:
                    target_wall = wall_start + sim_elapsed
                    while time.perf_counter() < target_wall:
                        await asyncio.sleep(max(0, target_wall - time.perf_counter()))
                # If lag < 0, sim is behind — run as fast as possible to catch up.
        finally:
            motor.stop()


async def run_on_hardware(controller):
    """Placeholder — no biped hardware yet.

    When a physical biped exists, mirror balance_bot/pendulum: open the motor
    backend and run the nested loops -- controller.update_outer(...) at OUTER_HZ
    and controller.compute(...) at INNER_HZ (pelvis from an onboard IMU) -- and
    ensure a safe stop on exit.
    """
    raise NotImplementedError(
        "biped hardware is not implemented yet — set run_hardware = False")


async def main():
    if not (run_hardware or run_sim):
        raise RuntimeError("Set at least one of run_hardware / run_sim to True.")
    controller = make_controller()
    if run_sim:
        await run_in_sim(controller, duration=sim_duration if run_hardware else None)
    if run_hardware:
        await run_on_hardware(controller)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
