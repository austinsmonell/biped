import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from biped_env import BipedEnv


def vecnorm_path(out):
    return out + "_vecnormalize.pkl"


def main():
    parser = argparse.ArgumentParser(
        description="Train a PPO outer-loop policy for the biped (inner PID).")
    parser.add_argument("--timesteps", type=int, default=1_000_000_000)
    # Auto-scale parallel envs to the CPU (capped so the PPO rollout buffer,
    # n_steps * n_envs, stays reasonable). Override with --n-envs.
    parser.add_argument("--n-envs", type=int, default=min(16, os.cpu_count() or 8))
    parser.add_argument("--out", default="ppo_biped", help="output model path (no extension)")
    parser.add_argument("--warmstart", type=bool, default=False,
                        help="continue training from --out")
    args = parser.parse_args()

    # Loop rates (inner PID / outer RL / physics) are configured in biped_env.
    # SubprocVecEnv runs each env in its own process so MuJoCo stepping (the
    # bottleneck) parallelizes across CPU cores. With one env it's just extra
    # IPC overhead, so fall back to the in-process DummyVecEnv there.
    vec_cls = SubprocVecEnv if args.n_envs > 1 else None
    venv = make_vec_env(BipedEnv, n_envs=args.n_envs, vec_env_cls=vec_cls)
    # Normalize observations and rewards. The biped obs mixes very different
    # scales (height ~0.74 m, joint pos ~0.04 rev, ang_vel in rad/s, action in
    # [-1, 1]); PPO's MLP needs these whitened to learn. Stats are saved next to
    # the model and reloaded by controller.RLController at deployment.
    if args.warmstart:
        env = VecNormalize.load(vecnorm_path(args.out), venv)
    else:
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=512,
        gae_lambda=0.95,
        gamma=0.995,        # ~6 s effective horizon at 30 Hz; value staying up
        learning_rate=3e-4,
        ent_coef=0.005,     # encourage exploration on this hard balance task
        tensorboard_log="./tb_biped",   # `tensorboard --logdir tb_biped` to watch
    )
    if args.warmstart:
        model = PPO.load(args.out, env=env, tensorboard_log="./tb_biped")

    try:
        print("Training started. Press Ctrl+C to force stop and save.")
        model.learn(total_timesteps=args.timesteps, progress_bar=True,
                    tb_log_name="ppo")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
    finally:
        model.save(args.out)
        env.save(vecnorm_path(args.out))   # obs/reward normalization stats
        print(f"saved model to {args.out}.zip and stats to {vecnorm_path(args.out)}")


if __name__ == "__main__":
    main()
