import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from biped_env import BipedEnv


def vecnorm_path(out):
    return out + "_vecnormalize.pkl"


def main():
    parser = argparse.ArgumentParser(
        description="Train a PPO outer-loop policy for the biped (inner PID).")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--out", default="ppo_biped", help="output model path (no extension)")
    parser.add_argument("--warmstart", action="store_true",
                        help="continue training from --out")
    args = parser.parse_args()

    # Loop rates (inner PID / outer RL / physics) are configured in biped_env.
    venv = make_vec_env(BipedEnv, n_envs=args.n_envs)
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
    )
    if args.warmstart:
        model = PPO.load(args.out, env=env)

    try:
        print("Training started. Press Ctrl+C to force stop and save.")
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
    finally:
        model.save(args.out)
        env.save(vecnorm_path(args.out))   # obs/reward normalization stats
        print(f"saved model to {args.out}.zip and stats to {vecnorm_path(args.out)}")


if __name__ == "__main__":
    main()
