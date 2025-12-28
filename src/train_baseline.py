import argparse
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import imageio

def make_eval_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env.reset()
    return env

class GifRecorderCallback(BaseCallback):
    def __init__(self, env_name, gif_path="ppo_doorkey.gif", record_every=100_000, fps=30):
        super().__init__()
        self.gif_path = gif_path
        self.record_every = record_every
        self.fps = fps
        self.eval_env = make_eval_env(env_name)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_every == 0:
            self.record_gif()
        return True

    def record_gif(self, episode_length=1600):
        frames = []
        obs, _ = self.eval_env.reset()

        for _ in range(episode_length):
            frames.append(self.eval_env.render())
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.eval_env.step(action)
            if terminated or truncated:
                break

        imageio.mimsave(self.gif_path, frames, fps=self.fps)
        print(f"Saved GIF to {self.gif_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="MiniGrid-DoorKey-8x8-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=500_000)
    args = p.parse_args()

    env = gym.make(args.env, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env.reset()

    model = PPO(
        policy="MlpPolicy", 
        env=env,
        seed=args.seed, 
        verbose=1, 
        tensorboard_log="./tb_logs/",
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=args.steps, tb_log_name="PPO_Sparse_Baseline", callback=GifRecorderCallback(env_name=args.env))
    model.save("artifacts/models/ppo_doorkey_sparse")

if __name__ == "__main__":
    main()