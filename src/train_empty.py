import argparse
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import imageio

def make_eval_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env.reset()
    return env

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))

class GifRecorderCallback(BaseCallback):
    def __init__(self, env_name, gif_path="artifacts/images/ppo_empty.gif", record_every=100_000, fps=30):
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
    p.add_argument("--env", default="MiniGrid-Empty-8x8-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--record-every", type=int, default=100_000)
    args = p.parse_args()

    env = gym.make(args.env, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env.reset()

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        policy="CnnPolicy", 
        env=env,
        policy_kwargs=policy_kwargs,
        seed=args.seed, 
        verbose=1, 
        tensorboard_log="./tb_logs/",
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=args.steps, tb_log_name="PPO_Empty_Baseline", callback=GifRecorderCallback(env_name=args.env, record_every=args.record_every))
    model.save("artifacts/models/ppo_empty")

if __name__ == "__main__":
    main()