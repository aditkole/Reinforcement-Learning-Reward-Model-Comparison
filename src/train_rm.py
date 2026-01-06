import argparse
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch
import torch.nn as nn
import imageio

def make_eval_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env.reset()
    return env

def make_env(env_name, rank, seed=0):
    """Factory function to create a single environment for multiprocessing."""
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")
        env = DoorKeyRewardWrapper(env)
        env = ImgObsWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init

class DoorKeyRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.has_key = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Reward for picking up key
        if not self.has_key and self.env.unwrapped.carrying is not None:
            if self.env.unwrapped.carrying.type == "key":
                # Increased reward for picking up key
                reward += 0.5 
                self.has_key = True
        
        # Reward for opening door
        if not self.door_open and self.door_pos:
            door_obj = self.env.unwrapped.grid.get(*self.door_pos)
            if door_obj is not None and door_obj.is_open:
                reward += 0.5
                self.door_open = True
                
        # Small penalty for time 
        reward -= 0.001 
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.has_key = False
        self.door_open = False
        self.door_pos = None
        
        obs, info = self.env.reset(**kwargs)
        
        # Find door position
        try:
            for i in range(self.env.unwrapped.width):
                for j in range(self.env.unwrapped.height):
                    obj = self.env.unwrapped.grid.get(i, j)
                    if obj is not None and obj.type == 'door':
                        self.door_pos = (i, j)
                        break
                if self.door_pos: break
        except Exception:
            pass
            
        return obs, info

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
    def __init__(self, env_name, gif_path="artifacts/images/ppo_doorkey_rm.gif", record_every=100_000, fps=30):
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
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--record-every", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments (cores)")
    args = p.parse_args()

    # Create vectorized environment for multi-core training
    env = SubprocVecEnv([make_env(args.env, i, args.seed) for i in range(args.n_envs)])
    env = VecMonitor(env)  # Wrap with VecMonitor to track episode stats

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
        n_steps=256,  # Per-env steps (total = n_steps * n_envs = 1024 with 4 envs)
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=args.steps, tb_log_name="PPO_RM_Shaped", callback=GifRecorderCallback(env_name=args.env, record_every=args.record_every))
    model.save("artifacts/models/ppo_doorkey_rm")

if __name__ == "__main__":
    main()