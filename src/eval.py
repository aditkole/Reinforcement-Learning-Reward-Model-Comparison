import argparse
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

def evaluate(model_path, env_name, n_eval_episodes, seed=None, log_dir="logs/eval"):
    """
    Evaluate a RL agent with the specified model and environment.
    """
    env = gym.make(env_name, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    model = PPO.load(model_path, env=env) 

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    episode_rewards = []
    episode_lengths = []

    print(f"Evaluating model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {n_eval_episodes}")
    print(f"Logging to: {log_dir}")
    
    for i in range(n_eval_episodes):
        current_seed = seed + i if seed is not None else None
        obs, _ = env.reset(seed=current_seed)
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Log to TensorBoard
        writer.add_scalar("Eval/Reward", total_reward, i)
        writer.add_scalar("Eval/Length", steps, i)
        
        print(f"Episode {i+1}: Reward = {total_reward}, Length = {steps}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print("\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    
    # Generate Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_eval_episodes + 1), episode_rewards, marker='o', linestyle='-', color='b', label='Reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Evaluation Rewards over {n_eval_episodes} Episodes")
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(log_dir, "eval_rewards.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close()

    writer.close()
    return mean_reward, std_reward

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (without .zip)")
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-8x8-v0", help="Environment ID")
    parser.add_argument("--n-eval-episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for evaluation")
    parser.add_argument("--log-dir", type=str, default="logs/eval", help="Directory to save logs and plots")
    
    args = parser.parse_args()

    evaluate(args.model_path, args.env, args.n_eval_episodes, args.seed, args.log_dir)

if __name__ == "__main__":
    main()
