import gymnasium as gym
import minigrid

def check_env_randomness(env_name, num_seeds=5):
    print(f"Checking {env_name}...")
    env = gym.make(env_name)
    
    first_obs = None
    consistent = True
    
    for i in range(num_seeds):
        obs, info = env.reset(seed=i)
        agent_pos = env.unwrapped.agent_pos
        goal_pos = None
        
        # Find goal position
        for i in range(env.unwrapped.grid.width):
            for j in range(env.unwrapped.grid.height):
                tile = env.unwrapped.grid.get(i, j)
                if tile and tile.type == 'goal':
                    goal_pos = (i, j)
                    break
            if goal_pos: break
            
        print(f"Seed {i}: Agent Start={agent_pos}, Goal={goal_pos}")

if __name__ == "__main__":
    check_env_randomness("MiniGrid-Empty-8x8-v0")
    print("-" * 20)
    check_env_randomness("MiniGrid-DoorKey-8x8-v0")
