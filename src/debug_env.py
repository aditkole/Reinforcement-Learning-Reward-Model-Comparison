import gymnasium as gym
import minigrid

env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
env.reset()

print("Environment Keys:")
print(dir(env.unwrapped))

print("\nGrid Objects:")
for i in range(env.unwrapped.width):
    for j in range(env.unwrapped.height):
        obj = env.unwrapped.grid.get(i, j)
        if obj is not None:
            print(f"Pos ({i}, {j}): {obj.type}, State: {getattr(obj, 'is_open', 'N/A')}, Locked: {getattr(obj, 'is_locked', 'N/A')}")

# Try to find the door
door_pos = None
for i in range(env.unwrapped.width):
    for j in range(env.unwrapped.height):
        obj = env.unwrapped.grid.get(i, j)
        if obj is not None and obj.type == 'door':
            door_pos = (i, j)
            break

print(f"\nDoor found at: {door_pos}")
if door_pos:
    door = env.unwrapped.grid.get(*door_pos)
    print(f"Door is open: {door.is_open}")
    print(f"Door is locked: {door.is_locked}")
