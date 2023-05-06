import time

import gym

# Create the Pong environment
env = gym.make("PongNoFrameskip-v4", render_mode="human")

env.metadata["render_fps"] = 30

# Reset the environment to start a new game
state = env.reset()

# Set the end time
end_time = time.time() + 15

while time.time() < end_time:
    # Render the game
    env.render()

    # Take a random action
    action = env.action_space.sample()

    # Step the environment with the chosen action
    next_state, reward, terminated, truncated, info = env.step(action)

    # Reset the environment if the game is over
    if terminated:
        state = env.reset()

# Close the environment
env.close()
