# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Initialize a list to hold all dataframes
all_data = []

# Iterate over each run for each agent
for i in range(1, 4):
    for j in range(1, 3):
        # Load the data
        agent = pd.read_csv(f"log/multi/agent_{j}_run_{i}.csv")
        # Cut out the episodes for which epsilon was held constant
        agent = agent.iloc[1000:]
        # Add columns to identify the agent and run in each DataFrame
        agent["agent"] = f"Agent {j}"
        agent["run"] = f"Run {i}"
        # Smooth the reward curve for this run
        window_size = 10  # Adjust this value based on your data
        agent["reward_smooth"] = agent["reward"].rolling(window_size).mean()
        all_data.append(agent)

# Combine all dataframes
all_data = pd.concat(all_data, ignore_index=True)

# Create a figure and a first axis for the reward
fig, ax1 = plt.subplots()

# Plot the mean reward and confidence intervals on the first y-axis
sns.lineplot(data=all_data, x="episode", y="reward_smooth", hue="agent", ax=ax1)

# Add a horizontal line at y=0
ax1.axhline(0, color="grey", linestyle="--", linewidth=0.5)
ax1.set_ylabel("Reward")

# Create a second y-axis for epsilon, sharing the x-axis with the first one
ax2 = ax1.twinx()

# Plot epsilon on the second y-axis, using the epsilon values from the first agent's first run as representative
agent = all_data[(all_data["agent"] == "Agent 1") & (all_data["run"] == "Run 1")]
sns.lineplot(data=agent, x="episode", y="epsilon", color="green", ax=ax2, legend=False)

ax2.set_ylabel("Epsilon")

# get the handles and labels for all lines
handles, labels = ax1.get_legend_handles_labels()

# Manually add the Epsilon label
labels += ["Epsilon"]

# create a new legend with all lines
ax1.legend(handles=handles, labels=labels)

plt.title("Reward and Epsilon over time")
plt.savefig("out/smoothed_reward_epsilon_over_time_multi.svg")
plt.show()

# %%
