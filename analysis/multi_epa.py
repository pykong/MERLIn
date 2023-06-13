# %%
import os
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# change cwd to script location
os.chdir(Path(__file__).parent)

# epsilon skip episode count
START_EPSILON_DECAY: Final[int] = 1000

# window size for reward smoothing
SMOOTH_WINDOW: Final[int] = 10  # Adjust this value based on your data

# initialize a list to hold all dataframes
all_data: Final[list[pd.DataFrame]] = []

# iterate over each run for each agent
for run_idx in range(1, 4):
    for agent_idx in range(1, 3):
        # ingest data from CSV file
        agent_df = pd.read_csv(f"log/multi/agent_{agent_idx}_run_{run_idx}.csv")
        # cut out the episodes for which epsilon was held constant
        agent_df = agent_df.iloc[START_EPSILON_DECAY:]
        # add columns to identify the agent and run in each DataFrame
        agent_df["agent"] = f"Agent {agent_idx}"
        agent_df["run"] = f"Run {run_idx}"
        # smooth the reward curve for this run
        agent_df["reward_smooth"] = agent_df["reward"].rolling(SMOOTH_WINDOW).mean()
        all_data.append(agent_df)

# combine all dataframes
all_data = pd.concat(all_data, ignore_index=True)

# create a figure and a first axis for the reward
fig, ax1 = plt.subplots()

# plot the mean reward and confidence intervals on the first y-axis
sns.lineplot(data=all_data, x="episode", y="reward_smooth", hue="agent", ax=ax1)

# add a horizontal line at y=0
ax1.axhline(0, color="grey", linestyle="--", linewidth=0.5)
ax1.set_ylabel("Reward")

# create a second y-axis for epsilon, sharing the x-axis with the first one
ax2 = ax1.twinx()

# plot epsilon on the second y-axis, using the epsilon values of
#  the first agent's first run as representative
agent_df = all_data[(all_data["agent"] == "Agent 1") & (all_data["run"] == "Run 1")]
sns.lineplot(
    data=agent_df, x="episode", y="epsilon", color="green", ax=ax2, legend=False
)
ax2.set_ylabel("Epsilon")

# get the handles and labels for all lines
handles, labels = ax1.get_legend_handles_labels()

# manually add the Epsilon label
labels += ["Epsilon"]

# create a new legend with all lines
ax1.legend(handles=handles, labels=labels)

plt.title("Reward and Epsilon over time")
plt.savefig("out/smoothed_reward_epsilon_over_time_multi.svg")
plt.show()


def eval():
    print("Run analysis of experimental results")


# %%
