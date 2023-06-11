# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
# load the data
agent1 = pd.read_csv("log/pong_double_dqn_cnn.csv")
agent2 = pd.read_csv("log/pong_double_dqn_cnn_2.csv")

# cut out the episodes for which epsilon was held constant
agent1 = agent1.iloc[1000:]
agent2 = agent2.iloc[1000:]

# add a column to identify the agent in each DataFrame
agent1["agent"] = "Agent 1"
agent2["agent"] = "Agent 2"

# combine the two dataframes
all_data = pd.concat([agent1, agent2])

# %%
# Reset index before smoothing operation
all_data.reset_index(drop=True, inplace=True)

# Smooth the reward and epsilon curves
window_size = 20  # Adjust this value based on your data
all_data["reward_smooth"] = (
    all_data.groupby("agent")["reward"]
    .rolling(window_size)
    .mean()
    .reset_index(0, drop=True)
)


# %%

# calculate descriptive statistics
desc_stats = all_data.groupby("agent").describe()

print(f"df.size: {desc_stats.size}")
print(desc_stats.head())
print(desc_stats["reward"])

# %%

# create a figure and a first axis for the reward
fig, ax1 = plt.subplots()

# plot the reward on the first y-axis
# sns.lineplot(data=all_data, x="episode", y="reward", hue="agent", ax=ax1)
sns.lineplot(data=all_data, x="episode", y="reward_smooth", hue="agent", ax=ax1)

# add a horizontal line at y=0
ax1.axhline(0, color="grey", linestyle="--", linewidth=0.5)
ax1.set_ylabel("Reward")

# create a second y-axis for epsilon, sharing the x-axis with the first one
ax2 = ax1.twinx()

# plot epsilon on the second y-axis
epsilon_line = sns.lineplot(
    data=agent1, x="episode", y="epsilon", color="green", ax=ax2, label="Epsilon"
)
ax2.set_ylabel("Epsilon")

# get the handles and labels for all lines
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# add epsilon line to the legend
handles.append(handles2[0])
labels.append(labels2[0])

# create a new legend with all lines
ax1.legend(handles=handles, labels=labels)

plt.title("Reward over time")
plt.savefig("out/smoothed_reward_epsilon_over_time.svg")
plt.show()


# %%
for col in ["reward", "steps", "loss"]:
    sns.histplot(data=all_data, x=col, hue="agent", kde=True, element="step")
    plt.title(f"Distribution of {col}")
    plt.savefig(f"out/distribution_of_{col}.svg")
    plt.show()

# %%
