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

# calculate descriptive statistics
desc_stats = all_data.groupby("agent").describe()

print(f"df.size: {desc_stats.size}")
print(desc_stats.head())
print(desc_stats["reward"])

# %%

# plot smoothed reward over time and save the plot to an SVG file
sns.lineplot(data=all_data, x="episode", y="reward", hue="agent")
plt.title("Smoothed reward over time")
plt.savefig("out/smoothed_reward_over_time.svg")
plt.show()


# %%
for col in ["reward", "steps", "loss"]:
    sns.histplot(data=all_data, x=col, hue="agent", kde=True, element="step")
    plt.title(f"Distribution of {col}")
    plt.savefig(f"out/distribution_of_{col}.svg")
    plt.show()

# %%
