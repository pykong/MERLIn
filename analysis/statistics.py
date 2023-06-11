import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

# load the data
agent1 = pd.read_csv("dqn_agent_1.csv")
agent2 = pd.read_csv("dqn_agent_2.csv")

# add a column to identify the agent in each DataFrame
agent1["agent"] = "Agent 1"
agent2["agent"] = "Agent 2"

# combine the two dataframes
all_data = pd.concat([agent1, agent2])

# smooth the reward and epsilon curves
window_size = 50  # adjust this value based on your data
all_data["reward_smooth"] = (
    all_data.groupby("agent")["reward"]
    .rolling(window_size)
    .mean()
    .reset_index(0, drop=True)
)
all_data["epsilon_smooth"] = (
    all_data.groupby("agent")["epsilon"]
    .rolling(window_size)
    .mean()
    .reset_index(0, drop=True)
)

# plot smoothed reward over time and save the plot to an SVG file
sns.lineplot(data=all_data, x="episode", y="reward_smooth", hue="agent")
plt.title("Smoothed reward over time")
plt.savefig("smoothed_reward_over_time.svg")
plt.show()

# plot smoothed epsilon over time and save the plot to an SVG file
sns.lineplot(data=all_data, x="episode", y="epsilon_smooth", hue="agent")
plt.title("Smoothed epsilon over time")
plt.savefig("smoothed_epsilon_over_time.svg")
plt.show()

# plot distribution of rewards, steps and epsilon and save the plots to SVG files
for col in ["reward", "steps", "epsilon"]:
    sns.histplot(data=all_data, x=col, hue="agent", kde=True, element="step")
    plt.title(f"Distribution of {col}")
    plt.savefig(f"distribution_of_{col}.svg")
    plt.show()

# calculate descriptive statistics
desc_stats = all_data.groupby("agent").describe()

# save the descriptive statistics as CSV
desc_stats.to_csv("desc_stats.csv")

# perform t-tests to compare agent performances
ttest_reward = ttest_ind(agent1["reward"], agent2["reward"])
ttest_steps = ttest_ind(agent1["steps"], agent2["steps"])

# create DataFrame for t-test results
ttest_results = pd.DataFrame(
    {
        "t-statistic": [ttest_reward.statistic, ttest_steps.statistic],
        "p-value": [ttest_reward.pvalue, ttest_steps.pvalue],
    },
    index=["Reward", "Steps"],
)

# save the t-test results as CSV
ttest_results.to_csv("ttest_results.csv")
