# Readme

## Usage

### 1. Configure experiments

Experiments can be defined as JSON files that are merged with the default
configuration before being passed into the main training loop. Parameters are
identical to the attributes of the `Config` class.

Example:

`experiments/experiment_one.json`

```json
{
  "max_episodes": 1000,
  "agent_name": "duelling_dqn",
  "alpha": 0.05
}
```

This will train the agent `duelling_dqn` for 1000 episodes at a learning rate
alpha of 0.5, while all other parameters will fall back to their default values
as defined in the `Config` class.

There is a one-to-one relationship between experiment definitions and
experiments being run. This means each JSON file in the experiment folder will
only result in a single experiment epoch being run.

### 2. Start training

After defining at least one experiment as described in the previous section,
start training by simply invoking the following command:

`poetry run train`
