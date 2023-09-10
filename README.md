<p align="center">
    <a href="#readme">
        <img alt="MERLIn logo" src="https://raw.githubusercontent.com/pykong/merlin-logo/main/logo.svg">
        <!-- Logo credits: Benjamin Felder -->
    </a>
</p>
<p align="center">
    <a href="#readme"><img alt="PlaceholderBadge" src="https://badgen.net/static/PyVersion/3.11/purple"></a>
    <a href="#readme"><img alt="PlaceholderBadge" src="https://badgen.net/static/Code-Quality/A+/green"></a>
    <a href="#readme"><img alt="PlaceholderBadge" src="https://badgen.net/static/Black/OK/green"></a>
    <a href="#readme"><img alt="PlaceholderBadge" src="https://badgen.net/static/Coverage/0.0/gray"></a>
    <a href="#readme"><img alt="PlaceholderBadge" src="https://badgen.net/static/MyPy/78.0/blue"></a>
    <a href="#readme"><img alt="PlaceholderBadge" src="https://badgen.net/static/Docs/0.0/gray"></a>
    <a href="https://github.com/pykong/merlin/main/LICENSE"><img alt="License" src="https://badgen.net/static/license/MIT/blue"></a>
    <a href="#readme"><img alt="PlaceholderBadge" src="https://badgen.net/static/Build/1.0.0/pink"></a>
    <a href="#readme"><img alt="PlaceholderBadge" src="https://badgen.net/static/stars/★★★★★/yellow"></a>
</p>
<p align="center">
    <a href="#readme">
        <img alt="MERLIn training GIF" src="https://github.com/pykong/merlin-logo/blob/main/merlin_train.gif?raw=true">
    </a>
</p>

# MERLIn

MERLIn short for `modular extensible reinforcement learning interface,` allows to easily define and run reinforcement learning experiments on top of [`PyTorch`](https://github.com/pytorch/pytorch) and [`Gym`](https://github.com/openai/gym).

This project started as a homework assignment for a reinforcement learning module from my Master's studies.
I made it public, hoping you find it useful or interesting.

## Usage

### 0. Install

MERLIn uses [`poetry`](https://python-poetry.org/) for dependency management.
To install all dependencies, run:

```sh
poetry install
```

### 1. Configure experiments

Experiments can be defined as [YAML](https://learnxinyminutes.com/docs/yaml/) files merged with the default
configuration before being passed into the main training loop. Parameters are
identical to the attributes of the `Config` class, and a table of [all parameters](https://github.com/pykong/merlin/tree/polish#training-parameters) is
given further down.

Example:

`experiments/experiment_one.yaml`

```yaml
---
max_episodes: 1000
agent_name: dueling_dqn
alpha: 0.05
```

This will train the agent `dueling_dqn` for 1000 episodes at a learning rate
alpha of 0.5, while all other parameters will fall back to their default values
as defined in the `Config` class.

#### Nested element definitions

Using the `variants` array, different flavors of the same base configuration can
be defined as objects in that array. The deeper nested parameter will overwrite those
higher up. Variants can be nested.

##### variants Example

```yaml
---
max_episodes: 1000
variants:
  - {}
  - alpha: 0.01243
  - max_episodes: 333
    variants:
      - gamma: 0.5
        memory_size: 99000
      - batch_size: 64
```

The above configuration defines the following experiments:

1. `max_episodes: 1000`
2. `max_episodes: 1000` and `alpha: 0.01243`
3. `max_episodes: 333`, `gamma: 0.5` and `memory_size: 99000`
4. `max_episodes: 333`, and `batch_size: 64`

### 2. Start training

After defining at least one experiment as described in the previous section, start training by simply invoking the following command:

`poetry run train`

#### Training in the background

To start training in the background, to allow training to proceed beyond the shell session, run the following script:

`./scripts/traing_bg.sh`

The script will also watch the generated log statements to provide continuous console
output.

### 3. Results

#### Console output

During training, the following outputs are continuously logged to the console:

1. episode index
2. epsilon
3. reward
4. train loss
5. episode steps
6. total episode time

Special events like model saving or video recording will also be logged if they
occur.

#### File output

Each experiment will generate a subfolder in the `results/` directory. Within
that subfolder, the following files will be placed:

1. `experiment.yaml`: The exact parameters the experiment was run with
2. A log holding the training logs, as printed out to the console (see section
   before)
3. Model checkpoints.
4. Video files of selected episode runs.
5. Images of the preprocessed state (optional).

### Statistical Analysis

MERLIn will automatically conduct some crude statistical analysis of the experimental results post-training.
You can manually trigger the analysis by running: `poetry run analyze <path/to/experiment/results>`.
Analysis results will be written to a subfolder of the results directory `analysis/`.

#### Summarization

As of `v1.0.0`, the last 2,000 episodes (as a hard-coded assumption of plateauing) are used to compare different algorithms.
The statistical analysis will aggregate all runs of each variant and calculate the following:

- mean reward
- std reward
- lower bound of the confidence interval for mean reward
- mean steps
- std steps

#### Plottings

Line plots of rewards over episodes and histograms showing the reward distribution of all variants are produced.

<p float="left">
  <img alt="MERLIn logo" src="https://raw.githubusercontent.com/pykong/merlin-logo/main/reward.svg" width="49%" />
  <img alt="MERLIn logo" src="https://raw.githubusercontent.com/pykong/merlin-logo/main/reward_dist.svg"  width="45%"/>
</p>

### Training Parameters

Below is an overview of the parameters to configure experiments.

| Parameter Name               | Description                                                                                      | Optional | Default      |
|------------------------------|--------------------------------------------------------------------------------------------------|----------|--------------|
| experiment                   | Unique id of the experiment.                                                                     | No       |              |
| variant                      | Unique id of the variant of an experiment.                                                       | No       |              |
| run                          | Unique id of the run of a variant.                                                               | Yes      | 0            |
| run_count                    | The number of independent runs of an experiment.                                                 | Yes      | 3            |
| env_name                     | The environment to be used.                                                                      | Yes      | 'pong'       |
| frame_skip                   | The number of frames to skip per action.                                                         | Yes      | 4            |
| input_dim                    | The input dimension of the model.                                                                | Yes      | 64           |
| num_stacked_frames           | The number of frames to stack.                                                                   | Yes      | 4            |
| step_penalty                 | Penalty given to the agent per step.                                                             | Yes      | 0.0          |
| agent_name                   | The agent to be used.                                                                            | Yes      | 'double_dqn' |
| net_name                     | The neural network to be used.                                                                   | Yes      | 'linear_deep_net' |
| target_net_update_interval   | The number of steps after which the target network should be updated.                            | Yes      | 1024         |
| episodes                     | The number of episodes to train for.                                                             | Yes      | 5000         |
| alpha                        | The learning rate of the agent.                                                                  | Yes      | 5e-6         |
| epsilon_decay_start          | The episode to start epsilon decay on.                                                           | Yes      | 1000         |
| epsilon_step                 | The absolute value to decrease epsilon by per episode.                                           | Yes      | 1e-3         |
| epsilon_min                  | The minimum epsilon value for epsilon-greedy exploration.                                        | Yes      | 0.1          |
| gamma                        | The discount factor for future rewards.                                                          | Yes      | 0.99         |
| memory_size                  | The size of the replay memory.                                                                   | Yes      | 500,000      |
| batch_size                   | The batch size for learning.                                                                     | Yes      | 32           |
| model_save_interval          | The number of steps after which the model should be saved. If None, model will be saved at the end of epoch only. | Yes | None           |
| video_record_interval        | Steps between video recordings.                                                                  | Yes      | 2500         |
| save_state_img               | Whether to take images during training.                                                          | Yes      | False        |
| use_amp                      | Whether to use automatic mixed precision.                                                        | Yes      | True         |

### Extending Agents, Environments, and Neural Networks

MERLIn boasts itself of being modular and extensible, meaning you can quickly implement new agents, environments, and neural networks.
So that you know, all you need to extend said objects is to derive a new class from the respective abstract base class and register it at the regarding registry.

#### Example: Implementing a new Neural Network

Create a new Python module, `app/nets/new_net.py`, holding a new class deriving from `BaseNet`.
You must provide a unique name via the name property.

```py
from app.nets._base_net import BaseNet


class NewNet(BaseNet):
    @classmethod
    @property
    def name(cls) -> str:
        return "new_net"  # give it a unique name here

    def _define_net(
        self, state_shape: tuple[int, int, int], num_actions: int
    ) -> nn.Sequential:
      # your PyTorch network definition goes here
```

Add `NewNet` to the registry of neural networks in `app/nets/__init__.py`, to make it automatically available to the `make_net` factory function.

```py

...

net_registry = [
    ...
    NewNet,  # register here
]

...

```

That's it. That simple. From now on, you can use the new network in your experiment definitions:

```yaml
---
net_name: new_net
```

### Scripts

The application comes with several bash scripts to help conduct certain
functions.

#### `check_cuda.sh` & `watch_gpu`

Print out information regarding the system's current CUDA installation and GPU usage for sanity-checking and troubleshooting.

#### `install_atari.sh`

Installs the Atari ROMs used by `Gym` into the virtual environment.

#### Sync scripts

Typically, you want to offload the training workload to a cloud virtual machine. In
In this regard, `sync_up.sh` will upload sources and experiments to that machine.
Afterward, the training results can be downloaded to your local system using
`sync_down.sh`.

A configuration-like connection data for both sync scripts is within the `sync.cfg` file.

## Limitations

This project is now more of a didactic exercise rather than an attempt to topple
established reinforcement learning frameworks such as [`RLlib`](https://docs.ray.io/en/latest/rllib/index.html).

As of `v1.0.0` the most crucial limitations of MERLIn stand as:

1. Single environment implemented, namely `Pong`.
2. Single class of agents implemented, namely variations of `DQN`.
3. Statistical analysis is rudimentary and does not happen parallel to training.

### Contributions welcome

If you like MERLIn and want to develop it further, feel free to fork and open any pull request. 🤓
