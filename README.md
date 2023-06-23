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

#### Training in the background

To start training in the background, to allow train to proceed beyond shell session,
run the following script:

`./scripts/traing_bg.sh`

The script will also watch the generated log statements to provide continous console
output.

### 3. Results

#### Console output

During training the following outputs are continuously logged to the console:

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

1. `experiment.json`: The exact parameters the experiment was run with
2. A log holding the training logs, as printed out to the console (see section
   before)
3. Model checkpoints.
4. Video files of selected episode runs.
5. Images of the preprocessed state (optional).

### Scripts

The application comes with several bash scripts to help conduct certain
functions.

#### `check_cuda.sh` & `watch_gpu`

Print out information regarding the current CUDA installation and GPU usage of
the system. For sanity-checking and troubleshooting purposes.

#### `install_atari.sh`

Installs the Atari ROMs used by gym into the virtual environment.

#### Sync scripts

Typically you want to offload the training workload to cloud virtual machine. In
this regard `sync_up.sh` will upload sources and experiments to that machine.
Afterwards the training results can be downloaded to your local system using
`sync_down.sh`.

Configuration such as connection data for both sync scripts are situated within
the `sync.cfg` file.
