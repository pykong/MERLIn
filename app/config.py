from dataclasses import dataclass

from typing import Self


@dataclass
class Config:
    """A configuration object holding all parameters for an experiment.

    Attributes:
    experiment_id (str): Unique id of the experiment.

    variant_id (str): Unique id of the variant of an experiment.

    run_id (int): Unique id of the run of a variant.

    episodes (int): Number of episodes to train on. Default is 5000.

    epsilon_decay_start (int): The episode at which epsilon decay should start. Default is 1000.

    env_name (str): The name of the environment to be used in the experiment. Default is 'pong'.

    frame_skip (int): Number of frames to skip in the environment per action. Default is 4.

    input_dim (int): The input dimension of the model. Default is 64.

    num_stacked_frames (int): Number of frames to stack together as input for the model. Default is 4.

    step_penalty (float): Penalty given to the agent at each step. Default is 0.0.

    agent_name (str): The name of the agent to be used in the experiment. Default is 'double_dqn'.

    net_name (str): The name of the neural network to be used in the experiment. Default is 'conv_net'.

    alpha (float): The learning rate of the agent. Default is 5e-6.

    epsilon_min (float): The minimum epsilon value for epsilon-greedy exploration. Default is 0.1.

    epsilon_step (float): The absolute value to decrease epsilon by per episode. Default is 1e-3.

    gamma (float): The discount factor for future rewards. Default is 0.99.

    memory_size (int): The size of the replay memory. Default is 500,000.

    batch_size (int): The batch size for learning. Default is 32.

    use_amp (bool): Whether to use automatic mixed precision. Default is True.

    target_net_update_interval (int): The number of steps after which the target network should be updated. Default is 1024.

    model_save_interval (int?): The number of steps after which the model should be saved. If None model will be saved at the end of epoch only. Default is None.

    video_record_interval (int): The number of steps after which a video recording should be made. Default is 2500.

    save_state_img (bool): If True, save images of the states during training. Default is False.
    """

    # id
    experiment_id: str
    variant_id: str
    run_id: int

    # environment parameters
    env_name: str = "pong"
    frame_skip: int = 4
    input_dim: int = 64
    num_stacked_frames: int = 4
    step_penalty: float = 0.0

    # agent parameters
    agent_name: str = "double_dqn"
    net_name: str = "linear_deep_net"
    target_net_update_interval: int = 1_024

    # training parameters
    episodes: int = 5_000
    alpha: float = 5e-6
    epsilon_decay_start: int = 1_000
    epsilon_step: float = 1e-3
    epsilon_min: float = 0.1
    gamma: float = 0.99
    memory_size: int = 500_000
    batch_size: int = 32

    # save parameter
    model_save_interval: int | None = None
    video_record_interval: int = 2_500

    # debugging
    save_state_img: bool = False

    # automatic mixed precision
    use_amp: bool = True

    def __hash__(self: Self) -> int:
        """Define hash based on compositition of the three ids."""
        return hash((self.experiment_id, self.variant_id, self.run_id))
