from dataclasses import dataclass


@dataclass
class Config:
    """A configuration object holding all parameters for an experiment.

    Attributes:
    max_episodes (int): Maximum number of episodes to train on. Default is 5000.

    start_epsilon_decay (int): The episode at which epsilon decay should start. Default is 1000.

    env_name (str): The name of the environment to be used in the experiment. Default is 'pong'.

    frame_skip (int): Number of frames to skip in the environment per action. Default is 4.

    input_dim (int): The input dimension of the model. Default is 64.

    num_stacked_frames (int): Number of frames to stack together as input for the model. Default is 4.

    step_penalty (float): Penalty given to the agent at each step. Default is 0.0.

    agent_name (str): The name of the agent to be used in the experiment. Default is 'double_dqn'.

    net_name (str): The name of the neural network to be used in the experiment. Default is 'neuroips_net'.

    load_agent (bool): If True, load a pre-trained agent. Default is False.

    alpha (float): The learning rate of the agent. Default is 5e-5.

    epsilon_min (float): The minimum epsilon value for epsilon-greedy exploration. Default is 0.1.

    gamma (float): The discount factor for future rewards. Default is 0.999.

    memory_size (int): The size of the replay memory. Default is 50,000.

    batch_size (int): The batch size for learning. Default is 32.

    use_amp (bool): Whether to use automatic mixed precision. Default is True.

    target_net_update_interval (int): The number of steps after which the target network should be updated. Default is 1024.

    model_save_interval (int): The number of steps after which the model should be saved. Default is 2048.

    video_record_interval (int): The number of steps after which a video recording should be made. Default is 2500.

    verbose (bool): If True, print extra information during training. Default is False.

    save_state_img (bool): If True, save images of the states during training. Default is False.
    """

    # training parameters
    max_episodes: int = 5_000
    start_epsilon_decay: int = 1_000

    # environment parameters
    env_name: str = "pong"
    frame_skip: int = 4
    input_dim: int = 84
    num_stacked_frames: int = 4
    step_penalty: float = 0.0

    # agent parameters
    agent_name: str = "double_dqn"
    net_name: str = "nature_net"
    load_agent: bool = False
    alpha: float = 5e-5
    epsilon_min: float = 0.1
    gamma: float = 0.999
    memory_size: int = 50_000
    batch_size: int = 32
    use_amp: bool = True

    # extra agent parameters
    target_net_update_interval: int = 1024

    # save parameter
    model_save_interval: int = 8192
    video_record_interval: int = 2500

    # debugging
    verbose: bool = False
    save_state_img: bool = False
