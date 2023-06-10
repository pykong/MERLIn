from dataclasses import dataclass


@dataclass
class Config:
    """A configuration object holding all parameters for an experiment."""

    # training parameters
    max_episodes: int = 3
    start_epsilon_decay: int = 1_000

    # environment parameters
    frame_skip: int = 4
    input_dim: int = 64
    num_stacked_frames: int = 4
    step_penalty: float = 0.0

    # agent parameters
    agent_name: str = "duelling_dqn"
    net_name: str = "ben_net"
    load_agent: bool = False
    alpha: float = 0.0001
    epsilon_min: float = 0.1
    gamma: float = 0.999  # discount factor gamma
    memory_size: int = 5_000
    batch_size: int = 32

    # extra agent parameters
    target_net_update_interval: int = 1024

    # save parameter
    model_save_interval: int = 2048
    video_record_interval: int = 2048

    # debugging
    verbose: bool = False
    save_state_img: bool = True
