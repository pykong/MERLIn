from dataclasses import dataclass


@dataclass
class Config:
    """A configuration object holding all parameters for an experiment."""

    # environment parameters
    max_episodes: int = 50_000
    frame_skip: int = 4
    step_penalty: float = 0.005

    # agent parameters
    agent_name: str = "double_linear_dqn"
    load_agent: bool = False
    alpha: float = 0.0005
    epsilon_min: float = 0.1
    gamma: float = 0.98  # discount factor gamma
    memory_size: int = 50_000
    batch_size: int = 32

    # extra agent parameters
    target_net_update_interval: int = 1_000

    # save parameter
    model_save_interval: int = 2048
    video_record_interval: int = 2048

    # debugging
    verbose: bool = False
    save_state_img: bool = False
