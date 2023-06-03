from dataclasses import dataclass


@dataclass
class Config:
    """A configuration object holding all parameters for an experiment."""

    # environment parameters
    max_episodes: int = 1
    frame_skip: int = 4
    step_penalty: float = 0.0

    # agent parameters
    alpha: float = 1e-3
    epsilon_min: float = 0.1
    gamma: float = 1 - 1e-4  # discount factor gamma
    memory_size: int = 64_000
    batch_size: int = 64

    # extra agent parameters
    target_net_update_interval: int = 1000

    # save parameter
    model_save_interval: int = 1024
    video_record_interval: int = 1024

    # debugging
    verbose: bool = False
    save_state_img: bool = False
