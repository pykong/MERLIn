from dataclasses import dataclass


@dataclass
class Config:
    """A configuration object holding all parameters for an experiment."""

    # environment parameters
    max_episodes: int = 50_000
    frame_skip: int = 4
    step_penalty: float = 0.0

    start_epsilon_decay: int = 300

    # agent parameters
    agent_name: str = "double_dqn_cnn"
    load_agent: bool = False
    alpha: float = 0.0005
    epsilon_min: float = 0.1
    gamma: float = 0.999  # discount factor gamma
    memory_size: int = 50_000
    batch_size: int = 128
    epochs: int = 1

    # extra agent parameters
    target_net_update_interval: int = 4_000

    # save parameter
    model_save_interval: int = 1048
    video_record_interval: int = 1048

    # debugging
    verbose: bool = False
    save_state_img: bool = False
