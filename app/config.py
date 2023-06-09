from dataclasses import dataclass


@dataclass
class Config:
    """A configuration object holding all parameters for an experiment."""

    # training parameters
    max_episodes: int = 64_000
    start_epsilon_decay: int = 1_000

    # environment parameters
    frame_skip: int = 4
    input_dim: int = 64
    frame_stack_gap: int = 4
    step_penalty: float = 0.01

    # agent parameters
    agent_name: str = "double_dqn_cnn"
    load_agent: bool = False
    alpha: float = 0.001
    epsilon_min: float = 0.1
    gamma: float = 0.999  # discount factor gamma
    memory_size: int = 64_000
    batch_size: int = 64
    epochs: int = 1

    # extra agent parameters
    target_net_update_interval: int = 8192

    # save parameter
    model_save_interval: int = 2048
    video_record_interval: int = 2048

    # debugging
    verbose: bool = False
    save_state_img: bool = False
