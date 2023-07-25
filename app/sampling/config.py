from dataclasses import dataclass


@dataclass
class SamplingConfig:
    # sampling parameters
    sample_count: int = 1_000

    # environment parameters
    env_name: str = "pong"
    frame_skip: int = 4
    input_dim: int = 64
    num_stacked_frames: int = 4
    step_penalty: float = 0.0
