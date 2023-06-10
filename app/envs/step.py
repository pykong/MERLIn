from typing import NamedTuple

import numpy as np


class Step(NamedTuple):
    state: np.ndarray
    reward: float
    done: bool
