import matplotlib.pyplot as plt
import numpy as np


def generate_color_mapping(variants: list[str]) -> dict[str, np.ndarray]:
    """
    Generates a color mapping for a list of experiments using the viridis_r colormap.

    Args:
    - experiments (List[str]): A list of unique experiment identifiers.

    Returns:
    - Dict[str, np.ndarray]: A dictionary mapping each experiment to a color.
    """
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(variants)))
    return {experiment: colors[i] for i, experiment in enumerate(variants)}
