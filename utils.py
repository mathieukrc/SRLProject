from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from controlled.controlled_env import h_fn


def visualise_controlled_area_cost(h: Callable[[float, float], int]):
    axis = np.linspace(-1.5, 1.5, 100)

    image = np.zeros((len(axis), len(axis), 3), dtype=np.uint8)

    for i, y in enumerate(axis):
        for j, x in enumerate(axis):
            cost = h(x, y)
            is_real_feasible = h_fn(np.array(((x, y)))) <= 0

            color = [0, 0, 0]
            if cost <= 0 or is_real_feasible:
                color[1] = 255

            if cost > 0 or not is_real_feasible:
                color[0] = 255

            image[i, j] = np.array(color)

    plt.imshow(image)
    plt.show()
