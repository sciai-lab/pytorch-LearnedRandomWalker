# -*- coding: utf-8 -*-

import numpy as np
from numba import jit


@jit(cache=True)
def build_laplacian2D(elap0, size):
    """
    lap0: Graph weights elements, must be (num pixels, 2)
    size: size original image, must be a 2D tuple
    returns: laplacian elements and COO indices
    """
    num_elements_lap = 2 * (size[1] - 1)*size[0] + 2 * (size[0] - 1)*size[1] + size[0]*size[1]
    elap = np.zeros(num_elements_lap)

    # lap_index and off diagonal lap index
    i_ind, j_ind = (np.zeros(num_elements_lap, dtype=np.int),
                    np.zeros(num_elements_lap, dtype=np.int))

    cout, sk = 0, 1
    for i in range(size[0]):
        for j in range(size[1]):
            k = i * size[1] + j

            if i > sk - 1:
                n = elap0[0, i - sk, j]
                elap[cout] = -n
                i_ind[cout], j_ind[cout] = k, k - sk * size[1]
                cout += 1
            else:
                n = 0

            if j > sk - 1:
                w = elap0[1, i, j - sk]
                elap[cout] = -w
                i_ind[cout], j_ind[cout] = k, k - sk
                cout += 1
            else:
                w = 0

            if i < size[0] - sk:
                s = elap0[0, i, j]
                elap[cout] = -s
                i_ind[cout], j_ind[cout] = k, k + sk * size[1]
                cout += 1
            else:
                s = 0

            if j < size[1] - sk:
                e = elap0[1, i, j]
                elap[cout] = -e
                i_ind[cout], j_ind[cout] = k, k + sk
                cout += 1
            else:
                e = 0

            norm = (n + w + s + e)
            elap[cout] = max(norm, 1e-5)
            i_ind[cout], j_ind[cout] = k, k
            cout += 1

    return elap, i_ind, j_ind
