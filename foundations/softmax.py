import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        print(z)
        print(np.max(z))
        shifted = z - np.max(z)
        print(shifted)
        exps = np.exp(shifted)
        print(exps)
        return np.round(exps/np.sum(exps), 4)
        pass
