import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        your_answer= 1 / (1 + np.exp(-z))
        return np.round(your_answer, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return [np.float64(max(0,i)) for i in z]
