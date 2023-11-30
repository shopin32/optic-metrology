import numpy as np


class Wave:

    def __init__(self, source_x: float, source_y: float):
        self.source_x = source_x
        self.source_y = source_y
    

    def compute_distances(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return  np.sqrt((x - self.source_x)**2 + (y - self.source_y)**2)