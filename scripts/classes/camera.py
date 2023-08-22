import numpy as np
from icecream import ic

class Camera():
    
    def __init__(self, name, accuracy, position) -> None:
        self.name = name
        self.accuracy = accuracy
        self.position = position
        self.measurement = None

    def take_measurement(self, target_position):
        self.measurement = target_position + np.random.normal(0, self.accuracy, target_position.shape)
        return self.measurement