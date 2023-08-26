import numpy as np
from icecream import ic

from classes.imm import InteractingMultipleModel

class Camera():
    
    def __init__(self, name, accuracy, position, initial_target_pose) -> None:
        self.name = name
        self.accuracy = accuracy
        self.position = position
        self.position_measurement = None
        self.velocity_measurement = None

        self.imm = InteractingMultipleModel(camera = self, initial_pose=initial_target_pose)

    def take_position_measurement(self, target_position):
        self.position_measurement = target_position + np.random.normal(0, self.accuracy, target_position.shape)
        return self.position_measurement
    
    def take_velocity_measurement(self, target_velocity):
        self.velocity_measurement = target_velocity + np.random.normal(0, self.accuracy, target_velocity.shape)
        return self.velocity_measurement
    
    def get_measurements(self):
        measurement = np.array([self.position_measurement[0],
                                self.velocity_measurement[0],
                                self.position_measurement[1],
                                self.velocity_measurement[1],
                                self.position_measurement[2],
                                self.velocity_measurement[2]])
        return measurement.reshape((6,1))