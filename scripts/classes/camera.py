import numpy as np
from icecream import ic

from classes.imm import InteractingMultipleModel

class Camera():
    
    def __init__(self, name, noise, camera_position, initial_target_state) -> None:
        self.name = name
        self.noise = noise
        self.position = camera_position
        self.position_measurement = None
        self.velocity_measurement = None
        self.acceleration_measurement = None
        self.neighbors = []
        self.avg_a = None
        self.avg_F = None

        self.received_a = []
        self.received_F = []

        self.imm = InteractingMultipleModel(camera = self, initial_target_state = initial_target_state)

    def take_position_measurement(self, target_position):
        self.position_measurement = target_position + np.random.normal(0, self.noise, target_position.shape)
        return self.position_measurement
    
    def take_velocity_measurement(self, target_velocity):
        self.velocity_measurement = target_velocity + np.random.normal(0, self.noise, target_velocity.shape)
        return self.velocity_measurement
    
    def take_acceleration_measurement(self, target_acceleration):
        self.acceleration_measurement = target_acceleration + np.random.normal(0, self.noise, target_acceleration.shape)
        return self.acceleration_measurement
    
    def get_measurements(self):
        measurement = np.array([self.position_measurement[0],
                                self.position_measurement[1],
                                self.position_measurement[2]])
        return measurement.reshape((3,1))
    
    def send_messages(self, a, F):
        # TODO: implement this
        if a is None and F is None:

            a = np.transpose(self.imm.const_accel_model.H) @ np.linalg.inv(self.imm.const_accel_model.R) @ self.get_measurements()
            F = np.transpose(self.imm.const_accel_model.H) @ np.linalg.inv(self.imm.const_accel_model.R) @ self.imm.const_accel_model.H

        self.received_a.append(a)
        self.received_F.append(F)
        
        for n in self.neighbors:
            n.received_a.append(a)
            n.received_F.append(F)

    def calculate_average_consensus(self):
        self.avg_a = np.sum(self.received_a, axis=0) / len(self.received_a) #np.mean(self.received_a)
        self.avg_F = np.sum(self.received_F, axis=0) / len(self.received_a)
        self.received_a = []
        self.received_F = []