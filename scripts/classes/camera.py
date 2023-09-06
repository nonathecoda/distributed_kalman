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
        self.neighbors = []
        self.avg_a = None
        self.avg_F = None

        self.received_a = []
        self.received_F = []

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
    
    def send_messages(self, a, F):
        # TODO: implement this
        if a is None and F is None:
            a = np.transpose(self.imm.kalman.H) @ np.linalg.inv(self.imm.kalman.R) @ self.get_measurements()
            F = np.transpose(self.imm.kalman.H) @ np.linalg.inv(self.imm.kalman.R) @ self.imm.kalman.H
        self.received_a.append(a)
        self.received_F.append(F)
        
        for n in self.neighbors:
            n.receive_message(a, F)
    
    def receive_message(self, a, F):
        self.received_a.append(a)
        self.received_F.append(F)

    def calculate_average_consensus(self):
        self.avg_a = np.sum(self.received_a, axis=0) / len(self.received_a) #np.mean(self.received_a)
        self.avg_F = np.sum(self.received_F, axis=0) / len(self.received_a)
        self.received_a = []
        self.received_F = []