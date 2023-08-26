import icecream as ic
import numpy as np

class Target():

    def __init__(self, name, x, y, z) -> None:
        self.name = name

        self.position = np.array([x, y, z])
        self.velocity = np.array([0, 0, 0])
        self.acceleration = np.array([0, 0, 0])
    
    def get_position(self):
        return self.position
    
    def set_position(self, position):
        self.position = position

    def set_velocity(self, velocity):
        self.velocity = velocity
    
    def get_velocity(self):
        return self.velocity

    def set_acceleration(self, acceleration):
        self.acceleration = acceleration
    
    def get_acceleration(self):
        return self.acceleration