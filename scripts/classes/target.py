import icecream as ic
import numpy as np

class Target():

    def __init__(self, name, x, y, z, yaw, pitch, roll) -> None:
        self.name = name
        self.x_coord = x
        self.y_coord = y
        self.z_coord = z
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
    
    def get_position(self):
        return np.array([self.x_coord, self.y_coord, self.z_coord, self.yaw, self.pitch, self.roll])
    
    def set_position(self, position):
        self.x_coord = position[0]
        self.y_coord = position[1]
        self.z_coord = position[2]
        self.yaw = position[3]
        self.pitch = position[4]
        self.roll = position[5]