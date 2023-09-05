from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import time

from classes.target import Target
from classes.camera import Camera
from classes.imm import InteractingMultipleModel
from classes.plotter import Plotter
from kinetic_models.const_velocity import CV_CYPR_Model


class Main():

    def __init__(self, path = None, n_frames = 5, n_cameras = 1, sensor_accuracy = 5) -> None:
        self.start_time = time.monotonic()
        self.path = path
        if self.path is None:
            self.path = np.array(list(self.gen_path(n_frames))).T # (3, N)
        self.n_frames = n_frames
        if n_cameras > 1:
            self.distributed = True
        else:
            self.distributed = False
        self.cameras = self.create_cameras(n_cameras, sensor_accuracy, path[:, 0])
    
        self.target = Target("target", 0, 0, 0)
        self.last_timestamp = 0
        

        self.plotter = Plotter()
        
        self.start_world()

    def start_world(self):
        print("Starting world...")
        
        # Uncomment this to plot the path
        fig = plt.figure(figsize=(1, 1))
        ax = fig.add_subplot(projection='3d')
        line, = ax.plot(self.path[0, 0:1], self.path[1, 0:1], self.path[2, 0:1])
        

        # Setting the axes properties
        ax.set_xlim3d([-1.0, 500.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-1.0, 500.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0.0, 300.0])
        ax.set_zlabel('Z')

        for i in range(0, self.n_frames):
            #self.start_time = time.monotonic()
            self.update_world(i, self.path, line)
            #duration = time.monotonic() - self.start_time
            #sleep_for = max(0,0.2 - duration)
            #time.sleep(sleep_for)
            
            
        #plt.show()


    def update_world(self, num, path, line):
        self.target.set_position(path[:, num])
        if num > 0:
            self.target.set_velocity(path[:, num] - path[:, num-1]+ np.random.normal(0, 0.5, (3,)))
        if num > 1:
            self.target.set_acceleration(path[:, num] - 2*path[:, num-1] + path[:, num-2])
    
        line.set_data(path[:2, :num])
        line.set_3d_properties(path[2, :num])
        
        if num%5 == 0 and num != 0: # take measurement every 5 frames
            timestep = num - self.last_timestamp
            self.last_timestamp = num
            for cam in self.cameras:
                cam.take_position_measurement(self.target.get_position())
                cam.take_velocity_measurement(self.target.get_velocity())

            # calculate average consensus
            if self.distributed == True:
                consensus_reached = False
                while consensus_reached == False:
                    consensus_reached = True
                    for cam in self.cameras:
                        cam.send_messages(cam.avg_a, cam.avg_F)
                    for cam in self.cameras:
                        cam.calculate_average_consensus()
                    for cam in self.cameras:
                        for neighbor in cam.neighbors:
                            if neighbor.avg_a != cam.avg_a:
                                consensus_reached = False
                            if neighbor.avg_F != cam.avg_F:
                                consensus_reached = False

            for cam in self.cameras:
                filtered_pose = cam.imm.update_pose(timestep = timestep, distributed = self.distributed, a = cam.avg_a, F = cam.avg_F)
                

            self.plotter.update_plot(cam.get_measurements().flatten(), filtered_pose.flatten(), self.target.get_position(), self.target.get_velocity(), self.last_timestamp)
  
        plt.pause(0.01) # this updates both plots
            
        if num == self.n_frames-1:
            plt.close()
            exit()

    def create_cameras(self, n_cameras, sensor_accuracy, initial_pose):
        cameras = []
        for i in range(n_cameras):
            cameras.append(Camera("camera "+str(i), sensor_accuracy, np.array([0, 0, 0, 0, 0, 0]), initial_pose))
        # make following and precedenting cameras neighbors to each camera
        if n_cameras > 2:
            for index, cam in enumerate(cameras):
                if (i-1) < 0:
                    cam.neighbors.append(cameras[-1])
                else:
                    cam.neighbors.append(cameras[i-1])
                if (i+1) <= n_cameras:
                    cam.neighbors.append(cameras[0])
                else:
                    cam.neighbors.append(cameras[i+1])
        if n_cameras == 2:
            cameras[0].neighbors.append(cameras[1])
            cameras[1].neighbors.append(cameras[0])
        return cameras

if __name__ == '__main__':
    n_cameras = 2
    path = np.zeros((3, 3000), dtype = float)
    for i in range(0, path.shape[1]):
        path[0, i] = i/6 #x
        path[1, i] = i/4 #y
        path[2, i] = i/8 #z
    
    sensor_accuracy = 0.1
        
    Main(path = path, n_cameras = n_cameras, sensor_accuracy = sensor_accuracy, n_frames = path.shape[1])