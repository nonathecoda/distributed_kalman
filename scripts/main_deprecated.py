from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import time

from classes.target import Target
from classes.camera import Camera
from classes.imm import InteractingMultipleModel
from scripts.plotters.plotter import Plotter
from kinetic_models.const_velocity import CV_CYPR_Model


class Main():

    def __init__(self, path = None, n_frames = 5, n_cameras = 1, sensor_accuracy = 5) -> None:
        np.set_printoptions(suppress=True, precision=20)
        np.seterr(all='raise')
        np.random.seed(0)

        self.start_time = time.monotonic()
        self.path = path
        if self.path is None:
            self.path = np.array(list(self.gen_path(n_frames))).T # (3, N)
        self.n_frames = n_frames
        if n_cameras > 1:
            self.distributed = True
        else:
            self.distributed = False
        self.cameras = self.create_cameras(n_cameras, sensor_accuracy)
    
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
            self.update_world(i, self.path, line)
            #duration = time.monotonic() - self.start_time
            #sleep_for = max(0,0.2 - duration)
            #time.sleep(sleep_for)
            
            
        #plt.show()


    def update_world(self, num, path, line):
        ic(num)
        self.target.set_position(path[:, num])
        ic(path[:, num])
        if num > 0:
            self.target.set_velocity(path[:, num] - path[:, num-1])
            ic(path[:, num] - path[:, num-1])
        if num > 1:
            self.target.set_acceleration(path[:, num] - 2*path[:, num-1] + path[:, num-2])
            ic(path[:, num] - 2*path[:, num-1] + path[:, num-2])
    
        line.set_data(path[:2, :num])
        line.set_3d_properties(path[2, :num])
        
        ic(self.target.get_position())
        ic(self.target.get_velocity())
        ic(self.target.get_acceleration())
        
        if num%3 == 0 and num > 3: # take measurement every 5 frames
            print("--------------------")
            print("TIMESTEP " + str(int((num/3)-1)))
            timestep = 3
            self.last_timestamp = num
            '''
            ic(self.target.get_position())
            ic(self.target.get_velocity())
            ic(self.target.get_acceleration())
            exit()
            '''
            
            
            for cam in self.cameras:
                cam.take_position_measurement(self.target.get_position())
                cam.take_velocity_measurement(self.target.get_velocity())
                cam.take_acceleration_measurement(self.target.get_acceleration())

            if self.distributed == True:
                # calculate average consensus
                consensus_reached = False
                round = 0
                while consensus_reached == False:
                    #print("Distributed consensus round " + str(round))
                    if round == 100:
                        print("Consensus not reached after 100 rounds")
                        exit()
                    round += 1
                    consensus_reached = True
                    for cam in self.cameras:
                        cam.send_messages(cam.avg_a, cam.avg_F)
                    for cam in self.cameras:
                        cam.calculate_average_consensus()
                    for cam in self.cameras:
                        for neighbor in cam.neighbors:
                            if not np.allclose(neighbor.avg_a, cam.avg_a, atol = 1e-08):
                                consensus_reached = False
                            if not np.allclose(neighbor.avg_F, cam.avg_F,  atol = 1e-08):
                                consensus_reached = False
                print("Cameras reached consensus after " + str(round) + " round(s).")
            
            for cam in self.cameras:
                filtered_pose, models = cam.imm.update_pose(timestep = timestep, distributed = self.distributed, a = cam.avg_a, F = cam.avg_F)
                cam.avg_a = None
                cam.avg_F = None

            self.plotter.update_plot(cam.get_measurements().flatten(), filtered_pose.flatten(), self.target.get_position(), self.target.get_velocity(), self.target.get_acceleration(), self.last_timestamp, models)
            

        plt.pause(0.01) # this updates both plots
            
        if num == self.n_frames-1:
            plt.close()
            exit()

    def create_cameras(self, n_cameras, sensor_accuracy):
        cameras = []
        for i in range(n_cameras):
            cameras.append(Camera(name = "camera "+str(i), accuracy = sensor_accuracy, camera_position = np.array([0, 0, 0, 0, 0, 0])))
       ##accuracy, position, initial_target_pose
       # make following and precedenting cameras neighbors to each camera
        if n_cameras > 2:
            for index, cam in enumerate(cameras):
                if (index) == 0:
                    cam.neighbors.append(cameras[-1])
                else:
                    cam.neighbors.append(cameras[index-1])
                if (index) == n_cameras-1:
                    cam.neighbors.append(cameras[0])
                else:
                    cam.neighbors.append(cameras[index+1])
        if n_cameras == 2:
            cameras[0].neighbors.append(cameras[1])
            cameras[1].neighbors.append(cameras[0])

        return cameras

if __name__ == '__main__':
    n_cameras = 1
    
    path = np.zeros((3, 3000), dtype = float)
    t = 0
    dt = 1
    a = 1
    v = 0
    
    for i in range(0, path.shape[1]):
        #if i > 500 : #constant acceleration
        #    path[0, i] = (i)*i*i#x 
        #    path[1, i] = (i)*i*i #y
        #    path[2, i] = (i)*i*i #z

        if i > 0: #constant acceleration
            t = t+dt
            v = v + dt*a
    
            path[0, i] = path[0, i-1] + dt*v + 0.5*dt**2*a #x 
            path[1, i] = path[1, i-1] + dt*v + 0.5*dt**2*a #y
            path[2, i] = path[2, i-1] + dt*v + 0.5*dt**2*a #z
        else:
            path[0, i] = 0 #x 
            path[1, i] = 0 #y
            path[2, i] = 0 #z
    
    sensor_accuracy = 0.1
    
    Main(path = path, n_cameras = n_cameras, sensor_accuracy = sensor_accuracy, n_frames = path.shape[1])