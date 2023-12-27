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

class World():
    def __init__(self):
        self.MEASUREMENT_FREQUENCY = 1

        # create path
        self.ACCELERATION = np.array([50, 50, 50])

        initial_pos = np.array([0, 0, 0])
        initial_vel = np.array([30, 30, 30])
        #initial_accel = np.array([0, 0, 0])
        initial_accel = self.ACCELERATION
        self.dt = 0.01
        self.coordinates, self.velocities, self.accelerations, self.timestamps = self.create_path(initial_pos, initial_vel, initial_accel, self.dt)

        # create cameras
        N_CAMERAS = 3
        SENSOR_NOISE = 0.1
        if N_CAMERAS > 1:
            self.distributed = True
        else:
            self.distributed = False
        self.cameras = self.create_cameras(N_CAMERAS, SENSOR_NOISE)
        
        # create target
        self.target = Target("target", initial_pos = initial_pos)

        # create plotter for graphs
        self.plotter = Plotter(initial_pos, initial_vel, initial_accel)

        # create animation
        fig = plt.figure(figsize=(1, 1))
        ax = fig.add_subplot(projection='3d')
        self.line, = ax.plot(self.coordinates[0, 0:1], self.coordinates[1, 0:1], self.coordinates[2, 0:1])
        
        ax.set_xlim3d([-1.0, 500.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-1.0, 500.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0.0, 300.0])
        ax.set_zlabel('Z')
        
        for k in range(0, self.coordinates.shape[1]):
            self.update_world(k)
        
    def update_world(self, k):
        # update target position
        self.target.set_position(self.coordinates[:, k])
        self.target.set_velocity(self.velocities[:, k])
        self.target.set_acceleration(self.accelerations[:, k])
        ic(k)
        ic(self.target.get_position())
        ic(self.target.get_velocity())
        ic(self.target.get_acceleration())
        
        # update animation
        self.line.set_data(self.coordinates[:2, :k])
        self.line.set_3d_properties(self.coordinates[2, :k])
        
        # filter step
        if k%self.MEASUREMENT_FREQUENCY == 0 and k > 0: # take measurement every x timesteps
            print("--------------------")
            print("MEASUREMENT " + str(int(k/self.MEASUREMENT_FREQUENCY)))
           
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
                filtered_pose, models = cam.imm.update_pose(timestep = self.dt, distributed = self.distributed, a = cam.avg_a, F = cam.avg_F)
                cam.avg_a = None
                cam.avg_F = None

            self.plotter.update_plot(cam.get_measurements().flatten(), filtered_pose.flatten(), self.target.get_position(), self.target.get_velocity(), self.target.get_acceleration(), self.timestamps[k], models)
            

        plt.pause(0.01) # this updates both plots
            
        if k == 3000-1:
            plt.close()
            exit()

    def create_path(self, initial_pos, initial_vel, initial_accel, dt):
        # initial_pos: initial position of the object (3d vector)
        # initial_vel: initial velocity of the object (3d vector)
        # initial_accel: initial acceleration of the object (3d vector)
        # dt: time step
        # return: coordinates, velocities, accelerations, timestamps

        coordinates = np.zeros((3, 3000), dtype = float)
        velocities = np.zeros((3, 3000), dtype = float)
        accelerations = np.zeros((3, 3000), dtype = float)
        timestamps = np.zeros((3000), dtype = float)

        for i in range(3000):
            if i == 0:
                coordinates[:, i] = initial_pos
                velocities[:, i] = initial_vel
                accelerations[:, i] = initial_accel
                timestamps[i] = 0
            if i > 300:
                coordinates[:, i] = coordinates[:, i-1] + velocities[:, i-1] * dt
                velocities[:, i] = velocities[:, i-1]
                accelerations[:, i] = np.array([0, 0, 0])
                timestamps[i] = timestamps[i-1] + dt
            else:
                coordinates[:, i] = coordinates[:, i-1] + velocities[:, i-1] * dt + 0.5 * accelerations[:, i-1] * dt**2
                velocities[:, i] = velocities[:, i-1] + accelerations[:, i-1] * dt
                accelerations[:, i] = self.ACCELERATION
                timestamps[i] = timestamps[i-1] + dt

        return coordinates, velocities, accelerations, timestamps

    def create_cameras(self, n_cameras, sensor_noise):
        initial_target_state = [self.coordinates[0, self.MEASUREMENT_FREQUENCY-1], self.velocities[0, self.MEASUREMENT_FREQUENCY-1],self.accelerations[0, self.MEASUREMENT_FREQUENCY-1], self.coordinates[1, self.MEASUREMENT_FREQUENCY-1], self.velocities[1, self.MEASUREMENT_FREQUENCY-1],self.accelerations[1, self.MEASUREMENT_FREQUENCY-1], self.coordinates[2, self.MEASUREMENT_FREQUENCY-1], self.velocities[2, self.MEASUREMENT_FREQUENCY-1],self.accelerations[2, self.MEASUREMENT_FREQUENCY-1]]
        ic(initial_target_state)
        
        cameras = []
        for i in range(n_cameras):
            cameras.append(Camera(name = "camera "+str(i), noise = sensor_noise, camera_position = np.array([0, 0, 0, 0, 0, 0]), initial_target_state = initial_target_state))
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
    # print options
    np.set_printoptions(suppress=True, precision=20)
    np.seterr(all='raise')
    np.random.seed(0)
    World()
    