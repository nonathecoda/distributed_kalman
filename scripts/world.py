from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import time

from classes.target import Target
from classes.camera import Camera
from classes.imm import InteractingMultipleModel
from kinetic_models.const_velocity import CV_CYPR_Model
from classes.path import get_figure_eight, get_constant_acceleration, get_constant_velocity, get_constant_turn, get_ca_orbit_cv, get_orbit
from plotters.plotter_ca import Plotter_CA
from plotters.plotter import Plotter
from plotters.plotter_cv import Plotter_CV
from plotters.plotter_ct import Plotter_CT

class World():
    def __init__(self):
        self.MEASUREMENT_FREQUENCY = 1
        TURN_RATE = 1
        self.DELAY = 2
        w = np.pi/5

        # create path
        initial_pos = np.array([200, 2, 50]) #m
        initial_vel = np.array([20, 25, 25]) #m/dt*seconds
        initial_accel = np.array([5, 5, 5]) #m/(dt*seconds)^2

        self.dt = 0.1 #seconds
        self.coordinates, self.velocities, self.accelerations, self.timestamps = get_ca_orbit_cv(w, initial_pos, initial_vel, initial_accel, self.dt)
        #self.coordinates, self.velocities, self.accelerations, self.timestamps = get_figure_eight(dt = self.dt)

        # create cameras
        N_CAMERAS = 2
        SENSOR_NOISE = 5
        if N_CAMERAS > 1:
            self.distributed = True
        else:
            self.distributed = False
        self.cameras = self.create_cameras(N_CAMERAS, SENSOR_NOISE)
        
        # create target
        self.target = Target("target", initial_pos = initial_pos)

        # create plotter for graphs
        #self.plotter = Plotter(initial_pos, initial_vel, initial_accel)
        self.plotter = Plotter(initial_pos, initial_vel, initial_accel)

        # create animation
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(projection='3d')
        self.line_real, = ax.plot(self.coordinates[0, 0:1], self.coordinates[1, 0:1], self.coordinates[2, 0:1])
        self.line_filtered, = ax.plot(self.coordinates[0, 0:1], self.coordinates[1, 0:1], self.coordinates[2, 0:1])

        self.filtered_coordinates = np.zeros((3, 3000), dtype = float)
        
        ax.set_xlim3d([-1.0, 4000.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-1.0, 4000.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0.0, 4000.0])
        ax.set_zlabel('Z')
        
        for k in range(0, self.coordinates.shape[1]):
            self.update_world(k)
        
    def update_world(self, k):
        # update target position
        self.target.set_position(self.coordinates[:, k])
        self.target.set_velocity(self.velocities[:, k])
        self.target.set_acceleration(self.accelerations[:, k])
        
        # update animation
        self.line_real.set_data(self.coordinates[:2, :k])
        self.line_real.set_3d_properties(self.coordinates[2, :k])
        
        # filter step
        if k%self.MEASUREMENT_FREQUENCY == 0 and k > self.DELAY: # take measurement every x timesteps
            print("--------------------")
            print("MEASUREMENT " + str(int(k/self.MEASUREMENT_FREQUENCY-self.DELAY)))
            
            for cam in self.cameras:
                cam.take_position_measurement(self.target.get_position())
                cam.take_velocity_measurement(self.target.get_velocity())
                cam.take_acceleration_measurement(self.target.get_acceleration())
            
            ic(self.target.get_position()[0])
            ic(self.target.get_velocity()[0])

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
                        cam.send_messages_imm()
                    for cam in self.cameras:
                        cam.calculate_average_consensus()
                    for cam in self.cameras:
                        for neighbor in cam.neighbors:
                            for model_index in range(len(neighbor.imm.models)):
                                if not np.allclose(neighbor.imm.models[model_index].avg_a, cam.imm.models[model_index].avg_a, atol = 1e-08):
                                    consensus_reached = False
                                if not np.allclose(neighbor.imm.models[model_index].avg_F, cam.imm.models[model_index].avg_F, atol = 1e-08):
                                    consensus_reached = False
                            
                print("Cameras reached consensus after " + str(round) + " round(s).")
                
            for cam in self.cameras:
                filtered_pose, models = cam.imm.update_pose(timestep = self.dt, distributed = self.distributed)
                for model in cam.imm.models:
                    model.avg_a = None
                    model.avg_F = None
            
            self.plotter.update_plot(cam.get_measurements().flatten(), filtered_pose.flatten(), self.target.get_position(), self.target.get_velocity(), self.target.get_acceleration(), self.timestamps[k], models)
            
            
            #update 3d animation with filtered pose
            self.filtered_coordinates[:, k] = ([filtered_pose[0], filtered_pose[2], filtered_pose[4]])
            self.line_filtered.set_data(self.filtered_coordinates[:2, :k])
            self.line_filtered.set_3d_properties(self.filtered_coordinates[2, :k])

            
        
        plt.pause(0.01) # this updates both plots
        
        if k == 3000-1:
        #if k == 3:
            plt.close()
            exit()
        
    def create_cameras(self, n_cameras, sensor_noise):
        initial_target_state = [self.coordinates[0, self.MEASUREMENT_FREQUENCY-1 + self.DELAY], self.velocities[0, self.MEASUREMENT_FREQUENCY-1 +self.DELAY],self.accelerations[0, self.MEASUREMENT_FREQUENCY-1+self.DELAY], self.coordinates[1, self.MEASUREMENT_FREQUENCY-1+self.DELAY], self.velocities[1, self.MEASUREMENT_FREQUENCY-1+self.DELAY],self.accelerations[1, self.MEASUREMENT_FREQUENCY-1+self.DELAY], self.coordinates[2, self.MEASUREMENT_FREQUENCY-1+self.DELAY], self.velocities[2, self.MEASUREMENT_FREQUENCY-1+self.DELAY],self.accelerations[2, self.MEASUREMENT_FREQUENCY-1+self.DELAY]]
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
    