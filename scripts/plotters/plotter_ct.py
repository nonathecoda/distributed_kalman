
import cv2
import time
import math
from icecream import ic
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
import numpy as np

class Plotter_CT():
    
    def __init__(self, initial_pos, initial_vel, initial_accel):
        fig = plt.figure(figsize=(20, 8))
        #plt.title('Kalman Filter')
        plt.axis('off')
        self.time = [-1]
        
        # data points position
        self.x_m = [0]
        self.x_f = [0]
        self.x_r = [initial_pos[0]]
        # data points velocity
        self.vx_m = [0]
        self.vx_f = [0]
        self.vx_r = [initial_vel[0]]
        #data points acceleration
        self.ax_m = [0]
        self.ax_f = [0]
        self.ax_r = [initial_accel[0]]
        # data points error
        self.error_x = [0]
        self.error_vx = [0]
        self.error_ax = [0]
        self.error_mx = [0]

        #data points model probabilities
        self.model1 = [0]
        self.model2 = [0]

        # subplot x position
        fig.add_subplot(2, 3, 1)
        plt.axis([0, 5, 0, 50])
        self.ln_x_m, = plt.plot(self.time, self.x_m, '-', label = 'measured', color='orangered')
        self.ln_x_r, = plt.plot(self.time, self.x_r, '-', label = 'real', color='blueviolet')
        self.ln_x_f, = plt.plot(self.time, self.x_f, '-', label = 'filtered', color='yellowgreen')
        plt.legend(handles=[self.ln_x_r, self.ln_x_f, self.ln_x_m])
        plt.title('x position')
        plt.xlabel('seconds', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('metres', loc = 'top', labelpad = 2.0, fontsize=9)

        # subplot x velocity
        fig.add_subplot(2, 3, 2)
        plt.axis([0, 5, -35, 35])
        self.ln_vx_r, = plt.plot(self.time, self.vx_r, '-', label = 'real', color='blueviolet')
        self.ln_vx_f, = plt.plot(self.time, self.vx_f, '-', label = 'filtered', color='yellowgreen')
        plt.legend(handles = [self.ln_vx_r, self.ln_vx_f])
        plt.title('x velocity')
        plt.xlabel('seconds', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('metres/sec', loc = 'top', labelpad = 2.0, fontsize=9)

        # subplot x acceleration
        fig.add_subplot(2, 3, 3)
        plt.axis([0, 5, -100, 100])
        self.ln_ax_r, = plt.plot(self.time, self.ax_r, '-', label = 'real', color='blueviolet')
        self.ln_ax_f, = plt.plot(self.time, self.ax_f, '-', label = 'filtered', color='yellowgreen')
        plt.legend(handles=[self.ln_ax_r, self.ln_ax_f])
        plt.title('x acceleration')
        plt.xlabel('seconds', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('metres/sec^2', loc = 'top', labelpad = 2.0, fontsize=9)

        # subplot x error
        fig.add_subplot(2, 3, 4)
        plt.axis([0, 5, 0, 10])
        self.ln_error_x, = plt.plot(self.time, self.error_x, '-', label = 'RMSD Filtered', color='red')
        self.ln_error_mx, = plt.plot(self.time, self.error_x, '-', label = 'RMSD Measurements', color='orange')
        plt.legend(handles=[self.ln_error_x, self.ln_error_mx])
        plt.title('RMSD x position')
        plt.xlabel('seconds', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('metres', loc = 'top', labelpad = 2.0, fontsize=9)

        # subplot x velocity error
        fig.add_subplot(2, 3, 5)
        plt.axis([0, 5, 0, 10])
        self.ln_error_vx, = plt.plot(self.time, self.error_vx, '-', label = 'RMSD Filtered', color='red')
        plt.legend(handles=[self.ln_error_vx])
        plt.title('RMSD x velocity')
        plt.xlabel('seconds', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('metres/sec', loc = 'top', labelpad = 2.0, fontsize=9)

        # subplot x acceleration error
        fig.add_subplot(2, 3, 6)
        plt.axis([0, 5, 0, 5])
        self.ln_error_ax, = plt.plot(self.time, self.error_ax, '-', label = 'RMSD Filtered', color='red')
        plt.legend(handles=[self.ln_error_ax])
        plt.title('RMSD x acceleration')
        plt.xlabel('seconds', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('metres/sec^2', loc = 'top', labelpad = 2.0, fontsize=9)
        
        plt.ion()
        print('Initialised Plotter')
        

    def update_plot(self, measurements, filtered_pose, real_position, real_velocity, real_acceleration, timestamp, models):
        
        self.time.append(timestamp-0.1)
        # Plotting x position
        self.x_m.append(measurements[0])
        self.x_f.append(filtered_pose[0])
        self.x_r.append(real_position[0])
        self.ln_x_m.set_data(self.time, self.x_m)
        self.ln_x_r.set_data(self.time, self.x_r)
        self.ln_x_f.set_data(self.time, self.x_f)
        
        # Plotting x velocity
        self.vx_f.append(filtered_pose[1])
        self.vx_r.append(real_velocity[0])
        self.ln_vx_f.set_data(self.time, self.vx_f)
        self.ln_vx_r.set_data(self.time, self.vx_r)

        # Plotting x acceleration
        self.ax_f.append(models[0].updated_state[2][0]) #filtered pose is 6d! TODO: fix this 
        self.ax_r.append(real_acceleration[0])
        self.ln_ax_r.set_data(self.time, self.ax_r)
        self.ln_ax_f.set_data(self.time, self.ax_f)

        # Plotting RMSD for x position
        rmsd_x = np.sqrt(np.mean((np.subtract(self.x_f,self.x_r))**2))
        self.error_x.append(rmsd_x)
        self.ln_error_x.set_data(self.time, self.error_x)

        rmsd_mx = np.sqrt(np.mean((np.subtract(self.x_m,self.x_r))**2))
        self.error_mx.append(rmsd_mx)
        self.ln_error_mx.set_data(self.time, self.error_mx)

        # Plotting RMSD for x velocity
        rmsd_vx = np.sqrt(np.mean((np.subtract(self.vx_f,self.vx_r))**2))
        self.error_vx.append(rmsd_vx)
        self.ln_error_vx.set_data(self.time, self.error_vx)

        # Plotting RMSD for x acceleration
        rmsd_ax = np.sqrt(np.mean((np.subtract(self.ax_f,self.ax_r))**2))
        self.error_ax.append(rmsd_ax)
        self.ln_error_ax.set_data(self.time, self.error_ax)