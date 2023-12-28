
import cv2
import time
import math
from icecream import ic
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
import numpy as np

class Plotter():
    
    def __init__(self, initial_pos, initial_vel, initial_accel):
        fig = plt.figure(figsize=(20, 8))
        plt.title('Kalman Filter')
        plt.axis('off')
        self.time = [0]
        
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
        #self.ax_f = [0]
        self.ax_r = [initial_accel[0]]
        # data points error
        self.error_x = [0]
        self.error_vx = [0]
        self.error_ax = [0]

        #data points model probabilities
        self.model1 = [0]
        self.model2 = [0]

        # subplot x position
        fig.add_subplot(3, 2, 1)
        plt.axis([0, 30, -100, 1000])
        self.ln_x_m, = plt.plot(self.time, self.x_m, '-', label = 'measured')
        self.ln_x_f, = plt.plot(self.time, self.x_f, '-', label = 'filtered')
        self.ln_x_r, = plt.plot(self.time, self.x_r, '-', label = 'real')
        plt.legend(handles=[self.ln_x_f, self.ln_x_r, self.ln_x_m])
        plt.title('x position')

        # subplot x velocity
        fig.add_subplot(3, 2, 2)
        plt.axis([0, 30, -100, 200])
        self.ln_vx_m, = plt.plot(self.time, self.vx_m, '-', label = 'measured')
        self.ln_vx_f, = plt.plot(self.time, self.vx_f, '-', label = 'filtered')
        self.ln_vx_r, = plt.plot(self.time, self.vx_r, '-', label = 'real')
        plt.legend(handles = [self.ln_vx_f, self.ln_vx_r, self.ln_vx_m])
        plt.title('x velocity')

        # subplot x acceleration
        fig.add_subplot(3, 2, 3)
        plt.axis([0, 30, -50, 100])
        #self.ln_ax_f, = plt.plot(self.time, self.ax_f, '-', label = 'filtered')
        self.ln_ax_r, = plt.plot(self.time, self.ax_r, '-', label = 'real')
        plt.legend(handles=[self.ln_ax_r])
        plt.title('x acceleration')

        # subplot x error
        fig.add_subplot(3, 2, 4)
        plt.axis([0, 30, 0, 50])
        self.ln_error_x, = plt.plot(self.time, self.error_x, '-', label = 'x position error', color='red')
        self.ln_error_vx, = plt.plot(self.time, self.error_vx, '-', label = 'x velocity error', color='blue')
        #self.ln_error_ax, = plt.plot(self.time, self.error_ax, '-', label = 'x acceleration error', color='green')
        plt.legend(handles=[self.ln_error_x, self.ln_error_vx])
        plt.title('Error in x')

        #subplot model probabilities
        fig.add_subplot(3, 2, 5)
        plt.axis([0, 30, 0, 1])
        self.ln_model_1, = plt.plot(self.time, self.model1, '-', label = 'cv')
        self.ln_model_2, = plt.plot(self.time, self.model2, '-', label = 'ca')
        plt.legend(handles=[self.ln_model_1, self.ln_model_2])
        plt.title('Model probabilities')

        fig.add_subplot(3, 2, 6)
        
        self.covariance = plt.imshow(np.identity(9), cmap='PiYG', interpolation='nearest')

        plt.ion()
        print('Initialised Plotter')
        

    def update_plot(self, measurements, filtered_pose, real_position, real_velocity, real_acceleration, timestamp, models):
        
        self.time.append(timestamp)
        # Plotting x position
        self.x_m.append(measurements[0])
        self.x_f.append(filtered_pose[0])
        self.x_r.append(real_position[0])
        self.ln_x_m.set_data(self.time, self.x_m)
        self.ln_x_r.set_data(self.time, self.x_r)
        self.ln_x_f.set_data(self.time, self.x_f)
        
        # Plotting x velocity
        #self.vx_m.append(measurements[0])
        self.vx_f.append(filtered_pose[1])
        self.vx_r.append(real_velocity[0])
        #self.ln_vx_m.set_data(self.time, self.vx_m)
        self.ln_vx_f.set_data(self.time, self.vx_f)
        self.ln_vx_r.set_data(self.time, self.vx_r)

        # Plotting x acceleration
        #self.ax_m.append(measurements[0])
        #self.ax_f.append(filtered_pose[2]) filtered pose is 6d!
        self.ax_r.append(real_acceleration[0])
        #self.ln_ax_m.set_data(self.time, self.ax_m)
        self.ln_ax_r.set_data(self.time, self.ax_r)
        #self.ln_ax_f.set_data(self.time, self.ax_f)

        # Plotting error in x position and velocity
        avg_error_x = np.average(abs(np.subtract(self.x_f, self.x_r)))
        self.error_x.append(avg_error_x)
        self.ln_error_x.set_data(self.time, self.error_x)
        avg_error_vx = np.average(abs(np.subtract(self.vx_f, self.vx_r)))
        self.error_vx.append(avg_error_vx)
        self.ln_error_vx.set_data(self.time, self.error_vx)
        #avg_error_ax = np.average(abs(np.subtract(self.ax_f, self.ax_r)))
        #self.error_ax.append(avg_error_ax)
        #self.ln_error_ax.set_data(self.time, self.error_ax)

        # Plotting model probabilities
        self.model1.append(models[0].model_probability)
        self.model2.append(models[1].model_probability)
        self.ln_model_1.set_data(self.time, self.model1)
        self.ln_model_2.set_data(self.time, self.model2)

        # Plotting covariance
        self.covariance.set_data(models[0].updated_covariance/100)