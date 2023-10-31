
import cv2
import time
import math
from icecream import ic
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
import numpy as np

class Plotter():
    
    def __init__(self):
        fig = plt.figure(figsize=(20, 8))
        plt.title('Kalman Filter')
        plt.axis('off')
        self.time = [0]
        
        # data points position
        self.x_m = [0]
        self.x_f = [0]
        self.x_r = [0]
        self.y_m = [0]
        self.y_f = [0]
        self.y_r = [0]
        self.z_m = [0]
        self.z_f = [0]
        self.z_r = [0]

        # data points velocity
        self.vx_m = [0]
        self.vx_f = [0]
        self.vx_r = [0]
        self.vy_m = [0]
        self.vy_f = [0]
        self.vy_r = [0]
        self.vz_m = [0]
        self.vz_f = [0]
        self.vz_r = [0]

        # data points error
        self.error_x = [0]
        self.error_y = [0]
        self.error_z = [0]
        self.error_vx = [0]
        self.error_vy = [0]
        self.error_vz = [0]

        #data points model probabilities
        self.model1 = [0]
        self.model2 = [0]

        # subplot x position
        fig.add_subplot(3, 3, 1)
        plt.axis([0, 3000, 0, 750])
        self.ln_x_m, = plt.plot(self.time, self.x_m, '-', label = 'x data measured')
        self.ln_x_f, = plt.plot(self.time, self.x_f, '-', label = 'x data filtered')
        self.ln_x_r, = plt.plot(self.time, self.x_r, '-', label = 'x data real')
        plt.legend(handles=[self.ln_x_m, self.ln_x_f, self.ln_x_r])

        # subplot y position
        fig.add_subplot(3, 3, 2)
        plt.axis([0, 3000, 0, 750])
        self.ln_y_m, = plt.plot(self.time, self.y_m, '-', label = 'y data measured')
        self.ln_y_f, = plt.plot(self.time, self.y_f, '-', label = 'y data filtered')
        self.ln_y_r, = plt.plot(self.time, self.y_r, '-', label = 'y data real')
        plt.legend(handles=[self.ln_y_m, self.ln_y_f, self.ln_y_r])

        # subplot z position
        fig.add_subplot(3, 3, 3)
        plt.axis([0, 3000, -50, 400])
        self.ln_z_r, = plt.plot(self.time, self.z_r, '-', label = 'z data real')
        self.ln_z_m, = plt.plot(self.time, self.z_m, '-', label = 'z data measured')
        self.ln_z_f, = plt.plot(self.time, self.z_f, '-', label = 'z data filtered')
        plt.legend(handles=[self.ln_z_r, self.ln_z_m, self.ln_z_f])

        # subplot x velocity
        fig.add_subplot(3, 3, 4)
        plt.axis([0, 3000, -20, 20])
        self.ln_vx_r, = plt.plot(self.time, self.vx_r, '-', label = 'x velocity real')
        self.ln_vx_m, = plt.plot(self.time, self.vx_m, '-', label = 'x velocity measured')
        self.ln_vx_f, = plt.plot(self.time, self.vx_f, '-', label = 'x velocity filtered')
        plt.legend(handles = [self.ln_vx_r, self.ln_vx_m, self.ln_vx_f])

        # subplot y velocity
        fig.add_subplot(3, 3, 5)
        plt.axis([0, 3000, -20, 20])
        self.ln_vy_r, = plt.plot(self.time, self.vy_r, '-', label = 'y velocity real')
        self.ln_vy_m, = plt.plot(self.time, self.vy_m, '-', label = 'y velocity measured')
        self.ln_vy_f, = plt.plot(self.time, self.vy_f, '-', label = 'y velocity filtered')
        plt.legend(handles=[self.ln_vy_r, self.ln_vy_m, self.ln_vy_f])

        # subplot z velocity
        """fig.add_subplot(3, 3, 6)
        plt.axis([0, 3000, -20, 20])
        self.ln_vz_r, = plt.plot(self.time, self.vz_r, '-', label = 'z velocity real')
        self.ln_vz_m, = plt.plot(self.time, self.vz_m, '-', label = 'z velocity measured')
        self.ln_vz_f, = plt.plot(self.time, self.vz_f, '-', label = 'z velocity filtered')
        plt.legend(handles=[self.ln_vz_r, self.ln_vz_m, self.ln_vz_f])
        """

        #subplot model probabilities
        fig.add_subplot(3, 3, 6)
        plt.axis([0, 3000, 0, 1])
        self.ln_model_1, = plt.plot(self.time, self.model1, '-', label = 'model 1')
        self.ln_model_2, = plt.plot(self.time, self.model2, '-', label = 'model 2')
        plt.legend(handles=[self.ln_model_1, self.ln_model_2])

        # subplot error in x
        fig.add_subplot(3, 3, 7)
        plt.axis([0, 3000, 0, 30])
        self.ln_error_x, = plt.plot(self.time, self.error_x, '-', label = 'x error', color='red')
        self.ln_error_vx, = plt.plot(self.time, self.error_vx, '-', label = 'x velocity error', color='green')
        plt.legend(handles=[self.ln_error_x, self.ln_error_vx])
        plt.title('Error in X')

        # subplot error in y
        fig.add_subplot(3, 3, 8)
        plt.axis([0, 3000, 0, 30])
        self.ln_error_y, = plt.plot(self.time, self.error_y, '-', label = 'y error', color='red')
        self.ln_error_vy, = plt.plot(self.time, self.error_vy, '-', label = 'y velocity error', color='green')
        plt.legend(handles=[self.ln_error_y, self.ln_error_vy])
        plt.title('Error in Y')

        # subplot error in z
        fig.add_subplot(3, 3, 9)
        plt.axis([0, 3000, 0, 30])
        self.ln_error_z, = plt.plot(self.time, self.error_z, '-', label = 'z error', color='red')
        self.ln_error_vz, = plt.plot(self.time, self.error_vz, '-', label = 'z velocity error', color='green')
        plt.legend(handles=[self.ln_error_z, self.ln_error_vz])
        plt.title('Error in Z')

        plt.ion()
        print('Initialised Plotter')
        

    def update_plot(self, measurements, filtered_pose, real_position, real_velocity, timestamp, models):
        
        self.time.append(timestamp)

        # Plotting x position
        self.x_m.append(measurements[0])
        self.x_f.append(filtered_pose[0])
        self.x_r.append(real_position[0])
        self.ln_x_m.set_data(self.time, self.x_m)
        self.ln_x_r.set_data(self.time, self.x_r)
        self.ln_x_f.set_data(self.time, self.x_f)
        

        # Plotting y position
        self.y_m.append(measurements[3])
        self.y_f.append(filtered_pose[3])
        self.y_r.append(real_position[1])
        self.ln_y_m.set_data(self.time, self.y_m)
        self.ln_y_r.set_data(self.time, self.y_r)
        self.ln_y_f.set_data(self.time, self.y_f)
        
        # Plotting z position
        self.z_m.append(measurements[6])
        self.z_f.append(filtered_pose[6])
        self.z_r.append(real_position[2])
        self.ln_z_m.set_data(self.time, self.z_m)
        self.ln_z_r.set_data(self.time, self.z_r)
        self.ln_z_f.set_data(self.time, self.z_f)

        # Plotting x velocity
        self.vx_m.append(measurements[1])
        self.vx_f.append(filtered_pose[1])
        self.vx_r.append(real_velocity[0])
        self.ln_vx_m.set_data(self.time, self.vx_m)
        self.ln_vx_r.set_data(self.time, self.vx_r)
        self.ln_vx_f.set_data(self.time, self.vx_f)

        # Plotting y velocity
        self.vy_m.append(measurements[4])
        self.vy_f.append(filtered_pose[4])
        self.vy_r.append(real_velocity[1])
        self.ln_vy_m.set_data(self.time, self.vy_m)
        self.ln_vy_r.set_data(self.time, self.vy_r)
        self.ln_vy_f.set_data(self.time, self.vy_f)

        # Plotting z velocity
        '''
        self.vz_m.append(measurements[7])
        self.vz_f.append(filtered_pose[7])
        self.vz_r.append(real_velocity[2])
        self.ln_vz_m.set_data(self.time, self.vz_m)
        self.ln_vz_r.set_data(self.time, self.vz_r)
        self.ln_vz_f.set_data(self.time, self.vz_f)
        '''

        # Plotting error in x position and velocity
        avg_error_x = np.average(abs(np.subtract(self.x_f, self.x_r)))
        self.error_x.append(avg_error_x)
        self.ln_error_x.set_data(self.time, self.error_x)
        avg_error_vx = np.average(abs(np.subtract(self.vx_f, self.vx_r)))
        self.error_vx.append(avg_error_vx)
        self.ln_error_vx.set_data(self.time, self.error_vx)

        # Plotting error in y position
        avg_error_y = np.average(abs(np.subtract(self.y_f, self.y_r)))
        self.error_y.append(avg_error_y)
        self.ln_error_y.set_data(self.time, self.error_y)
        avg_error_vy = np.average(abs(np.subtract(self.vy_f, self.vy_r)))
        self.error_vy.append(avg_error_vy)
        self.ln_error_vy.set_data(self.time, self.error_vy)
       
        # Plotting error in z position
        avg_error_z = np.average(abs(np.subtract(self.z_f, self.z_r)))
        self.error_z.append(avg_error_z)
        self.ln_error_z.set_data(self.time, self.error_z)
        avg_error_vz = np.average(abs(np.subtract(self.vz_f, self.vz_r)))
        self.error_vz.append(avg_error_vz)
        self.ln_error_vz.set_data(self.time, self.error_vz)

        # Plotting model probabilities
        self.model1.append(models[0].model_probability)
        self.model2.append(models[1].model_probability)
        self.ln_model_1.set_data(self.time, self.model1)
        self.ln_model_2.set_data(self.time, self.model2)
