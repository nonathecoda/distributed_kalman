
import cv2
import time
import math
from icecream import ic
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
import numpy as np

class Plotter_CV():
    
    def __init__(self, initial_pos, initial_vel, initial_accel):
        fig = plt.figure(figsize=(20, 8))
        #plt.title('Kalman Filter')
        plt.axis('off')
        self.time = [-1]

        self.initialised = False
        
        # data points position
        self.x_m = [0]
        self.x_f = [0]
        self.x_r = [0]
        # data points velocity
        self.vx_m = [0]
        self.vx_f = [0]
        self.vx_r = [0]
        #data points acceleration
        self.ax_m = [0]
        self.ax_f = [0]
        self.ax_r = [0]
        # data points error
        self.error_fx = [0]
        self.error_vx = [0]
        self.error_ax = [0]
        self.error_mx = [0]

        #rmsd stuf
        self.rmsd_mx = [0]
        self.rmsd_x = [0]

        #data points model probabilities
        self.model1 = [0]
        self.model2 = [0]
        self.model3 = [0]
        self.model4 = [0]

        '''
        # subplot x position
        fig.add_subplot(2, 1, 1)
        plt.axis([0, 100, 0, 1000])
        self.ln_x_m, = plt.plot(self.time, self.x_m, '-', label = 'measured', color='orangered')
        self.ln_x_r, = plt.plot(self.time, self.x_r, '-', label = 'real', color='blueviolet')
        self.ln_x_f, = plt.plot(self.time, self.x_f, '-', label = 'filtered', color='yellowgreen')
        plt.legend(handles=[self.ln_x_r, self.ln_x_f, self.ln_x_m])
        plt.title('x position')
        plt.xlabel('time (0.1 sec)', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('distance(m)', loc = 'top', labelpad = 2.0, fontsize=9)
        '''
        #subplot model probabilities
        fig.add_subplot(2, 1, 1)
        plt.axis([0, 15, -0.1, 1.1])
        self.ln_model_1, = plt.plot(self.time, self.model1, '-', label = 'cv')
        self.ln_model_2, = plt.plot(self.time, self.model2, '-', label = 'ca')
        self.ln_model_3, = plt.plot(self.time, self.model3, '-', label = 'turn')
        self.ln_model_4, = plt.plot(self.time, self.model4, '-', label = 'cp')
        plt.legend(handles=[self.ln_model_1, self.ln_model_2, self.ln_model_3, self.ln_model_4])
        plt.title('Model probabilities')
        plt.xlabel('time (0.1 sec)', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('probability', loc = 'top', labelpad = 2.0, fontsize=9)

        # subplot error x position
        fig.add_subplot(2, 1, 2)
        plt.axis([0, 15, -2, 50])
        self.ln_error_mx, = plt.plot(self.time, self.error_mx, '-', label = 'measured', color='orangered')
        self.ln_error_fx, = plt.plot(self.time, self.error_fx, '-', label = 'filtered', color='yellowgreen')
        plt.legend(handles = [self.ln_error_fx, self.ln_error_mx])
        plt.title('absolute error in x position')
        plt.xlabel('time (0.1 sec)', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('distance (m)', loc = 'top', labelpad = 2.0, fontsize=9)
        
        
        
        
        '''
        # subplot x velocity
        fig.add_subplot(2, 2, 2)
        plt.axis([0, 100, -0, 50])
        self.ln_vx_r, = plt.plot(self.time, self.vx_r, '-', label = 'real', color='blueviolet')
        self.ln_vx_f, = plt.plot(self.time, self.vx_f, '-', label = 'filtered', color='yellowgreen')
        plt.legend(handles = [self.ln_vx_r, self.ln_vx_f])
        plt.title('x velocity')
        plt.xlabel('time (0.1 sec)', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('velocity (m/0.1 sec)', loc = 'top', labelpad = 2.0, fontsize=9)
        
        
        # subplot x error
        fig.add_subplot(2, 2, 3)
        plt.axis([0, 100, 0, 12])
        self.ln_error_x, = plt.plot(self.time, self.error_x, '-', label = 'RMSD Filtered', color='red')
        self.ln_error_mx, = plt.plot(self.time, self.error_x, '-', label = 'RMSD Measurements', color='orange')
        plt.legend(handles=[self.ln_error_x, self.ln_error_mx])
        plt.title('RMSD x position')
        plt.xlabel('time (0.1 sec)', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('distance (m)', loc = 'top', labelpad = 2.0, fontsize=9)

        # subplot x velocity error
        fig.add_subplot(2, 2, 4)
        plt.axis([0, 100, 0, 10])
        self.ln_error_vx, = plt.plot(self.time, self.error_vx, '-', label = 'RMSD Filtered', color='red')
        plt.legend(handles=[self.ln_error_vx])
        plt.title('RMSD x velocity')
        plt.xlabel('time (0.1 sec)', loc = 'right', labelpad = 2.0, fontsize=9)
        plt.ylabel('velocity (m/0.1 sec', loc = 'top', labelpad = 2.0, fontsize=9)
        '''
        self.start_time = time.time()
        plt.ion()
        print('Initialised Plotter')
        

    def update_plot(self, measurements, filtered_pose, real_position, real_velocity, real_acceleration, timestamp, models):
        
        if self.initialised == False:
            self.time.pop(0)
            # data points position
            self.x_m.pop(0)
            self.x_f.pop(0)
            self.x_r.pop(0)
            # data points velocity
            self.vx_m.pop(0)
            self.vx_f.pop(0)
            self.vx_r.pop(0)
            #data points acceleration
            self.ax_m.pop(0)
            self.ax_f.pop(0)
            self.ax_r.pop(0)
            # data points error
            self.error_fx.pop(0)
            self.error_vx.pop(0)
            self.error_ax.pop(0)
            self.error_mx.pop(0)

            #data points model probabilities
            self.model1.pop(0)
            self.model2.pop(0)
            self.model3.pop(0)
            self.model4.pop(0)

            self.rmsd_mx.pop(0)
            self.rmsd_x.pop(0)
            self.initialised = True

        self.time.append(timestamp)

        
        # Plotting x position
        self.x_m.append(measurements[0])
        self.x_f.append(filtered_pose[0])
        self.x_r.append(real_position[0])
        '''
        self.ln_x_m.set_data(self.time, self.x_m)
        
        self.ln_x_r.set_data(self.time, self.x_r)
        self.ln_x_f.set_data(self.time, self.x_f)
        '''

        # Plotting model probabilities
        self.model1.append(models[0].model_probability)
        self.ln_model_1.set_data(self.time, self.model1)
        self.model2.append(models[1].model_probability)
        self.ln_model_2.set_data(self.time, self.model2)
        self.model3.append(models[2].model_probability)
        self.ln_model_3.set_data(self.time, self.model3)
        self.model4.append(models[3].model_probability)
        self.ln_model_4.set_data(self.time, self.model4)

        #Plot error x position
        self.error_fx.append(abs(real_position[0]-filtered_pose[0]))
        self.error_mx.append(abs(real_position[0]-measurements[0]))
        self.ln_error_fx.set_data(self.time, self.error_fx)
        self.ln_error_mx.set_data(self.time, self.error_mx)

        # Calculate RMSD stuff
        rmsd_x = np.sqrt(np.mean((np.subtract(self.x_f,self.x_r))**2))
        self.rmsd_x.append(rmsd_x)
        
        rmsd_mx = np.sqrt(np.mean((np.subtract(self.x_m,self.x_r))**2))
        self.rmsd_mx.append(rmsd_mx)
        
        
        if timestamp > 20:
            end_time = time.time()
            ic(len(self.time))
            ic((end_time - self.start_time)/len(self.time))
            ic(rmsd_x)
            ic(rmsd_mx)
            #plt.savefig('/Users/antonia/Desktop/Plots/plot_cv.png')
            exit()
        
          