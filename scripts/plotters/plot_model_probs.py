
import cv2
import time
import math
from icecream import ic
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
import numpy as np

class Plotter_Model_Probs():
    
    def __init__(self, initial_pos, initial_vel, initial_accel):
        fig = plt.figure(figsize=(20, 8))
        self.time = [0]

        #data points model probabilities
        self.model1 = [0]
        self.model2 = [0]
        self.model3 = [0]

        #plot model probabilities
        plt.title('Model Probabilities')
        plt.axis([0, 50, -0.1, 1.1])
        self.ln_model_1, = plt.plot(self.time, self.model1, '-', label = 'cv')
        self.ln_model_2, = plt.plot(self.time, self.model2, '-', label = 'ca')
        self.ln_model_3, = plt.plot(self.time, self.model3, '-', label = 'ct')
        plt.legend(handles=[self.ln_model_1, self.ln_model_2, self.ln_model_3])
        plt.xlabel('time (0.1 sec)', loc = 'right', labelpad = 2.0, fontsize=18)
        plt.ylabel('model probability', loc = 'top', labelpad = 2.0, fontsize=18)

        self.initialised = False

        plt.ion()
        print('Initialised Plotter')
        

    def update_plot(self, timestamp, models):
        
        if self.initialised == False:
            self.initialised = True
            self.model1.pop(0)
            self.model2.pop(0)
            self.model3.pop(0)
            self.time.pop(0)

        self.time.append(timestamp)
    
        # Plotting model probabilities
        self.model1.append(models[0].model_probability)
        self.ln_model_1.set_data(self.time, self.model1)
        self.model2.append(models[1].model_probability)
        self.ln_model_2.set_data(self.time, self.model2)
        self.model3.append(models[2].model_probability)
        self.ln_model_3.set_data(self.time, self.model3)
