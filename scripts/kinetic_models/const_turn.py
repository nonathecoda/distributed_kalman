from icecream import ic
import numpy as np
import sys
from classes.kalman import KalmanFilter

class Const_Turn_Model(KalmanFilter):
    '''
    constant velocity,
    constant turn rate,
    moving in z plane
    from https://www.hindawi.com/journals/mpe/2014/649276/
    '''
    def __init__(self, std_dev_process_noise_q, measurement_noise_r, initial_pose, name = "orbit"):
        self.name = name
        #initial_pose, H, measurement_noise_r, std_dev_process_noise_q
        H = np.array([  [1,     0,  0,  0,  0],
                        [0,     0,  1,  0,  0],
                        [0,     0,  0,  0,  1]])

        super().__init__(initial_pose, H, measurement_noise_r, std_dev_process_noise_q)
        self.dims = H.shape[1]
        print(self.name)
            
    @property
    def dt(self):
        return self.dt

    @dt.setter
    def dt(self, dt):
        w = np.pi/5
        print("Orbit model")
        ic(dt)
        self.state_transition_matrix =  np.array([[1,    np.sin(w*dt)/w,        0,      -((1-np.cos(w*dt))/w),  0],
                                                  [0,    np.cos(w*dt),          0,      -np.sin(w*dt),          0],
                                                  [0,    ((1-np.cos(w*dt))/w),  1,      np.sin(w*dt)/w,         0],
                                                  [0,    np.sin(w*dt),          0,      np.cos(w*dt),           0],
                                                  [0,    0,                     0,      0,                      1]])

        G = np.array([[0.5*np.square(dt)  ],
                      [dt                 ],
                      [0.5*np.square(dt)  ],
                      [dt                 ],
                      [0.5*np.square(dt)  ]])
        self.process_noise_matrix = (G @ np.transpose(G)) * np.square(self.std_dev_process_noise)
        #self.process_noise_matrix = np.eye(self.state_transition_matrix.shape[0])
    
    
   