from icecream import ic
import numpy as np
import sys
from classes.kalman import KalmanFilter

class CV_CYPR_Model(KalmanFilter):
    '''
    constant velocity,
    constant yaw pitch roll angles.
    '''
    def __init__(self, std_dev_process_noise_q, measurement_noise_r, initial_pose, name = "cv"):
        self.name = name
        #initial_pose, H, measurement_noise_r, std_dev_process_noise_q
        H = np.array([  [1,     0,  0,  0,  0,  0],
                        [0,     0,  1,  0,  0,  0],
                        [0,     0,  0,  0,  1,  0]])

        super().__init__(initial_pose, H, measurement_noise_r, std_dev_process_noise_q)
        self.dims = H.shape[1]
        print(self.name)
            
    @property
    def dt(self):
        return self.dt

    @dt.setter
    def dt(self, timestep):
        self.state_transition_matrix =  np.array([[1,    timestep,  0,      0,          0,          0           ],
                                                  [0,    1,         0,      0,          0,          0           ],
                                                  [0,    0,         1,      timestep,   0,          0           ],
                                                  [0,    0,         0,      1,          0,          0           ],
                                                  [0,    0,         0,      0,          1,          timestep    ],
                                                  [0,    0,         0,      0,          0,          1           ]])

        G = np.array([[0.5*np.square(timestep)  ],
                      [timestep                 ],
                      [0.5*np.square(timestep)  ],
                      [timestep                 ],
                      [0.5*np.square(timestep)  ],
                      [timestep                 ]])
        self.process_noise_matrix = (G @ np.transpose(G)) * np.square(self.std_dev_process_noise)
        #self.process_noise_matrix = np.eye(self.state_transition_matrix.shape[0])
    
    
   