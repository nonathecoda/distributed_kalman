from icecream import ic
import numpy as np
from classes.kalman import KalmanFilter

class CV_CYPR_Model(KalmanFilter):
    '''
    constant velocity,
    constant yaw pitch roll angles.
    '''
    def __init__(self, std_dev, initial_pose, name):
        super().__init__(initial_pose, std_dev)
        self.name = name
        
    @property
    def dt(self):
        return self.dt

    @dt.setter
    def dt(self, timestep):
                
        self.state_transition_matrix =  np.array([[1,    timestep,       0,     0,      0,          0,              0,      0,          0],
                                                  [0,    1,              0,     0,      0,          0,              0,      0,          0],
                                                  [0,    0,              0,     0,      0,          0,              0,      0,          0],
                                                  [0,    0,              0,     1,      timestep,   0,              0,      0,          0],
                                                  [0,    0,              0,     0,      1,          0,              0,      0,          0],                                                 
                                                  [0,    0,              0,     0,      0,          0,              0,      0,          0],
                                                  [0,    0,              0,     0,      0,          0,              1,      timestep,   0],
                                                  [0,    0,              0,     0,      0,          0,              0,      1,          0],
                                                  [0,    0,              0,     0,      0,          0,              0,      0,          0]])

        G = np.array([[0.5*np.square(timestep)  ],
                      [timestep                 ],
                      [1                        ],
                      [0.5*np.square(timestep)  ],
                      [timestep                 ],
                      [1                        ],
                      [0.5*np.square(timestep)  ],
                      [timestep                 ],
                      [1                        ]])
        self.process_noise_matrix = np.matmul(G, np.transpose(G)) * np.square(self.std_dev)
    
    
   