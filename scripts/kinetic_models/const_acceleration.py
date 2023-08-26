from icecream import ic
import numpy as np
from classes.kalman import KalmanFilter

class CA_CYPR_Model(KalmanFilter):
    '''
    constant acceleration,
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
                
        self.state_transition_matrix =  np.array([[1,    timestep,       (timestep**2)/2,    0,      0,          0,              0,      0,          0             ],
                                                  [0,    1,              timestep,           0,      0,          0,              0,      0,          0             ],
                                                  [0,    0,              1,                  0,      0,          0,              0,      0,          0             ],
                                                  [0,    0,              0,                  1,      timestep,   (timestep**2)/2,0,      0,          0             ],
                                                  [0,    0,              0,                  0,      1,          timestep,       0,      0,          0             ],                                                 
                                                  [0,    0,              0,                  0,      0,          1,              0,      0,          0             ],
                                                  [0,    0,              0,                  0,      0,          0,              1,      timestep,   (timestep**2)/2],
                                                  [0,    0,              0,                  0,      0,          0,              0,      1,          timestep      ],
                                                  [0,    0,              0,                  0,      0,          0,              0,      0,          1             ]])

        G = np.array([  [(1/6)*np.power(timestep, 3)],
                        [0.5*np.square(timestep)    ],
                        [timestep                   ],
                        [(1/6)*np.power(timestep, 3)],
                        [0.5*np.square(timestep)    ],
                        [timestep                   ],
                        [(1/6)*np.power(timestep, 3)],
                        [0.5*np.square(timestep)    ],
                        [timestep                   ]])
        self.process_noise_matrix = np.matmul(G, np.transpose(G)) * np.square(self.std_dev)
    
   