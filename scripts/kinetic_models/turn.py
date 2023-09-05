from icecream import ic
import numpy as np
from classes.kalman import KalmanFilter

class CP_CYPR_RATE_Model(KalmanFilter):
    '''
    the way i name those classes is really good, i agree.
    constant pose,
    constant rate of yaw pitch roll.
    '''

    def __init__(self,initial_pose, std_dev_process_noise, name):
        super().__init__(initial_pose, std_dev_process_noise)
        self.name = name
        
    @property
    def dt(self):
        return self.dt

    @dt.setter
    def dt(self, timestep):
                
        self.state_transition_matrix =  np.array([[1,    0,              0,     0,      0,          0,      0,      0,          0,              0,      0,         0,           0,          0,      0       ],
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              0,      0,         0,           0,          0,      0       ],
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              0,      0,         0,           0,          0,      0       ],
                                                  [0,    0,              0,     1,      0,          0,      0,      0,          0,              0,      0,         0,           0,          0,      0       ],
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              0,      0,         0,           0,          0,      0       ],                                                 
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              0,      0,         0,           0,          0,      0       ],
                                                  [0,    0,              0,     0,      0,          0,      1,      0,          0,              0,      0,         0,           0,          0,      0       ],
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              0,      0,         0,           0,          0,      0       ],
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              0,      0,         0,           0,          0,      0       ],
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              1,      timestep,  0,           0,          0,      0       ],
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              0,      1,         0,           0,          0,      0       ],
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              0,      0,         1,           timestep,   0,      0       ],
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              0,      0,         0,           1,          0,      0       ],
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              0,      0,         0,           0,          1,      timestep],
                                                  [0,    0,              0,     0,      0,          0,      0,      0,          0,              0,      0,         1,           0,          0,      1       ]])

        G = np.array([[0.5*np.square(timestep)  ],
                      [0                        ],
                      [0                        ],
                      [0.5*np.square(timestep)  ],
                      [0                        ],
                      [0                        ],
                      [0.5*np.square(timestep)  ],
                      [0                        ],
                      [0                        ],
                      [0.5*np.square(timestep)  ],
                      [timestep                 ],
                      [0.5*np.square(timestep)  ],
                      [timestep                 ],
                      [0.5*np.square(timestep)  ],
                      [timestep                 ]])
        self.process_noise_matrix = np.matmul(G, np.transpose(G)) * np.square(self.std_dev_process_noise)