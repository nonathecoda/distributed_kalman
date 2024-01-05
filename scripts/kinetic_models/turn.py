from icecream import ic
import numpy as np
from classes.kalman import KalmanFilter

class CP_CYPR_RATE_Model(KalmanFilter):
    '''
    the way i name those classes is really good, i agree.
    constant pose,
    constant rate of yaw pitch roll.
    '''

    def __init__(self, std_dev_process_noise_q, measurement_noise_r, initial_pose, name = "ct"):
        self.name = name
        #initial_pose, H, measurement_noise_r, std_dev_process_noise_q
        H = np.array([  [1, 0,  0,  0,  0,  0,  0,  0,  0],
                        [0, 0,  0,  1,  0,  0,  0,  0,  0]
                        [0, 0,  0,  0,  0,  0,  1,  0,  0]])
        super().__init__(initial_pose, H, measurement_noise_r, std_dev_process_noise_q)
        self.dims = H.shape[1]
        print(self.name)
        
        
    @property
    def dt(self):
        return self.dt

    @dt.setter
    def dt(self, dt):
        w = 1
        self.state_transition_matrix =  np.array([[1,    w**(-1)*np.sin(w*dt),          w**(-2)*(1-np.cos(w*dt)),       0,                  0,                      0,                          0,      0,                      0                       ],
                                                  [0,    np.cos(w*dt),                  w**(-1)*np.sin(w*dt),           0,                  0,                      0,                          0,      0,                      0                       ],
                                                  [0,    -w*np.sin(w*dt),               np.cos(w*dt),                   0,                  0,                      0,                          0,      0,                      0                       ],
                                                  [0,    0,                             0,                              1,                  w**(-1)*np.sin(w*dt),   w**(-2)*(1-np.cos(w*dt)),   0,      0,                      0                       ],
                                                  [0,    0,                             0,                              0,                  np.cos(w*dt),           w**(-1)*np.sin(w*dt),       0,      0,                      0                       ],                                                 
                                                  [0,    0,                             0,                              0,                  -w*np.sin(w*dt),        np.cos(w*dt),               0,      0,                      0                       ], 
                                                  [0,    0,                             0,                              0,                  0,                      0,                          1,      w**(-1)*np.sin(w*dt),   w**(-2)*(1-np.cos(w*dt))],
                                                  [0,    0,                             0,                              0,                  0,                      0,                          0,      np.cos(w*dt),           w**(-1)*np.sin(w*dt)    ],
                                                  [0,    0,                             0,                              0,                  0,                      0,                          0,      -w*np.sin(w*dt),        np.cos(w*dt)            ]])

        G = np.array([  [(1/6)*np.power(dt, 3)],
                        [0.5*np.square(dt)    ],
                        [dt                   ],
                        [(1/6)*np.power(dt, 3)],
                        [0.5*np.square(dt)    ],
                        [dt                   ],
                        [(1/6)*np.power(dt, 3)],
                        [0.5*np.square(dt)    ],
                        [dt                   ]])
        
        self.process_noise_matrix = np.matmul(G, np.transpose(G)) * np.square(self.std_dev_process_noise)
        #self.process_noise_matrix = np.eye(self.state_transition_matrix.shape[0])
    
        