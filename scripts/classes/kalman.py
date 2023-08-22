from icecream import icecream
import numpy as np
from icecream import ic

class KalmanFilter():
    '''
    Best explanation: https://www.cbcity.de/das-kalman-filter-einfach-erklaert-teil-2
    '''

    def __init__(self, initial_pose, std_dev) -> None:
        self.predicted_state = None
        self.predicted_covariance = None
        self.updated_state = None
        self.updated_covariance = None
        self.std_dev = 0
        self.dt = 0
        self.I = None
        self.H = None
        self.R = None
        self.state_transition_matrix = None
        self.std_dev = std_dev
        self.updated_state = np.array([[initial_pose.position.x_coord],
                                      [0],
                                      [0],
                                      [initial_pose.position.y_coord],
                                      [0],
                                      [0],
                                      [initial_pose.position.z_coord],
                                      [0],
                                      [0],
                                      [initial_pose.orientation.yaw_pitch_roll[0]],
                                      [0],
                                      [initial_pose.orientation.yaw_pitch_roll[1]],
                                      [0],
                                      [initial_pose.orientation.yaw_pitch_roll[2]],
                                      [0]])
        print("Initial position: ", self.updated_state)
        self.updated_covariance = np.identity(self.updated_state.shape[0])
        
        self.I = np.identity(self.updated_state.shape[0])
        self.H = np.identity(self.updated_state.shape[0]) #this is for the case that our sensor measures everything that the state represents -> position, linear velocity, linear acceleration, orientation, orientation rate
        self.R = np.zeros(self.H.shape, int); np.fill_diagonal(self.R, 5) #this is for the case that our sensor measures everything that the state represents -> (15x15)
        
    
        # IMM variables
        self.model_probability = None
        self.mixed_state = None
        self.mixed_covariance = None
        self.psi = 0
        self.likelihood = None

    

    def predict(self, dt):
        self.dt = dt
        self.predicted_state = self.state_transition_matrix @ self.updated_state
        self.predicted_covariance = self.state_transition_matrix @ self.updated_covariance @ np.transpose(self.state_transition_matrix) + self.process_noise_matrix

    def update(self, z):

        K = self.predicted_covariance @ np.transpose(self.H) @ np.linalg.inv(self.H @ self.predicted_covariance @ np.transpose(self.H) + self.R)
        self.updated_state = self.predicted_state + K @ (z - self.H @ self.predicted_state)
        self.updated_covariance = (self.I - K @ self.H) @ self.predicted_covariance
