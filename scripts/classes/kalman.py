from icecream import icecream
import numpy as np
from icecream import ic


class KalmanFilter():
    '''
    Best explanation: https://www.cbcity.de/das-kalman-filter-einfach-erklaert-teil-2
    '''

    def __init__(self, initial_pose, std_dev_process_noise) -> None:
        self.predicted_state = None
        self.predicted_covariance = None
        self.updated_state = None
        self.updated_covariance = None
        self.std_dev_process_noise = 0
        self.dt = 0
        
        self.state_transition_matrix = None
        self.std_dev_process_noise = std_dev_process_noise
        self.updated_state = np.array([[initial_pose[0]],
                                      [0],
                                      [0],
                                      [initial_pose[1]],
                                      [0],
                                      [0],
                                      [initial_pose[2]],
                                      [0],
                                      [0]])
        print("Initial position: ", self.updated_state)
        self.updated_covariance = np.identity(self.updated_state.shape[0]) #dont put this to zero, otherwise the filter will diverge for some reason :(
        
        self.I = np.identity(self.updated_state.shape[0])
        self.H = np.array([ [1, 0,  0,  0,  0,  0,  0,  0,  0],
                            [0, 1,  0,  0,  0,  0,  0,  0,  0],
                            [0, 0,  0,  1,  0,  0,  0,  0,  0],
                            [0, 0,  0,  0,  1,  0,  0,  0,  0],
                            [0, 0,  0,  0,  0,  0,  1,  0,  0],
                            [0, 0,  0,  0,  0,  0,  0,  1,  0]])
        #self.R = np.zeros(self.H.shape, int); np.fill_diagonal(self.R, 5) #this is for the case that our sensor measures everything that the state represents -> (15x15)
        self.R = np.zeros((6,6), int); np.fill_diagonal(self.R, 5000)
    
        # IMM variables
        self.model_probability = None
        self.mixed_state = None
        self.mixed_covariance = None
        self.psi = 0
        self.likelihood = None

    def predict(self, dt):
        self.dt = dt
        self.predicted_state = self.state_transition_matrix @ self.mixed_state
        self.predicted_covariance = (self.state_transition_matrix @ self.mixed_covariance @ np.transpose(self.state_transition_matrix)) + self.process_noise_matrix
        if has_negative_diagonal(self.predicted_covariance):
            print("predicted covariance has negative diagonal")
            exit()

    def update(self, z, distributed, a = None, F = None):
        if distributed == True:
            self.updated_covariance = np.linalg.inv(np.linalg.inv(self.predicted_covariance) + F)
            self.updated_state = self.updated_state + self.updated_covariance @ (a - F @ self.predicted_state)
            #self.updated_state = self.updated_covariance @ (self.predicted_covariance @ self.predicted_state + a)
        elif distributed == False:
            K = self.predicted_covariance @ np.transpose(self.H) @ np.linalg.inv(self.H @ self.predicted_covariance @ np.transpose(self.H) + self.R)
            self.updated_state = self.predicted_state + K @ (z - self.H @ self.predicted_state)
            self.updated_covariance = (self.I - K @ self.H) @ self.predicted_covariance

def has_negative_diagonal(matrix):

        # Iterate through the diagonal elements
        for i in range(len(matrix)):
            if matrix[i][i] < 0:
                return True  # Found a negative value on the diagonal

        return False  # No negative values on the diagonal