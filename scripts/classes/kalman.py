from icecream import icecream
import numpy as np
from icecream import ic


class KalmanFilter():
    '''
    Best explanation: https://www.cbcity.de/das-kalman-filter-einfach-erklaert-teil-2
    '''

    def __init__(self, initial_pose, H, measurement_noise_r, std_dev_process_noise_q) -> None:
        self.predicted_state = None
        self.predicted_covariance = None
        self.updated_state = None
        self.updated_covariance = None
        self.std_dev_process_noise = 0
        self.dt = 0
        
        self.state_transition_matrix = None
        self.std_dev_process_noise = std_dev_process_noise_q
        self.updated_state = initial_pose
        self.updated_covariance = np.identity(initial_pose.shape[0]); np.fill_diagonal(self.updated_covariance, 0.1) #dont put this to zero, otherwise the filter will diverge for some reason :(

        self.I = np.identity(initial_pose.shape[0])
        self.H = H
        #self.R = np.zeros(self.H.shape, int); np.fill_diagonal(self.R, 5) #this is for the case that our sensor measures everything that the state represents -> (15x15)
        self.R = np.zeros((3,3), float); np.fill_diagonal(self.R, measurement_noise_r)
    
        # IMM variables
        self.model_probability = None
        self.mixed_state = None
        self.mixed_covariance = None
        self.psi = 0
        self.likelihood = None

    def predict(self, dt):
        print("Kalman predict " + self.name)
        self.dt = dt
        self.predicted_state = self.state_transition_matrix @ self.mixed_state
        self.predicted_covariance = (self.state_transition_matrix @ self.mixed_covariance @ np.transpose(self.state_transition_matrix)) + self.process_noise_matrix
        
        print(self.name, ":")
        ic(self.predicted_covariance)
        ic(self.predicted_state)

        if has_negative_diagonal(self.predicted_covariance):
            print("predicted covariance has negative diagonal")
            exit()

    def update(self, z, distributed, a = None, F = None):
        
        print("Kalman update " + self.name)
        if distributed == True:
            self.updated_covariance = np.linalg.inv(np.linalg.pinv(self.predicted_covariance) + F)
            if has_negative_diagonal(self.updated_covariance):
                print(self.name + ": updated_covariance has negative diagonal.")
                ic(self.updated_covariance)
                exit()
            self.updated_state = self.predicted_state + np.linalg.inv(self.updated_covariance) @ (a - F @ self.predicted_state)
            ic(self.updated_state)
            #exit()
        else:
            
            K = (self.predicted_covariance @ np.transpose(self.H)) @ np.linalg.inv((self.H @ self.predicted_covariance @ np.transpose(self.H)) + self.R)
            self.updated_state = self.predicted_state + K @ (z - self.H @ self.predicted_state)
            self.updated_covariance = (self.I - K @ self.H) @ self.predicted_covariance @ np.transpose(self.I - K @ self.H) + K @ self.R @ np.transpose(K) # Joseph form
            

def has_negative_diagonal(matrix):
        # Iterate through the diagonal elements
        for i in range(len(matrix)):
            if matrix[i][i] < 0:
                return True  # Found a negative value on the diagonal
        return False  # No negative values on the diagonal
