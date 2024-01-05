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
        self.microgain = np.identity(initial_pose.shape[0]); np.fill_diagonal(self.microgain, 0.1) #dont put this to zero, otherwise the filter will diverge for some reason :(

        self.I = np.identity(initial_pose.shape[0])
        self.H = H
        #self.R = np.zeros(self.H.shape, int); np.fill_diagonal(self.R, 5) #this is for the case that our sensor measures everything that the state represents -> (15x15)
        self.R = np.zeros((3,3), float); np.fill_diagonal(self.R, measurement_noise_r)
    
        # IMM variables
        self.model_probability = None
        self.mixed_state = None
        self.psi = 0
        self.likelihood = None

        #Distributed stuff
        self.received_a = []
        self.received_F = []

        self.avg_a = None
        self.avg_F = None


    def predict(self, dt):
        print("Kalman predict " + self.name)
        self.dt = dt
        self.predicted_state = self.state_transition_matrix @ self.mixed_state
        self.predicted_covariance = (self.state_transition_matrix @ self.microgain @ np.transpose(self.state_transition_matrix)) + self.process_noise_matrix
        print(self.name, ":")
        #ic(self.predicted_covariance)
        ic(self.predicted_state)

        if has_negative_diagonal(self.predicted_covariance):
            print("predicted covariance has negative diagonal")
            exit()

    

    def update(self, z, distributed, a = None, F = None):
        '''
        Large Kalman gains correspond to low measurement noise, and so when measurement noise is low,
        we can subtract off a lot of our current uncertainty. When measurement noise is high, the gain will be small,
        so we won't subtract off very much.
        '''
        print("Kalman update " + self.name)
        if distributed == True:
            
            #TODO: where is this from? https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4434303 /https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=a6c9a3a6cd43b0447f9bd92547cc13aeacb346dc
            self.microgain = np.linalg.inv(np.linalg.inv(self.predicted_covariance) + F)
            self.updated_state = self.predicted_state + self.microgain @ (a - F @ self.predicted_state)
            ic(self.microgain)
            ic(np.linalg.inv(self.microgain))
            ic(self.updated_state)
            
            if has_negative_diagonal(self.microgain):
                print(self.name + ": updated_covariance has negative diagonal.")
                exit()
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
