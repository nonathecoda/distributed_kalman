from icecream import ic
import numpy as np
import math

from kinetic_models.const_acceleration import CA_CYPR_Model
from kinetic_models.const_velocity import CV_CYPR_Model
from kinetic_models.turn import CP_CYPR_RATE_Model
from classes.kalman import KalmanFilter

class InteractingMultipleModel:
    '''Implementation following Anthony F. Genovese: "The Interacting Multiple Model Algorithm for
        Accurate State Estimation of Maneuvring Targets." '''
    
    

    def __init__(self, camera, initial_target_state):
        self.camera = camera

        initial_pose_9d = np.reshape(initial_target_state, (9,1))
        initial_pose_6d = np.delete(initial_pose_9d, [2,5,8], 0)
        
        #self.kalman = KalmanFilter(std_dev_process_noise=0.1, measurement_noise_r =  5, initial_pose=initial_pose)
        self.const_vel_model = CV_CYPR_Model(std_dev_process_noise_q=1, measurement_noise_r = 10, initial_pose=initial_pose_6d, name = "constant velocity")
        self.const_accel_model = CA_CYPR_Model(std_dev_process_noise_q=1, measurement_noise_r = 2000, initial_pose=initial_pose_9d, name = "constant acceleration")
        self.turn_model = CP_CYPR_RATE_Model(std_dev_process_noise_q=10000, measurement_noise_r = 1, initial_pose=initial_pose_9d, name = "turn")
        self.models = [self.const_vel_model, self.const_accel_model]
        self.models = [self.const_accel_model]

        cv2ca = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]])
 
        self.model_transition_matrix = [[np.eye(6), cv2ca.T],
                                        [cv2ca, np.eye(9)]]


        self.state_switching_matrix= np.array([[0.8, 0.2],
                                               [0.25, 0.75]])
        
        self.state_switching_matrix= np.array([[1]])
        self.model_transition_matrix = [[np.eye(9)]]

        for count, probs in enumerate(self.state_switching_matrix[0]):
            self.models[count].model_probability = probs

        #self.models[0].model_probability = 0.1
        #self.models[1].model_probability = 0.9

        self.covariance = None
        self.combined_state = None
    
    def update_pose(self, timestep, distributed, a = None, F = None):
        print(self.camera.name + ": IMM START")
        measured_pose = self.camera.get_measurements()

        # Calculate Psi for each model
        for j, model_j in enumerate(self.models):
            model_j.psi = 0
            for i, model_i in enumerate(self.models):
                model_j.psi = model_j.psi + (self.state_switching_matrix[i][j]*model_i.model_probability)
            
        # Calculate mixed states for each model
        for j, model_j in enumerate(self.models):
            model_j.mixed_state = np.zeros((model_j.dims,1))
            for i, model_i in enumerate(self.models):
                mu_ij = (1/model_j.psi)*self.state_switching_matrix[i][j]*model_i.model_probability
                model_j.mixed_state = model_j.mixed_state + (self.model_transition_matrix[j][i] @ (model_i.updated_state * mu_ij))
                
        # Calculate mixed covariances for each model
        for j, model_j in enumerate(self.models):
            model_j.mixed_covariance = np.zeros((model_j.dims,model_j.dims))
            for i, model_i in enumerate(self.models):
                mu_ij = (1/model_j.psi)*self.state_switching_matrix[i][j]*model_i.model_probability                
                dummy_P = model_i.updated_covariance + ((model_i.updated_state - model_i.mixed_state)@ np.transpose(model_i.updated_state - model_i.mixed_state))
                model_j.mixed_covariance = model_j.mixed_covariance + (mu_ij* (self.model_transition_matrix[j][i] @ dummy_P @ self.model_transition_matrix[j][i].T))

        # Execute Kalman Filter for each model
        for model in self.models:
            if model.dims == 6:
                a_dummy = np.delete(a, [2,5,8], 0)
                F_dummy = np.delete(F, [2,5,8], 0)
                F_dummy = np.delete(F_dummy, [2,5,8], 1)                
            else:
                a_dummy = a
                F_dummy = F
            model.predict(timestep)
            model.update(measured_pose, distributed, a_dummy, F_dummy)

        # Update likelihood for each model
        for j, model_j in enumerate(self.models):
            Z_j = measured_pose - model_j.H @ model_j.predicted_state
            S_j = model_j.H @ model_j.predicted_covariance @ np.transpose(model_j.H) + model_j.R
            try:
                model_j.likelihood = (1/np.sqrt(np.linalg.det(2*math.pi*S_j))) * np.exp(-0.5 * ((np.transpose(Z_j) @ np.linalg.inv(S_j) @ Z_j)[0][0]))
            except FloatingPointError: # underflow error? It's because the predicted covariance is too big -> model apparently has a very low likelihood
                model_j.likelihood = 1.0e-100

        # Update probability for each model
        for j, model_j in enumerate(self.models):
            c = 0
            for i, model_i in enumerate(self.models):
                c = c + (model_i.likelihood*model_i.psi)
            model_j.model_probability = (1/c)*model_j.likelihood*model_j.psi
        
        # for using only constant acceleration, replace combined state formula
        cv2ca = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]])
        '''
        self.combined_state = self.combined_state + (cv2ca.T@(model_j.updated_state * model_j.model_probability))
        '''
        # Combine state estimates to combined state
        self.combined_state = np.zeros((6,1))
        for j, model_j in enumerate(self.models):
            #self.combined_state = self.combined_state + (self.model_transition_matrix[0][j]@(model_j.updated_state * model_j.model_probability))
            self.combined_state = self.combined_state + (cv2ca.T@(model_j.updated_state * model_j.model_probability))
        # Combined covariance TODO: needed?
        #self.covariance = np.zeros((9,9))
        #for j, model_j in enumerate(self.models):
        #    self.covariance = self.covariance + (model_j.model_probability * (model_j.updated_covariance + ((model_j.updated_state - self.combined_state) @ np.transpose(model_j.updated_state - self.combined_state))))

        #if has_negative_diagonal(self.covariance):
        #    print("combined covariance has negative diagonal")
        #    exit()
        return self.combined_state.flatten(), self.models
    
def has_negative_diagonal(matrix):

    # Iterate through the diagonal elements
    for i in range(len(matrix)):
        if matrix[i][i] < 0:
            return True  # Found a negative value on the diagonal

    return False  # No negative values on the diagonal