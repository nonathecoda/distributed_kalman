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
    
    def __init__(self, camera, initial_pose):
        self.camera = camera
        self.initial_pose = initial_pose
        
        self.kalman = KalmanFilter(std_dev_process_noise=0.1, initial_pose=self.initial_pose)
        self.const_vel_model = CV_CYPR_Model(std_dev_process_noise=0.1, initial_pose=self.initial_pose, name = "constant velcoity")
        self.const_accel_model = CA_CYPR_Model(std_dev_process_noise=0.1, initial_pose=self.initial_pose, name = "constant acceleration")
        #self.turn_model = CP_CYPR_RATE_Model(std_dev_process_noise=0.01, initial_pose=self.initial_pose, name = "turn")
        #self.models = [self.const_vel_model, self.const_accel_model, self.turn_model]
        self.models = [self.const_accel_model]
        
        #self.state_switching_matrix = np.array([[0.55, 0.15, 0.3],
        #                                       [0.3, 0.60, 0.1],
        #                                       [0.35, 0.05, 0.6]])
        


        self.state_switching_matrix= np.array([[1.0]])
        for count, probs in enumerate(self.state_switching_matrix[0]):
            self.models[count].model_probability = probs # TODO: came up with this myself, how do you actually initialise the model probabilities?

        self.covariance = None
        self.combined_state = None
    
    def update_pose(self, timestep, distributed = False, a = None, F = None):
        print("IMM update - ", self.camera.name)
        measured_pose = self.camera.get_measurements()
        # Calculate Psi for each model
        for j, model_j in enumerate(self.models):
            model_j.psi = 0
            for i, model_i in enumerate(self.models):
                model_j.psi = model_j.psi + (self.state_switching_matrix[i][j]*model_i.model_probability)
            
        # Calculate mixed states for each model
        for j, model_j in enumerate(self.models):
            model_j.mixed_state = np.zeros((9,1))
            for i, model_i in enumerate(self.models):
                mu_ij = (1/model_j.psi)*self.state_switching_matrix[i][j]*model_i.model_probability
                model_j.mixed_state = model_j.mixed_state + (model_i.updated_state * mu_ij)

        # Calculate mixed covariances for each model
        for j, model_j in enumerate(self.models):
            model_j.mixed_covariance = np.zeros((9,9))
            for i, model_i in enumerate(self.models):
                mu_ij = (1/model_j.psi)*self.state_switching_matrix[i][j]*model_i.model_probability
                model_j.mixed_covariance = model_j.mixed_covariance + (mu_ij * (model_i.updated_covariance + ((model_i.updated_state - model_j.mixed_state) @ np.transpose(model_i.updated_state - model_j.mixed_state))))
                ic(model_i.updated_covariance)
                ic(model_i.updated_state)
                ic(model_j.mixed_state)
                ic(mu_ij)
                ic(model_j.psi)
                ic(self.state_switching_matrix[i][j])
                ic(model_i.model_probability)

        # Execute Kalman Filter for each model
        for model in self.models:
            model.predict(timestep)
            model.update(measured_pose, distributed, a, F)

        # Update likelihood for each model
        for j, model_j in enumerate(self.models):
            Z_j = measured_pose - model_j.H @ model_j.predicted_state # TODO: check if H@x is correct or if it only needs x
            S_j = model_j.H @ model_j.predicted_covariance @ np.transpose(model_j.H) + model_j.R
            model_j.likelihood = (1/np.sqrt(np.linalg.det(2*math.pi*S_j))) * math.exp(-0.5 * np.transpose(Z_j) @ np.linalg.inv(S_j) @ Z_j)
            ic(measured_pose)
            ic(model_j.predicted_state)
            ic(Z_j)
            ic(S_j)

        # Update probability for each model
        for j, model_j in enumerate(self.models):
            c = 0
            for i, model_i in enumerate(self.models):
                c = c + (model_i.likelihood*model_i.psi)
            model_j.model_probability = (1/c)*model_j.likelihood*model_j.psi

        # Combine state estimates to combined state
        self.combined_state = np.zeros((9,1))
        for j, model_j in enumerate(self.models):
            self.combined_state = self.combined_state + (model_j.updated_state * model_j.model_probability)

        # Combined covariance
        self.covariance = np.zeros((9,9))
        for j, model_j in enumerate(self.models):
            self.covariance = self.covariance + (model_j.model_probability * (model_j.updated_covariance + (self.combined_state - model_j.updated_state) @ np.transpose(self.combined_state - model_j.updated_state)))
        ic(self.combined_state)
        ic(self.covariance)

        return self.combined_state.flatten()