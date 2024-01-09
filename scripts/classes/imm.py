from icecream import ic
import numpy as np
import math

from kinetic_models.const_acceleration import CA_CYPR_Model
from kinetic_models.const_velocity import CV_CYPR_Model
from kinetic_models.const_turn import Const_Turn_Model
from kinetic_models.const_position import Const_Pos_Model
from classes.kalman import KalmanFilter

class InteractingMultipleModel:
    '''Implementation following Anthony F. Genovese: "The Interacting Multiple Model Algorithm for
        Accurate State Estimation of Maneuvring Targets." '''
    
    

    def __init__(self, camera, initial_target_state):
        self.camera = camera

        initial_pose_9d = np.reshape(initial_target_state, (9,1))
        initial_pose_6d = np.delete(initial_pose_9d, [2,5,8], 0)
        initial_pose_5d = np.delete(initial_pose_6d, [-1], 0)
        initial_pose_3d = np.delete(initial_pose_5d, [1, 3], 0)

        
        #self.kalman = KalmanFilter(std_dev_process_noise=0.1, measurement_noise_r =  5, initial_pose=initial_pose)
        self.const_vel_model = CV_CYPR_Model(std_dev_process_noise_q=10, measurement_noise_r = 18, initial_pose=initial_pose_6d, name = "constant velocity")
        self.const_accel_model = CA_CYPR_Model(std_dev_process_noise_q=10, measurement_noise_r = 20, initial_pose=initial_pose_9d, name = "constant acceleration")
        self.orbit_model = Const_Turn_Model(std_dev_process_noise_q=100000, measurement_noise_r =5, initial_pose=initial_pose_5d, name = "constant turn")
        self.const_pos_model = Const_Pos_Model(std_dev_process_noise_q=10, measurement_noise_r = 25, initial_pose=initial_pose_3d, name = "constant position")
        
        #state transition matrices
        self.cv2ca = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]])

        self.cv2corbit = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0]])

        cv2cv = np.eye(6)
        ca2orbit = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0]])
        
        orbit2cp = np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1]])

        ### 1 Model ###
        #self.models = [self.orbit_model]
        #self.state_switching_matrix= np.array([[1]])
        #self.model_transition_matrix = [[np.eye(5)]]
        
        ### 2 Models ###

        #self.models = [self.const_vel_model, self.orbit_model]
        #self.model_transition_matrix = [[np.eye(6), self.cv2ct.T],
        #                                [self.cv2ct, np.eye(5)]]
        #self.state_switching_matrix= np.array([[0.98, 0.02],
        #                                       [0.02, 0.98]])

        
        ### 3 Models ###
        
        self.models = [self.const_vel_model, self.const_accel_model, self.orbit_model, self.const_pos_model]
        

        self.model_transition_matrix = [[np.eye(6),                     self.cv2ca.T,       np.dot(self.cv2ca.T, ca2orbit.T), self.cv2ca.T@ca2orbit.T@orbit2cp.T],
                                        [self.cv2ca,                    np.eye(9),          ca2orbit.T, np.dot(ca2orbit.T, orbit2cp.T)],
                                        [np.dot(ca2orbit, self.cv2ca),  ca2orbit,           np.eye(5), orbit2cp.T],
                                        [orbit2cp@ca2orbit@self.cv2ca,  orbit2cp@ca2orbit,     orbit2cp, np.eye(3)]]


        self.state_switching_matrix= np.array([[0.97, 0.01, 0.01, 0.01],
                                               [0.01, 0.97, 0.01, 0.01],
                                               [0.01, 0.01, 0.97, 0.01],
                                               [0.01, 0.01, 0.01, 0.97]])
        
        
        for count, probs in enumerate(self.state_switching_matrix[3]):
            self.models[count].model_probability = probs


        self.covariance = None
        self.combined_state = None
    
    def update_pose(self, timestep, distributed):
        print(self.camera.name + ": IMM START")
        measured_pose = self.camera.get_measurements()

        # Calculate Psi for each model
        for j, model_j in enumerate(self.models):
            model_j.psi = 0
            for i, model_i in enumerate(self.models):
                model_j.psi = model_j.psi + (self.state_switching_matrix[i][j]*model_i.model_probability) #predicted model probability
            
        # Calculate mixed states for each model
        for j, model_j in enumerate(self.models):
            model_j.mixed_state = np.zeros((model_j.dims,1))
            for i, model_i in enumerate(self.models):
                mu_ij = (1/model_j.psi)*self.state_switching_matrix[i][j]*model_i.model_probability #mixing probability
                model_j.mixed_state = model_j.mixed_state + (self.model_transition_matrix[j][i] @ (model_i.updated_state * mu_ij))

        # Execute Kalman Filter for each model
        for model in self.models:
            model.predict(timestep)
            model.update(measured_pose, distributed, model.avg_a, model.avg_F)

        # Update likelihood for each model
        for j, model_j in enumerate(self.models):
            Z_j = measured_pose - model_j.H @ model_j.predicted_state # TODO: ist das ok so? oder sollte das irgendwie mit a und F gemacht werden?
            ic(model_j.H @ model_j.predicted_state)
            ic(model_j.H)
            ic(model_j.predicted_state)
            ic(measured_pose)

            S_j = model_j.H @ model_j.predicted_covariance @ np.transpose(model_j.H) + model_j.R
            try:
                model_j.likelihood = (1/np.sqrt(np.linalg.det(2*math.pi*S_j))) * np.exp(-0.5 * ((np.transpose(Z_j) @ np.linalg.inv(S_j) @ Z_j)[0][0]))
                ic(-0.5 * ((np.transpose(Z_j) @ np.linalg.inv(S_j) @ Z_j)[0][0]))
                ic(np.exp(-0.5 * ((np.transpose(Z_j) @ np.linalg.inv(S_j) @ Z_j)[0][0])))
                ic(Z_j)
                ic(S_j)
            except FloatingPointError: # underflow error? It's because the predicted covariance is too big -> model apparently has a very low likelihood
                print("FloatingPointError 2")
                model_j.likelihood = 1.0e-100

        # Update probability for each model
        for j, model_j in enumerate(self.models):
            c = 0
            for i, model_i in enumerate(self.models):
                try:
                    c = c + (model_i.likelihood*model_i.psi)
                except FloatingPointError: # underflow error? It's because the predicted covariance is too big -> model apparently has a very low likelihood
                    print("FloatingPointError 2")
                    c = c + 1.0e-100
            model_j.model_probability = (1/c)*model_j.likelihood*model_j.psi
        
        # for using only constant acceleration, replace combined state formula
        '''
        self.combined_state = self.combined_state + (self.cv2ca.T@(model_j.updated_state * model_j.model_probability))
        '''
        # Combine state estimates to combined state
        self.combined_state = np.zeros((6,1))
        for j, model_j in enumerate(self.models):
            self.combined_state = self.combined_state + (self.model_transition_matrix[0][j]@(model_j.updated_state * model_j.model_probability))
            #self.combined_state = self.combined_state + (self.cv2ca.T@(model_j.updated_state * model_j.model_probability))
            #self.combined_state = self.combined_state + (self.cv2corbit.T@(model_j.updated_state * model_j.model_probability))
       
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