from icecream import ic
import numpy as np

def get_figure_eight(dt):
    # initial_pos: initial position of the object (3d vector)
    # initial_vel: initial velocity of the object (3d vector)
    # initial_accel: initial acceleration of the object (3d vector)
    # dt: time step
    # return: coordinates, velocities, accelerations, timestamps
    A = 150
    coordinates = np.zeros((3, 3000), dtype = float)
    velocities = np.zeros((3, 3000), dtype = float)
    accelerations = np.zeros((3, 3000), dtype = float)
    timestamps = np.zeros((3000), dtype = float)

    for t in range(3000):
        coordinates[0, t] = A + (A*np.cos(t*dt) / (1 + np.sin(t*dt)**2))
        coordinates[1, t] = A + (A*np.cos(t*dt)*np.sin(t*dt) / (1 + np.sin(t*dt)**2))
        coordinates[2, t] = 50

        velocities[0, t] = -A*np.sin(t*dt) / (1 + np.sin(t*dt)**2) + (A*np.cos(t*dt)**2*np.sin(t*dt)) / (1 + np.sin(t*dt)**2)**2
        velocities[1, t] = -A*np.sin(t*dt)*np.sin(t*dt) / (1 + np.sin(t*dt)**2) + (A*np.cos(t*dt)*np.sin(t*dt)**2) / (1 + np.sin(t*dt)**2)**2
        velocities[2, t] = 0

        accelerations[0, t] = -A*np.cos(t*dt) / (1 + np.sin(t*dt)**2) - (2*A*np.cos(t*dt)**2*np.sin(t*dt)) / (1 + np.sin(t*dt)**2)**2 + (A*np.cos(t*dt)**4*np.sin(t*dt)) / (1 + np.sin(t*dt)**2)**3
        accelerations[1, t] = -A*np.cos(t*dt)*np.sin(t*dt) / (1 + np.sin(t*dt)**2) + (2*A*np.cos(t*dt)*np.sin(t*dt)**2) / (1 + np.sin(t*dt)**2)**2 - (A*np.cos(t*dt)**3*np.sin(t*dt)**2) / (1 + np.sin(t*dt)**2)**3
        accelerations[2, t] = 0
        
        timestamps[t] = timestamps[t-1] + dt
    return coordinates, velocities, accelerations, timestamps

def get_constant_acceleration(initial_pos, initial_vel, initial_accel, dt):
    # initial_pos: initial position of the object (3d vector)
    # initial_vel: initial velocity of the object (3d vector)
    # initial_accel: initial acceleration of the object (3d vector)
    # dt: time step
    # return: coordinates, velocities, accelerations, timestamps

    coordinates = np.zeros((3, 3000), dtype = float)
    velocities = np.zeros((3, 3000), dtype = float)
    accelerations = np.zeros((3, 3000), dtype = float)
    timestamps = np.zeros((3000), dtype = float)

    for i in range(3000):
        if i == 0:
            coordinates[:, i] = initial_pos
            velocities[:, i] = initial_vel
            accelerations[:, i] = initial_accel
            timestamps[i] = 0
        else:
            coordinates[:, i] = coordinates[:, i-1] + velocities[:, i-1] * dt + 0.5 * accelerations[:, i-1] * dt**2
            velocities[:, i] = velocities[:, i-1] + accelerations[:, i-1] * dt
            accelerations[:, i] = initial_accel
            timestamps[i] = timestamps[i-1] + dt

    return coordinates, velocities, accelerations, timestamps

def get_constant_turn(turn_rate, initial_pos, initial_vel, initial_accel, dt):

    coordinates = np.zeros((3, 3000), dtype = float)
    velocities = np.zeros((3, 3000), dtype = float)
    accelerations = np.zeros((3, 3000), dtype = float)
    timestamps = np.zeros((3000), dtype = float)

    for i in range(3000):
        if i == 0:
            coordinates[:, i] = initial_pos
            velocities[:, i] = initial_vel
            accelerations[:, i] = initial_accel
            timestamps[i] = 0
        else:
            coordinates[:, i] = coordinates[:, i-1] + velocities[:, i-1] * (turn_rate**(-1)*np.sin(turn_rate*dt)) + accelerations[:, i-1] * (turn_rate**(-2)*(1-np.cos(turn_rate*dt)))
            velocities[:, i] = velocities[:, i-1]*np.cos(turn_rate*dt) + accelerations[:, i-1] * (turn_rate**(-1)*np.sin(turn_rate*dt))
            accelerations[:, i] = velocities[:, i-1]*(-turn_rate*np.sin(turn_rate*dt)) + accelerations[:, i-1]*np.cos(turn_rate*dt)
            timestamps[i] = timestamps[i-1] + dt

    return coordinates, velocities, accelerations, timestamps

def get_cv_ct_ca(turn_rate, initial_pos, initial_vel, initial_accel, dt):
    coordinates = np.zeros((3, 3000), dtype = float)
    velocities = np.zeros((3, 3000), dtype = float)
    accelerations = np.zeros((3, 3000), dtype = float)
    timestamps = np.zeros((3000), dtype = float)

    for i in range(3000):
        if i == 0:
            coordinates[:, i] = initial_pos
            velocities[:, i] = initial_vel
            accelerations[:, i] = initial_accel
            timestamps[i] = 0
        #cv
        elif i < 200:
            coordinates[:, i] = coordinates[:, i-1] + velocities[:, i-1] * dt + 0.5
            velocities[:, i] = velocities[:, i-1]
            accelerations[:, i] = 0
            timestamps[i] = timestamps[i-1] + dt 
        # ct
        elif i >=200 and i < 400:
            coordinates[:, i] = coordinates[:, i-1] + velocities[:, i-1] * (turn_rate**(-1)*np.sin(turn_rate*dt)) + accelerations[:, i-1] * (turn_rate**(-2)*(1-np.cos(turn_rate*dt)))
            velocities[:, i] = velocities[:, i-1]*np.cos(turn_rate*dt) + accelerations[:, i-1] * (turn_rate**(-1)*np.sin(turn_rate*dt))
            accelerations[:, i] = velocities[:, i-1]*(-turn_rate*np.sin(turn_rate*dt)) + accelerations[:, i-1]*np.cos(turn_rate*dt)
            timestamps[i] = timestamps[i-1] + dt
        # ca
        else:
            coordinates[:, i] = coordinates[:, i-1] + velocities[:, i-1] * dt + 0.5 * accelerations[:, i-1] * dt**2
            velocities[:, i] = velocities[:, i-1] + accelerations[:, i-1] * dt
            accelerations[:, i] = initial_accel
            timestamps[i] = timestamps[i-1] + dt 

    return coordinates, velocities, accelerations, timestamps