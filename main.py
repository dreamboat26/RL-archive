import random
import numpy as np
import matplotlib.pyplot as plt
from Particle_filter import particles
from Twiddle import twiddle_param_tuner
from Error_Function import error_function

grid = np.array([[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0]])


init = [0,0,0.0]
goal = [grid.shape[0]-1, grid.shape[1]-1]
steering_noise    = 0.1
distance_noise    = 0.03
measurement_noise = 0.3
#best_params = [0.1,0.2,2.0,6.0]
inital_params = [0.2,0.2,1.0,1.0]
change_rate_params = [0.05,0.05,1.0,1.0]

params, err = twiddle_param_tuner(inital_params,change_rate_params,error_function,grid,init,goal,steering_noise,measurement_noise,distance_noise)
print("Final twiddle error = {}".format(err))
