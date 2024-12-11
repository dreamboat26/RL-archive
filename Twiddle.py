import random
import numpy as np
import matplotlib.pyplot as plt
from math import *

def twiddle_param_tuner(inital_params,change_rate_params,error_function,grid, init, goal,steering_noise,measurement_noise,distance_noise,tolerance=0.001): 
    error_fun = error_function(grid, init, goal,steering_noise,measurement_noise,distance_noise) 
    best_err = error_fun.get_error(inital_params)
    n=0
    while sum(change_rate_params) > tolerance:
        for i in range(len(inital_params)):
            inital_params[i]+=change_rate_params[i]
            error_fun = error_function(grid, init, goal,steering_noise,measurement_noise,distance_noise) 
            error = error_fun.get_error(inital_params)
            if error < best_err:
                best_err = error
                change_rate_params[i]*=1.1
            else:
                inital_params[i]-=(2.0*change_rate_params[i])
                error_fun = error_function(grid, init, goal,steering_noise,measurement_noise,distance_noise) 
                error = error_fun.get_error(inital_params)
                if error < best_err:
                    best_err = error
                    change_rate_params[i]*=1.1
                else:
                    inital_params[i]+=change_rate_params[i]
                    change_rate_params[i]*=0.9
        n+=1
        print('Twiddle:',n,'Current Params:',inital_params,'Best Error:',best_err)
    return inital_params,best_err
