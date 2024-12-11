from Robot import my_robot
from Astar_planner_with_smoothing import get_path
from Particle_filter import particles
import numpy as np
from math import *

class error_function:
  def __init__(self,grid, init, goal,steering_noise,measurement_noise,distance_noise):
    self.grid = grid
    self.init = init
    self.goal = goal
    self.initial_robot_postion_x = init[0]
    self.initial_robot_postion_y = init[1]
    self.initial_robot_orientation = init[2]
    self.speed = 0.1
    self.steering_noise = steering_noise
    self.measurement_noise = measurement_noise
    self.distance_noise = distance_noise

  def simulate(self,path,controller_params,timeout = 1000):
    
    robot = my_robot(self.goal)
    robot.set(self.initial_robot_postion_x,self.initial_robot_postion_y,self.initial_robot_orientation)
    robot.set_noise(self.steering_noise, self.distance_noise, self.measurement_noise)
    #print('Robot Created')
    filter_for_localisation = particles(robot.x, robot.y, robot.orientation,robot.goal,robot.steering_noise,robot.distance_noise, robot.measurement_noise)
    #print('Particle Filter Initialised')
    cte  = 0.0
    err  = 0.0
    N    = 0
    index = 0 # index into the path
    while not robot.check_goal() and N < timeout and index < len(path)-1:
        diff_cte = - cte
        
        # ----------------------------------------
        # compute the CTE

        # start with the present robot estimate
        estimate = filter_for_localisation.get_position()

        ### ENTER CODE HERE
        dx = path[index+1][0] - path[index][0]
        dy = path[index+1][1] - path[index][1]
        Rx = estimate[0] - path[index][0]
        Ry = estimate[1] - path[index][1]
        cte = (Ry*dx - Rx*dy)/(sqrt(dx**2 + dy**2))
        # ----------------------------------------
        u = (Rx*dx + Ry*dy)/(sqrt(dx**2 + dy**2)) 
        if u > 1:
            index+=1
            
        diff_cte += cte

        steer = - controller_params[0] * cte - controller_params[1] * diff_cte 

        robot = robot.move(self.grid, steer, self.speed)

        robot.check_collision(self.grid)
        filter_for_localisation.move(self.grid, steer, self.speed)

        Z = robot.sense()
        filter_for_localisation.sense(Z)

        err += (cte ** 2)
        N += 1

    #if robot.check_goal():
        #print("Robot reached goal in {} steps with {} collisions".format(robot.num_steps,robot.num_collisions))
    #else:
        #print("Robot didn't reached goal in {} steps with {} collisions".format(robot.num_steps,robot.num_collisions))
    return robot.check_goal(),robot.num_collisions,robot.num_steps

  
  def get_error(self,parameters):
    weight_data       = parameters[0]
    weight_smooth     = parameters[1]
    p_gain            = parameters[2]
    d_gain            = parameters[3]
    path = get_path(self.grid ,self.init,self.goal,weight_data,weight_smooth)
    avg_err = 0
    for i in range(10):
      check_goal,num_collisions,num_steps = self.simulate(path,[p_gain,d_gain])
      avg_err+=((1-check_goal)*(1000)+num_collisions*100+num_steps)
    return int(avg_err/10) 