# Robotic-Car-Path-Planning

This Repository Contains a project I created in which I  used all tools that I learned in Artificial Intelligence for Robotics Course  by Sebastian Thrun Udacity.

In this project I used an algorithm called Twiddle to optimise parameters for PD Controller and a Smoother to make a robotic car reach its goal on a smooth path without colliding.

1. First we takes inputs like grid(map of world) , initial position , goal , initial values of  parameters and initial change rate of parameters.
2. Then we initialises the robotic car with appropriate noise in its motion and measurement model. 
3. Then we use A* Algorithm along with a smoother and find the optimal path.  
4. Then we  initialise particle filter which we use for localisation of car. 
5. Then we run our robot and particle filter on the path provided by A* using PD Controller.  
6. We  calculate the cross track error whole time the robot is moving.
7. We then use this error to update the values of parameters of PD Controller and Smoother using twiddle.



This is final Path followed by the car:
![Alt Text](https://github.com/Rahul-Choudhary-3614/Robotic-Car-Path-Planning-/blob/main/images/Result.png?raw=true)


A* Algorithm:
![Alt Text](https://github.com/Rahul-Choudhary-3614/Robotic-Car-Path-Planning-/blob/main/images/astar_grid.gif?raw=true)

Smoother :
![Alt Text](https://github.com/Rahul-Choudhary-3614/Robotic-Car-Path-Planning-/blob/main/images/path_smoothing.gif?raw=true)

Particle Filter:
![Alt Text](https://github.com/Rahul-Choudhary-3614/Robotic-Car-Path-Planning-/blob/main/images/particle_filter.gif?raw=true)
