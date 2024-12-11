import numpy as np
class plan:

    # --------
    # init: 
    #    creates an empty plan
    #

    def __init__(self, grid, init, goal,weight_data,weight_smooth,cost = 1,tolerance = 0.000001):
        self.cost = cost
        self.weight_data = weight_data
        self.weight_smooth = weight_smooth
        self.tolerance = tolerance
        self.grid = grid
        self.init = init
        self.goal = goal
        self.make_heuristic()
        self.path = []
        self.spath = []

    # --------
    #
    # make heuristic function for a grid
        
    def make_heuristic(self):
      self.heuristic = [[0 for row in range(len(self.grid[0]))] 
                          for col in range(len(self.grid))]
      for i in range(len(self.grid)):    
        for j in range(len(self.grid[0])):
          self.heuristic[i][j] = abs(i - self.goal[0]) + abs(j - self.goal[1])
    # ------------------------------------------------
    # 
    # A* for searching a path to the goal
    #
    #

    def astar(self):


        if self.heuristic == []:
            raise ValueError("Heuristic must be defined to run A*")

        # internal motion parameters
        delta = [[-1,  0], # go up
                 [ 0,  -1], # go left
                 [ 1,  0], # go down
                 [ 0,  1]] # do right


        # open list elements are of the type: [f, g, h, x, y]

        closed = [[0 for row in range(len(self.grid[0]))] for col in range(len(self.grid))]
        action = [[0 for row in range(len(self.grid[0]))] for col in range(len(self.grid))]

        closed[self.init[0]][self.init[1]] = 1


        x = self.init[0]
        y = self.init[1]
        h = self.heuristic[x][y]
        g = 0
        f = g + h

        open = [[f, g, h, x, y]]

        found  = False # flag that is set when search complete
        resign = False # flag set if we can't find expand
        count  = 0


        while not found and not resign:

            # check if we still have elements on the open list
            if len(open) == 0:
                resign = True
                print('###### Search terminated without success')
            else:
                # remove node from list
                open.sort()
                open.reverse()
                next = open.pop()
                x = next[3]
                y = next[4]
                g = next[1]

            # check if we are done

            if x == self.goal[0] and y == self.goal[1]:
                found = True
                #print('A* search successful')

            else:
                # expand winning element and add to new open list
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(self.grid) and y2 >= 0 \
                            and y2 < len(self.grid[0]):
                        if closed[x2][y2] == 0 and self.grid[x2][y2] == 0:
                            g2 = g + self.cost
                            h2 = self.heuristic[x2][y2]
                            f2 = g2 + h2
                            open.append([f2, g2, h2, x2, y2])
                            closed[x2][y2] = 1
                            action[x2][y2] = i

            count += 1

        # extract the path

        invpath = []
        x = self.goal[0]
        y = self.goal[1]
        invpath.append([x, y])
        while x != self.init[0] or y != self.init[1]:
            x2 = x - delta[action[x][y]][0]
            y2 = y - delta[action[x][y]][1]
            x = x2
            y = y2
            invpath.append([x, y])

        self.path = []
        for i in range(len(invpath)):
            self.path.append(invpath[len(invpath) - 1 - i])




    # ------------------------------------------------
    # 
    # this is the smoothing function
    #
    def smooth(self):

        if self.path == []:
            raise ValueError("Run A* first before smoothing path")

        self.spath = [[0 for row in range(len(self.path[0]))] for col in range(len(self.path))]   
        for i in range(len(self.path)):
            for j in range(len(self.path[0])):
                self.spath[i][j] = self.path[i][j]

        change = self.tolerance
        while change >= self.tolerance:
            change = 0.0
            for i in range(1, len(self.path)-1):
                for j in range(len(self.path[0])):
                    aux = self.spath[i][j]
                    dummy = self.spath[i][j]
                    self.spath[i][j] += self.weight_data * (self.path[i][j] - self.spath[i][j])
                    self.spath[i][j] += self.weight_smooth * (self.spath[i-1][j] + self.spath[i+1][j] - (2.0 * dummy))
                    if i >= 2:
                      self.spath[i][j] += 0.5 * self.weight_smooth * (2.0 * self.spath[i-1][j] - self.spath[i-2][j] - dummy)
                    if i <= len(self.path) - 3:
                        self.spath[i][j] += 0.5 * self.weight_smooth * (2.0 * self.spath[i+1][j] - self.spath[i+2][j] -dummy)
            change += abs(aux - self.spath[i][j])
                

def get_path(grid, init, goal,weight_data,weight_smooth,cost = 1,tolerance = 0.000001):
    path = plan(grid, init, goal,weight_data,weight_smooth,cost,tolerance)
    path.astar()
    path.smooth()
    return path.spath
