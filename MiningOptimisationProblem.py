#Importing all the packages that will be used.
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
from gurobipy import GRB

#Road quality Rc associated with all of the arcs/roads (p1, p2, p3,..,p8).
road_qualities = np.array([0.5, 1, 0.1, 0.06, 0.15, 0.2, 0.05, 0], dtype=float)

#Road distance L associated with all of the arcs/roads (p1, p2, p3,..,p8).
road_distances = np.array([0.5, 1, 0.1, 0.06, 0.15, 0.2, 0.05, 0], dtype=float)

#W_L. Given width of each road
#road_widths = np.array([0.5, 1, 0.1, 0.06, 0.15, 0.2, 0.05, 0], dtype=float)

#Weight limits of each road. Accounts for geology that will limit weight on road and introduces kg.
#road_weight_limits = np.array([0.5, 1, 0.1, 0.06, 0.15, 0.2, 0.05, 0], dtype=float)

#S_L. Given speed limit of each road. We want to be acting at max speed.
#speeds = np.array([0.5, 1, 0.1, 0.06, 0.15, 0.2, 0.05, 0], dtype=float)

#s_t. Given road straightness class of each road.
#road_straightness = np.array([0.5, 1, 0.1, 0.06, 0.15, 0.2, 0.05, 0], dtype=float)

#gradient of the given road. Given gradient/slope of each road
#road_grads = np.array([0.5, 1, 0.1, 0.06, 0.15, 0.2, 0.05, 0], dtype=float)

#num_vars represents the number of decision variables and will be used for a variety of calculations.
num_vars = len(road_distances) 

#This number represent the demand of crusher 1.
crusher_demand = 30000

#Creating matrix that'll evaluate the equality constraints. There are x rows, representing the x equality constraints. 
#For the number of columns, I set it to the number of decision variables (e.g., ...). In each entry of the Aeq matrix 
#(the LHS of the equality constraints), there will be a vector featuring every single decision variable (x decision 
#variables). beq represents the RHS of the equality constraints. As there are x constraints, beq has x entries.

Aeq = np.array([
    [0,   0,   0,   0.85,   0, 0.98, 0, 0.94, 0], #equality constraint 1 (could be crusher demand!)
    [0, 0,   0,   0,      0.9,    0,    0.95, 0, 0 ], #equality constraint 2
    [1,   0,  -1,  -1,     -1,      0,    0, 0, 0],   #equality constraint 3
    [0,   1,   0,   0,      0,     -1,   -1,   0, 0],  #equality constraint 4
    [0,   0,   -0.98, 0,     0,      0,    0,   1, 1]  #equality constraint 5
    ]) 

beq = np.array([crusher demand, 0, 0, 0, 0])


#Defining the lower bound capacity constraints of all decision variables. This is simply 0 because
#roads can't have below 0 volume!:
lb = np.zeros(num_vars)

def cost_function(roadDistance,roadQuality,speed,volume):
    #actual cost function

#Capacity function that returns the upper volume capacity for a given road.
def capacity_function(roadDistance, roadQuality, speed, road_s_t, road_gradient, road_weight, road_wth):
    return (((speed(1-road_gradient/45))*road_weight*road_wth)/(roadQuality*road_s_t*roadDistance))

#a given vector (ub) a given amount of times (intervals).
ub = np.array([capacity_function(road_distances[0], road_qualities[0], speeds[0], ...),
cost_function(road_distances[1], road_qualities[1], speeds[1], ...),
cost_function(road_distances[2], road_qualities[2], speeds[2], ...),...])

#A Gurobi Model is created to find the minimal cost.
model = gp.Model("Mine Minimum Cost")

#I add all the decision variables to the Gurobi model and apply the capacity constraints to them
#using the upper and lower bounds.
x = model.addMVar(shape=num_vars, lb=lb, ub=ub, name="road flow capacity variables")