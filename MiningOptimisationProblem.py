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