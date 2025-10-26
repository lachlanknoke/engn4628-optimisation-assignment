#Importing all the packages that will be used.
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import math
from gurobipy import GRB

def gradient(elevation,distance):
    if elevation <=0:
        return 0
    else:
        return ((180/math.pi)*(math.atan((elevation/(distance*1000)))))

w_L = 1250
#Speed limits associated with all of the arcs (m1->int3, m1->int4, m1->int5,m2->locations, int1->crusher 1, int 1->crusher2, int 1-> int 2
#int2->locations,int3->locations,int4->locations,int5->locations,Crusher1->locations, Crusher2->locations).
s_L = {'m1_int3':60, 'm1_int4':40, 'm1_int5':40, 'm2_int1':30,'int1_C1':60,'int1_C2':50,'int1_int2':30,'int2_C1':50,'int2_C2':50,'int2_int1':30,
       'int2_int3':50,'int3_C2':60,'int3_m1':60,'int3_int2':50,'int4_C2':60,'int4_m1':40,'int5_C1':60,'int5_C2':50,'int5_m1':40,'C1_C2':60,
       'C1_int1':60,'C1_int2':50,'C1_int5':60,'C2_C1':60,'C2_int1':50,'C2_int2':50,'C2_int3':60,'C2_int4':60,'C2_int5':50}

#Straightness associated with all of the arcs (m1->int3, m1->int4, m1->int5,m2->locations, int1->crusher 1, int 1->crusher2, int 1-> int 2
#int2->locations,int3->locations,int4->locations,int5->locations,Crusher1->locations, Crusher2->locations).
s_T = {'m1_int3': 1, 'm1_int4': 2.2, 'm1_int5': 3, 'm2_int1': 1, 'int1_C1': 2.2, 'int1_C2': 3, 'int1_int2': 1, 'int2_C1': 3, 'int2_C2': 2.2, 'int2_int1': 1, 'int2_int3': 2.2,
 'int3_C2': 1, 'int3_m1': 1, 'int3_int2': 2.2, 'int4_C2': 1, 'int4_m1': 2.2, 'int5_C1': 2.2, 'int5_C2': 2.2, 'int5_m1': 3, 'C1_C2': 2.2, 'C1_int1': 2.2, 'C1_int2': 3,
  'C1_int5': 2.2, 'C2_C1': 2.2, 'C2_int1': 3, 'C2_int2': 2.2, 'C2_int3': 1, 'C2_int4': 1, 'C2_int5': 2.2}

#curvature associated with all of the arcs (m1->int3, m1->int4, m1->int5,m2->locations, int1->crusher 1, int 1->crusher2, int 1-> int 2
#int2->locations,int3->locations,int4->locations,int5->locations,Crusher1->locations, Crusher2->locations).
r_C = {'m1_int3': 2.2, 'm1_int4': 3, 'm1_int5': 3, 'm2_int1': 1, 'int1_C1': 2.2, 'int1_C2': 3, 'int1_int2': 1, 'int2_C1': 3, 'int2_C2': 2.2, 'int2_int1': 1, 'int2_int3': 2.2,
 'int3_C2': 1, 'int3_m1': 2.2, 'int3_int2': 2.2, 'int4_C2': 1, 'int4_m1': 3, 'int5_C1': 2.2, 'int5_C2': 2.2, 'int5_m1': 3, 'C1_C2': 2.2, 'C1_int1': 2.2, 'C1_int2': 3,
  'C1_int5': 2.2, 'C2_C1': 2.2, 'C2_int1': 3, 'C2_int2': 2.2, 'C2_int3': 1, 'C2_int4': 1, 'C2_int5': 2.2}

#Distances associated with all of the arcs (m1->int3, m1->int4, m1->int5,m2->locations, int1->crusher 1, int 1->crusher2, int 1-> int 2
#int2->locations,int3->locations,int4->locations,int5->locations,Crusher1->locations, Crusher2->locations).
distances = {'m1_int3':7.65, 'm1_int4':6.72, 'm1_int5':9.71, 'm2_int1':2.31,'int1_C1':15.12,'int1_C2':16.93,'int1_int2':1.83,'int2_C1':19.89,'int2_C2':11.86,'int2_int1':1.83,
       'int2_int3':8.07,'int3_C2':7.53,'int3_m1':7.65,'int3_int2':8.07,'int4_C2':6.66,'int4_m1':6.72,'int5_C1':21.4,'int5_C2':4.76,'int5_m1':9.71,'C1_C2':15.65,
       'C1_int1':15.12,'C1_int2':19.89,'C1_int5':21.4,'C2_C1':15.65,'C2_int1':16.93,'C2_int2':11.86,'C2_int3':7.53,'C2_int4':6.66,'C2_int5':4.76}

distances = {'m1_int3':7.65, 'm1_int4':4.02, 'm1_int5':9.71, 'm2_int1':2.31,'int1_C1':15.12,'int1_C2':16.93,'int1_int2':1.83,'int2_C1':19.89,'int2_C2':11.86,'int2_int1':1.83,
       'int2_int3':8.07,'int3_C2':7.53,'int3_m1':7.65,'int3_int2':8.07,'int4_C2':8.59,'int4_m1':6.72,'int5_C1':21.1,'int5_C2':3.37,'int5_m1':9.71,'C1_C2':15.65,
       'C1_int1':15.12,'C1_int2':19.89,'C1_int5':21.4,'C2_C1':15.65,'C2_int1':16.93,'C2_int2':11.86,'C2_int3':7.53,'C2_int4':6.66,'C2_int5':4.76}

#Gradients associated with all of the arcs (m1->int3, m1->int4, m1->int5,m2->locations, int1->crusher 1, int 1->crusher2, int 1-> int 2
#int2->locations,int3->locations,int4->locations,int5->locations,Crusher1->locations, Crusher2->locations).
gradients = {'m1_int3':gradient(-8,distances['m1_int3']), 'm1_int4':gradient(-6,distances['m1_int4']), 'm1_int5':gradient(16,distances['m1_int5']), 'm2_int1':gradient(13,distances['m2_int1'])
             ,'int1_C1':gradient(-1,distances['int1_C1']),'int1_C2':gradient(17,distances['int1_C2']),'int1_int2':gradient(-3,distances['int1_int2']),'int2_C1':gradient(2,distances['int2_C1']),
             'int2_C2':gradient(20,distances['int2_C2']),'int2_int1':gradient(3,distances['int2_int1']),'int2_int3':gradient(3,distances['int2_int3']),'int3_C2':gradient(17,distances['int3_C2']),
             'int3_m1':gradient(8,distances['int3_m1']),'int3_int2':gradient(-3,distances['int3_int2']),'int4_C2':gradient(15,distances['int4_C2']),'int4_m1':gradient(6,distances['int4_m1']),
             'int5_C1':gradient(-25,distances['int5_C1']),'int5_C2':gradient(-7,distances['int5_C2']),'int5_m1':gradient(-16,distances['int5_m1']),'C1_C2':gradient(18,distances['C1_C2']),
             'C1_int1':gradient(1,distances['C1_int1']),'C1_int2':gradient(-2,distances['C1_int2']),'C1_int5':gradient(25,distances['C1_int5']),'C2_C1':gradient(-18,distances['C2_C1']),
             'C2_int1':gradient(-17,distances['C2_int1']),'C2_int2':gradient(-20,distances['C1_int2']),'C2_int3':gradient(-17,distances['C2_int3']),'C2_int4':gradient(-15,distances['C2_int4']),
             'C2_int5':gradient(7,distances['C2_int5'])}

#This arrays represents the demands of the two crushers.
crusher_demand = np.array([1150,1150])


#Creating an equality matrix for the flow constraints (m1->int3, etc.)
Aeq = np.array([
    [0, 0, 0, 1, -1, -1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], #Int1 flow in - flow out = 0
    [0, 0, 0, 0,  0,  0,  1,-1,-1,-1,-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], #Int2 flow in - flow out = 0
    [1, 0, 0, 0,  0,  0,  0, 0, 0, 0, 1, -1, -1, -1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],   #Int3 flow in - flow out = 0
    [0, 1, 0, 0,  0,  0,  0, 0, 0,0,  0,  0,  0,  0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,1,0],  #Int4 flow in - flow out = 0
    [0, 0, 1, 0,  0,  0,  0, 0, 0,0,0,0,0,0,0,0,-1,-1,-1,0,0,0,1,0,0,0,0,0,1],  #Int5 flow in - flow out = 0
    [0, 0, 0, 0,  1,  0,  0, 1, 0,0,0,0,0,0,0,0,1,0,0,-1,-1,-1,-1,1,0,0,0,0,0],   #Crusher 1 demand flow in - flow out = 1150
    [0, 0, 0, 0,  0,  1,  0, 0, 1,0,0,1,0,0,1,0,0,1,0,1,0,0,0,-1,-1,-1,-1,-1,-1] #Crusher 2 demand flow in - flow out = 1150 
    ]) 

#Creating an initial equality row vector that will be compared to the above Aeq
#matrix to determine if equality constraints are met.
beq = np.array([0,0,0,0,0,1150,1150])      

#Count all the decision variables.
num_vars = len(distances)

#Defining the lower bound capacity constraints of all the decision variables:
lb = np.zeros(num_vars)

#Function that finds the max capacity of a given road.
def capacity_function(grad,sT,rC,sL,wL,distance):
    part1 = (1-grad/25)/(sT*rC)
    part2 = (sL*wL)/distance
    return (part1*part2)

#Function that finds the cost per unit of a given road. As it is quite a long equation, it was split up into
#3 components.
def cost_function(sL,rC,sT,distance):
    fuel_cost = 1.6*0.04*pow((250/304),0.35)*pow((sL/64.5),1.22)
    Maintenance_cost = 0.000822*pow((250/304),0.9)*pow((sL/64.5),1.75)
    road_cost = distance*rC*sT*(fuel_cost+Maintenance_cost)
    return road_cost

#Define the upper bound of all the roads, which will need the capacity function and the dictionaries and arrays defined earlier. 
#Define the cost per unit of the roads, which will need the capacity function and the dictionaries and arrays defined earlier. 

# Create an empty dictionary to store the upper bounds of each road
road_capacities = np.array([])

# Create an empty dictionary to store the cost per unit of each road
road_costs = np.array([])

# Loop through all the road dictionary's keys (e.g., 'int2_C2','C2_C1')
for road in distances.keys():
    # For each arc, gather all the corresponding values
    speed = s_L[road]
    straight = s_T[road]
    curve = r_C[road]
    dist = distances[road]
    grad = gradients[road]
    
    # Call capacity function on all the values for this specific road to get max capacity for that road.
    capacity = capacity_function(grad,straight,curve,speed,w_L,dist)
    # Call cost function on all the values for this specific road to get cost per unit for that road.
    cost = cost_function(speed,curve,straight,dist)

    #Store the result in our new dictionaries
    road_capacities=np.append(road_capacities,capacity)
    road_costs=np.append(road_costs,cost)

#A Gurobi Model is created to find the minimal cost.
model = gp.Model("Mining network Minimum Cost")

#I add all the decision variables to the Gurobi model and apply the capacity constraints to them using the upper and lower bounds.
x = model.addMVar(shape=num_vars, lb=lb, ub=road_capacities, name="flow variables")


#Objective function. Want to find the cheapest possible combination of decision variables that meets the demand. 
model.setObjective(road_costs @ x, GRB.MINIMIZE)

#Equality Constraints. The flow and demand constraints of each intermediate node and crusher are added to the model.
model.addConstr(Aeq @ x == beq, name="flow and crusher demand constraints")

#I also need to add constraints that account for the limited volume from the two supply mines; initial mass + flow in - flow out >=0  for every single interval.
model.addConstr(-x[0]-x[1]-x[2]+x[13]+x[16]+x[19]+5500 >= 0, name="mine 1 flow inequality")
model.addConstr(-x[3]+5500 >= 0, name="mine 2 flow inequality")

#Solve the model
model.optimize()

#Print out the optimal solution if one is found.
if model.status == GRB.OPTIMAL:
    print(f"""
    Optimal Solution:
    cost: ${model.objVal:.2f}""")
    solution = x.X


print(solution)

#Print a dictiopnary to allow easy interpretation
#new_costs_dictionary = dict(zip(list(distances.keys()), solution))
#print(new_costs_dictionary)

new_costs_dictionary2 = dict(zip(list(distances.keys()), road_capacities))
print(new_costs_dictionary2)

#new_costs_dictionary2 = dict(zip(list(distances.keys()), road_costs))
#print(new_costs_dictionary2)


