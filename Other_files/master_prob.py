from collections import defaultdict
from config import *
import pickle
import numpy as np
from gurobipy import Model, GRB, quicksum
import copy

""" filenames = ['x_d_df.pkl', 'x_e_df.pkl', 'l_df.pkl']
# Dictionary to hold the loaded dataframes
loaded_dataframes = {}
# Loop through the filenames and load each dataframe
for filename in filenames:
   with open(filename, 'rb') as file:
      loaded_dataframes[filename[:-4]] = pickle.load(file)  # Remove .pkl extension for key
# Access the loaded dataframes
x_d_df = loaded_dataframes['x_d_df']
x_e_df = loaded_dataframes['x_e_df'] """




rnd = np.random
rnd.seed(10)

#override some config parameters
q[0] = 0
#T_max_EV = 700
#print(T_max_EV)

def ev_travel_cost(route):
    rev=False
    b = 1
    l = 0
    for i in range(len(route)-1):
        l+=q[route[i]]
        b = b - (a[route[i],route[i+1]]/EV_velocity)*(gamma+gamma_l*l) 
    clockwise_cost = 260*EV_cost*(1-b)
    
    b = 1
    l = 0
    route = list(route)
    route.reverse()
    for i in range(len(route)-1):
        l+=q[route[i]]
        b = b - (a[route[i],route[i+1]]/EV_velocity)*(gamma+gamma_l*l) 
    counter_clockwise_cost = 260*EV_cost*(1-b)

    if clockwise_cost>counter_clockwise_cost:
        rev=True
    
    return min(clockwise_cost,counter_clockwise_cost),rev

def gv_travel_cost(route):
    cost_GV = 0 
    try:
        route = eval(route)
    except:
        for i in range(1,len(route)-1):
            cost_GV += a[(route[i],route[0])]*GV_cost*q[route[i]] + a[(route[i],route[0])]*GV_cost
        return cost_GV

def gv_node_cost(node):
    return a[(node,0)]*GV_cost*q[node] + a[(0,node)]*GV_cost

degree_2_coalition_cost = {}
for item in degree_2_coalition:
    route = copy.deepcopy(item)
    cost, rev = ev_travel_cost(route)
    if rev:
        degree_2_coalition_cost[item[::-1]] = cost
    else: degree_2_coalition_cost[item] = cost

degree_2_coalition = tuple(degree_2_coalition_cost)




###########Gurobi model############
mdl = Model('master_problem')


#SETS and PARAMETERS
r_set = [[0, node, 0] for node in V if node != 0]

#r_set[0].insert(2,6)
#r_set.remove([0,6,0])
#r_set = [[0,2,4,0],[0,10,0],[0,7,0],[0,1,8,0],[0,9,5,0],[0,3,0],[0,6,0]]
r_set = r_set + list(degree_2_coalition) #+ list(degree_3_coalition) #+list(degree_4_coalition)+list(degree_5_coalition)


r_set = [tuple(route) for route in r_set]

r_set_ev_cost = {}
r_set_gv_cost = {}

arc_set_Ar = {}
for item in r_set:
    arr = []
    if len(item)>3:
        for i in range(1,len(item)-2):
            arr.append(tuple([item[i],item[i+1]]))
    arc_set_Ar[item]=arr




for item in r_set:
    route = copy.deepcopy(item)
    cost, rev = ev_travel_cost(route)
    if rev:
        r_set_ev_cost[tuple(item)] = cost
        #r_set_ev_cost[tuple(item[::-1])] = cost
    else: r_set_ev_cost[tuple(item)] = cost
    r_set_gv_cost[tuple(item)]  = gv_travel_cost(route)

individual_cost = {}

for item in r_set:
    coalition_cost = r_set_ev_cost[item]
    sum_standalone_cost = r_set_gv_cost[item]
    for nodes in item[1:-1]:
        individual_cost[nodes] = coalition_cost * ((a[0,nodes]+a[nodes,0]*q[nodes])*GV_cost)/sum_standalone_cost



c_r = {item:0 for item in r_set}
for item in r_set:
   l = len(item)
   for i in range(l-1):
      c_r[item] += a[(item[i],item[i+1])]

delta = {}

# Iterate over all routes and nodes
for route in r_set:
    for i in range(1,len(V)+1):  # Nodes from 0 to 10
        # Set delta_i,r = 1 if node i is in route r, otherwise 0
        delta[(i, route)] = 1 if i in route else 0




#NEED ==> r_set, individual cost, degree_2_coalition_cost, degree_2_coalition,
#         r_set_ev_cost, r_set_gv_cost, intermediate_arcs

#DECISION VARIABLES
y_r = {}
for item in r_set:
    y_r[tuple(item)] = mdl.addVar(vtype=GRB.CONTINUOUS, name=f"y_r[{item}]", lb=0, ub=1)

mdl.update()

#CONSTRAINTS
mdl.addConstrs((quicksum(delta[(i, route)] * y_r[route] for route in r_set) == 1 for i in N), name=f"delta")
mdl.addConstrs(((r_set_ev_cost[route]-r_set_gv_cost[route])*y_r[route] <= 0 for route in r_set), name=f"IR")

#for route in r_set:
#    for r in arc_set_Ar[route]:
#        for (o,i,j,o) in degree_2_coalition:
#            if r[0] == i and j not in route:
#                mdl.addConstr((individual_cost[r[0]] + individual_cost[r[1]]) * y_r[route] <= degree_2_coalition_cost[(0,r[0],j,0)], name=f"Stability_{r[0]}_{r[1]}_{r[0]}_{j}")
#            if r[0] == j and i not in route:
#                mdl.addConstr((individual_cost[r[0]] + individual_cost[r[1]]) * y_r[route] <= degree_2_coalition_cost[(0,i,r[0],0)], name=f"Stability_{r[0]}_{r[1]}_{i}_{r[0]}")
#            if r[1] == i and j not in route:
#                mdl.addConstr((individual_cost[r[0]] + individual_cost[r[1]]) * y_r[route] <= degree_2_coalition_cost[(0,r[1],j,0)], name=f"Stability_{r[0]}_{r[1]}_{r[1]}_{j}")
#            if r[1] == j and i not in route:
#                mdl.addConstr((individual_cost[r[0]] + individual_cost[r[1]]) * y_r[route] <= degree_2_coalition_cost[(0,i,r[1],0)], name=f"Stability_{r[0]}_{r[1]}_{i}_{r[1]}")

#for route in r_set:
#    for r in route[1:-1]:
#        for (o,i,j,o) in degree_2_coalition:
#            if r == i:
#                mdl.addConstr(individual_cost[r] * y_r[route] <= (gv_node_cost(r)/(gv_node_cost(r)+gv_node_cost(j))) * degree_2_coalition_cost[(0,r,j,0)], name=f"Stability_r{route}_{r}_{j}")


#for route1 in r_set:
#    for route2 in r_set:
#        #if route1!=route2:
#            for r1 in route1[1:-1]:
#                for r2 in route2[1:-1]:
#                    if r1 != r2:
#                        try:
#                            mdl.addConstr(individual_cost[r1] * y_r[route1] + individual_cost[r2] * y_r[route2]<= degree_2_coalition_cost[(0,r1,r2,0)], name=f"Stability_r{r1}_{r2}")
#                        except:
#                            mdl.addConstr(individual_cost[r1] * y_r[route1] + individual_cost[r2] * y_r[route2]<= degree_2_coalition_cost[(0,r2,r1,0)], name=f"Stability_r{r1}_{r2}")

            
mdl.update()
mdl.modelSense = GRB.MINIMIZE

#SET OBJECTIVE
mdl.setObjective((quicksum(c_r[route]*y_r[route] for route in r_set)))
mdl.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/master_prob.lp")
mdl.optimize()
#mdl.Params.MIPGap = 0.05
#mdl.params.NonConvex = 2
#mdl.Params.TimeLimit = 2000 #seconds


# Retrieve dual values and map them to routes
dual_values = {}
for constr in mdl.getConstrs():
    if constr.ConstrName.startswith("delta"):
        # Extract the route and node index from the constraint name
        i = int(constr.ConstrName.split('[')[1][:-1])
        # The corresponding route can be inferred by matching 'i' and the route
        dual_values[i] = constr.Pi
with open('dual_values.pkl', 'wb') as f:
    pickle.dump(dual_values, f)

def get_vars(item,opt_route):
   vars = [var for var in opt_route.getVars() if f"{item}" in var.VarName]
   names = opt_route.getAttr('VarName', vars)
   values = opt_route.getAttr('X', vars)
   return dict(zip(names, values))
y_r_result = get_vars('y_r',mdl)

for item in y_r_result:
    if y_r_result[item]>0:
        print(f"{item}={y_r_result[item]}")

# Save (pickle) the dictionary to a file
with open('y_r_result.pkl', 'wb') as file:
    pickle.dump(y_r_result, file)






            