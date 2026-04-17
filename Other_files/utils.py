import numpy as np
from config import *
import itertools
import copy


rnd = np.random
#rnd.seed(10)

#override some config parameters
q[0] = 0



def generate_k_degree_coalition(N, k):
    # Generate all combinations of k nodes
    combinations = list(itertools.combinations(N, k))
    # Generate all possible routes starting and ending at depot 0
    degree_k_coalition = [tuple([0] + list(comb) + [0]) for comb in combinations]
    return degree_k_coalition

degree_2_coalition = generate_k_degree_coalition(N, 2)
degree_2_coalition_final = []
degree_2_coalition_route = []

for item in degree_2_coalition:
    if a[item[0],item[1]] + a[item[1],item[2]] + a[item[2],item[0]] > a[item[0],item[2]] + a[item[2],item[1]] + a[item[1],item[0]]:
        degree_2_coalition_route.append(tuple([item[0],item[2],item[1],item[0]]))
        degree_2_coalition_final.append(tuple([item[2],item[1]]))
    else:
      degree_2_coalition_route.append(tuple([item[0],item[1],item[2],item[0]]))
      degree_2_coalition_final.append(tuple([item[1],item[2]]))

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

degree_2_coalition_cost = {}
for item in degree_2_coalition:
    route = copy.deepcopy(item)
    cost, rev = ev_travel_cost(route)
    if rev:
        degree_2_coalition_cost[tuple([item[2],item[1]])] = cost
    else: degree_2_coalition_cost[tuple([item[1],item[2]])] = cost

standalone_cost_degree_2 = {}
for item in degree_2_coalition_cost:
   standalone_cost_degree_2[item] = {}
for item in degree_2_coalition_cost:
   standalone_cost_degree_2[item][item[0]] = adjustment*degree_2_coalition_cost[item]* (a[item[0],0]*GV_cost+a[item[0],0]*GV_cost*q[item[0]])/((a[item[0],0]*GV_cost+a[item[0],0]*GV_cost*q[item[0]])+(a[item[1],0]*GV_cost+a[item[1],0]*GV_cost*q[item[1]]))
   standalone_cost_degree_2[item][item[1]] = adjustment*degree_2_coalition_cost[item]* (a[item[1],0]*GV_cost+a[item[1],0]*GV_cost*q[item[1]])/((a[item[0],0]*GV_cost+a[item[0],0]*GV_cost*q[item[0]])+(a[item[1],0]*GV_cost+a[item[1],0]*GV_cost*q[item[1]]))
standalone_cost_degree_2_copy = copy.deepcopy(standalone_cost_degree_2)
for item in standalone_cost_degree_2_copy:
   standalone_cost_degree_2[(item[1],item[0])]=standalone_cost_degree_2[item]