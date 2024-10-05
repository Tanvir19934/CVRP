from gurobipy import GRB, Model, quicksum
import random
import pickle
from config import *
import itertools
import copy
import time
def ev_travel_cost(route):
    rev=False
    b = 1
    l = 0
    q[0] = 0
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

def standalone_cost(adjustment):
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
    return standalone_cost_degree_2


def load_routes():
    filenames = ['data.pkl']
    # Dictionary to hold the loaded dataframes
    loaded_dataframes = {}
    # Loop through the filenames and load each dataframe
    for filename in filenames:
        with open(filename, 'rb') as file:
            loaded_dataframes[filename[:-4]] = pickle.load(file)  # Remove .pkl extension for key
    # Access the loaded dataframes
    ev_routes = loaded_dataframes['data']
    return ev_routes


def lp(route, standalone_cost_degree_2):
    mdl = Model(f'lp{route}')
    N_lp = route[1:-1]
    V_lp = route[0:-1]
    C_route, _ = ev_travel_cost(route)
    # Decision variables
    p = {}
    e_IR = {}
    e_S = {}

    e_BB = mdl.addVar(vtype=GRB.CONTINUOUS, name = "e_BB")
    for i in N_lp:
        p[i] = mdl.addVar(vtype=GRB.CONTINUOUS, name = f"p{i}")
        e_IR[i] = mdl.addVar(vtype=GRB.CONTINUOUS, name = f"e_IR{i}")
        e_S[i] = mdl.addVar(vtype=GRB.CONTINUOUS, name = f"e_S{i}")

    #IR
    mdl.addConstrs((p[i]<=a[i,0]*GV_cost*q[i]+a[i,0]*GV_cost+e_IR[i]) for i in N_lp)

    #BB
    mdl.addConstr(quicksum(p[i] for i in N_lp)+(quicksum(e_IR[i] for i in N_lp)) + (quicksum(e_S[i] for i in N_lp))+ e_BB >= C_route)

    for i in N_lp:
        for j in N_lp:
            if i!=j:
                mdl.addConstr(p[i]<=standalone_cost_degree_2[i,j][i]+e_S[i],name="stability")

    mdl.addConstrs((p[i]>=(a[(i,j)]/EV_velocity)*(gamma+gamma_l*q[i])*260*EV_cost) for i in N_lp for j in N_lp if i!=j)


    mdl.setObjective(e_BB + (quicksum(e_IR[i] for i in N_lp)) + (quicksum(e_S[i] for i in N_lp)))


    mdl.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/coalition.lp.lp")
    mdl.optimize()
    def get_vars(item):
       vars = [var for var in mdl.getVars() if f"{item}" in var.VarName]
       names = mdl.getAttr('VarName', vars)
       values = mdl.getAttr('X', vars)
       return dict(zip(names, values))
    
    p_result = get_vars('p')
    e_S_result = get_vars('e_S')
    e_BB_result = get_vars('e_BB')
    e_IR_result = get_vars('e_IR')

    return p_result,e_S_result,e_BB_result,e_IR_result

if __name__ == "__main__":
    start = time.perf_counter()
    standalone_cost_degree_2 = standalone_cost(0.8)
    ev_routes = load_routes()
    p_result_dict = {} 
    e_S_result_dict = {}
    e_BB_result_dict = {}
    e_IR_result_dict = {}
    total_p = 0
    total_S = 0
    total_IR = 0
    total_BB = 0
    for route in ev_routes:
        p_result,e_S_result,e_BB_result,e_IR_result = lp(route,standalone_cost_degree_2)
        p_result_dict[f"{route}"] = p_result
        e_S_result_dict[f"{route}"] = e_S_result
        e_BB_result_dict[f"{route}"] = e_BB_result
        e_IR_result_dict[f"{route}"] = e_IR_result
        total_p += sum(p_result.values())
        total_S += sum(e_S_result.values())
        total_IR += sum(e_IR_result.values())
        total_BB += sum(e_BB_result.values())
    end = time.perf_counter()
    print(f"total payment = {total_p}")
    print(f"total stability = {total_S}")
    print(f"total IR = {total_IR}")
    print(f"total BB = {total_BB}")
    print(f"total subsidy = {total_BB+total_IR+total_S}")
    print(f"Execution time = {end-start}")



    2
