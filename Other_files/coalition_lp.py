from gurobipy import GRB, Model, quicksum
import pickle
from config import *
import time
from utils import ev_travel_cost, standalone_cost_degree_2

def load_routes():
    filenames = ['data.pkl']
    # Dictionary to hold the loaded dataframes
    loaded_dataframes = {}
    # Loop through the filenames and load each dataframe
    for filename in filenames:
        with open(filename, 'rb') as file:
            loaded_dataframes[filename[:-4]] = pickle.load(file)  # Remove .pkl extension for key
    # Access the loaded dataframes
    if 'data' in loaded_dataframes:
        ev_routes = loaded_dataframes['data']
    else:
        raise KeyError("The key 'data' was not found in the loaded dataframes.")
    return ev_routes


def lp(route, standalone_cost_degree_2,N_whole):
    mdl = Model(f'lp{route}')
    N_lp = route[1:-1]
    V_lp = route[0:-1]
    C_route, _ = ev_travel_cost(route)
    immediate = {}
    for idx, i in enumerate(route):
        if i!=0:
            immediate[i]=route[idx+1]
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
    #mdl.addConstrs((p[i]<=(a[(i,0)]/EV_velocity)*(gamma+gamma_l*q[i])*260*EV_cost+(a[(i,0)]/EV_velocity)*(gamma+gamma_l*0)*260*EV_cost) for i in N_lp)


    #BB
    mdl.addConstr(quicksum(p[i] for i in N_lp)+(quicksum(e_IR[i] for i in N_lp)) + (quicksum(e_S[i] for i in N_lp))+ e_BB == C_route)

    #Stability
    for i in N_lp:
        for j in N_whole:
            if i!=j:
                mdl.addConstr(p[i]<=standalone_cost_degree_2[i,j][i]+e_S[i],name="stability")

    #mdl.addConstrs((p[i]<=(a[(i,j)]/EV_velocity)*(gamma+gamma_l*q[i])*260*EV_cost) for i in N_lp for j in N_lp if i!=j and immediate[i]==j)



    mdl.setObjective(e_BB + (quicksum(e_IR[i] for i in N_lp)) + (quicksum(e_S[i] for i in N_lp)))


    mdl.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/coalition.lp")
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
    #standalone_cost_degree_2 = standalone_cost(1)
    ev_routes = load_routes()
    N_whole = N

    p_result_dict = {} 
    e_S_result_dict = {}
    e_BB_result_dict = {}
    e_IR_result_dict = {}
    total_p = 0
    total_S = 0
    total_IR = 0
    total_BB = 0
    total_ev_cost = 0
    for route in ev_routes:
        p_result,e_S_result,e_BB_result,e_IR_result = lp(route,standalone_cost_degree_2,N_whole)
        p_result_dict[f"{route}"] = p_result
        e_S_result_dict[f"{route}"] = e_S_result
        e_BB_result_dict[f"{route}"] = e_BB_result
        e_IR_result_dict[f"{route}"] = e_IR_result
        total_p += sum(p_result.values())
        total_S += sum(e_S_result.values())
        total_IR += sum(e_IR_result.values())
        total_BB += sum(e_BB_result.values())
        ev_cost, _ = ev_travel_cost(route)
        total_ev_cost +=  ev_cost
    end = time.perf_counter()
    print(ev_routes)
    print(f"total payment = {total_p,total_ev_cost}")
    print(f"total stability = {total_S}")
    print(f"total IR = {total_IR}")
    print(f"total BB = {total_BB}")
    print(f"total subsidy = {total_BB+total_IR+total_S}")
    print(f"Execution time = {end-start}")
    print(p_result_dict)
    print(e_S_result_dict)

    node = 8
    l = []
    for item in standalone_cost_degree_2:
        if item[0]==node or item[1]==node:
            l.append(standalone_cost_degree_2[item][node])
    print(min(l))

    2
