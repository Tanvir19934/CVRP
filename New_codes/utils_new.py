from config_new import *
from gurobipy import Model, GRB, quicksum
import itertools
from itertools import permutations


def ev_travel_cost(route):
    b = 1
    l = 0
    for i in range(len(route)-1):
        l+=q[route[i]]
        b = b - (a[route[i],route[i+1]]/EV_velocity)*(gamma+gamma_l*l) 
    cost = 260*EV_cost*(1-b)
    return cost

def gv_tsp_cost(route):
    cost_GV = a[(route[0],route[1])]*GV_cost
    l = 0
    try:
        route = eval(route)
    except:
        for i in range(1,len(route)-1):
            l+=q[route[i]]
            cost_GV += a[(route[i],route[i+1])]*GV_cost*l
        return cost_GV
    
def check_values(d):
    for key, value in d.items():
        if value != 0 and value!=1:
            return False
    return True

def partial_path(label,current_node):
    node = current_node
    path = []
    while label:
        path.append(label.node)
        label = label.parent
    if node in path[1:]:
        return True
    else: return False

def reconstruct_path(label):
    path = []
    while label:
        path.append(label.node)
        label = label.parent
    path = path[::-1]
    if path[0]=='s':
        path[0] = 0
    if path[-1]=='t':
        path[-1]= 0
    return path

def construct_tour(edges):
    # Build a dictionary for quick lookup of the next node
    next_node_map = {i: j for i, j in edges}
    
    # Start from any node (choosing the first node from the first edge)
    start_node = edges[0][0]
    route = [start_node]
    
    current_node = start_node
    while True:
        next_node = next_node_map.get(current_node)
        if next_node is None or next_node == start_node:  # End of cycle
            break
        route.append(next_node)
        current_node = next_node

    route.append(start_node)  # Complete the cycle
    return route

def print_solution(final_model) -> None:
    model_vars = {var.VarName: var.X for var in final_model.getVars()}
    var_names = list(model_vars.keys())

    solution_routes = []
    payments = {}

    for var in var_names:
        value = round(model_vars[var], 2) # Get the variable's value

        if var.startswith("p_"):  # Payment variables
            payments[int(var.split("_")[-1])] = value  # Convert index to int
        elif value != 0 and var.startswith("y_r_["):  # Route variables
            print(f"{var}: {value}")
            solution_routes.append(list(eval(var[5:-1])))  # Extract tuple inside []

    # Print results
    c_r = {tuple(item):0 for item in solution_routes}
    for item in solution_routes:
        if len(item) > 3:
            c_r[tuple(item)] = ev_travel_cost(item)

    total_dv_miles_traveled = 0
    total_ev_miles_traveled = 0

    for route in solution_routes:
        if len(route) == 3:
            for i in range(0,len(route)-1):
                total_dv_miles_traveled += a[(route[i], route[i+1])]
        else:
            for i in range(0,len(route)-1):
                total_ev_miles_traveled += a[(route[i], route[i+1])]
    print(solution_routes)
    print(f"Total miles cost (ev+dv): {total_dv_miles_traveled*w_dv + total_ev_miles_traveled*w_ev}")
    print(f"Total EV miles traveled: {total_ev_miles_traveled}")
    print(f"Total DV miles traveled: {total_dv_miles_traveled}")
    print(f"Total mniles traveled: {total_dv_miles_traveled + total_ev_miles_traveled}")
    print(f"Objective value: {final_model.getObjective().getValue()}")
    print(f"Objective value (manual): {total_dv_miles_traveled*w_dv + total_ev_miles_traveled*w_ev+theta*(sum(c_r.values())-sum(payments.values()))}")
    print("Total payment received:", sum(payments.values()))
    print("Total coalition cost:", sum(c_r.values()))
    print("Total Subsidy:", sum(c_r.values())-sum(payments.values()))
    print(f"Payments: {payments}")

def generate_all_possible_routes(N):
    
    all_routes = []

    def generate_k_degree_coalition(N, k):
        # Generate all combinations of k nodes
        combinations = list(itertools.combinations(N, k))
        # Generate all possible routes starting and ending at depot 0
        degree_k_coalition = [tuple([0] + list(comb) + [0]) for comb in combinations]
        return degree_k_coalition
    

    for item in N:
        all_routes.extend(generate_k_degree_coalition(N, item))
    
    return all_routes

def tsp_tour(route):

    if len(route) == 3 and route[0]==0:
        return a[(0,route[1])]* GV_cost + a[(route[1],0)] * q[route[1]] * GV_cost, route

    intermediate_nodes = route[1:-1]

    all_routes = [[0] + list(p) + [0] for p in permutations(intermediate_nodes)]

    routes_list = [tuple(all_routes) for all_routes in all_routes]

    route_cost = {}

    for item in routes_list:
        route_cost[tuple(item)] = gv_tsp_cost(item)

    model = Model("TSP")
    x = model.addVars(routes_list, vtype=GRB.BINARY, name="x")

    model.addConstr(quicksum(x[i] for i in routes_list) == 1)

    model.setObjective(quicksum(route_cost[i] * x[i] for i in routes_list), GRB.MINIMIZE)


    model.Params.OutputFlag = 0
    model.optimize()

    # Extract the solution
    if model.status == GRB.OPTIMAL:
        for i in routes_list:
            if x[i].X > 0.5:
                tour = i
                break
        return model.getObjective().getValue(), tour