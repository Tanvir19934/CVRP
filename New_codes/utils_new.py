from config_new import *
from gurobipy import Model, GRB, quicksum
import itertools

def ev_travel_cost(route):
    b = 1
    l = 0
    for i in range(len(route)-1):
        l+=q[route[i]]
        b = b - (a[route[i],route[i+1]]/EV_velocity)*(gamma+gamma_l*l) 
    cost = 260*EV_cost*(1-b)
    return cost

def gv_travel_cost(route):
    cost_GV = 0 
    try:
        route = eval(route)
    except:
        for i in range(1,len(route)-1):
            cost_GV += a[(route[i],route[0])]*GV_cost*q[route[i]] + a[(route[i],route[0])]*GV_cost
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
    q[0] = 0
    if len(route)==3:
        return a[(0,route[0])]* GV_cost + a[(0,route[0])] * q[route[1]] * GV_cost, route
    if len(route)<=5:
        cost = 0
        l = 0
        for i in range(0,len(route)-1):
            l = l + q[route[i]]
            cost += a[(route[i],route[i+1])] * l * GV_cost * 1
        return cost, route
    else:
        nodes = set(route)
        depot = 0

        # Create the model
        model = Model("TSP")

        # Create decision variables
        x = {}
        for i in nodes:
            for j in nodes:
                if i != j:
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # Set the objective function
        model.setObjective(quicksum(a[i, j] * x[i, j] for i in nodes for j in nodes if i != j), GRB.MINIMIZE)

        # Add constraints
        # Each node must be entered exactly once
        for j in nodes:
            model.addConstr(quicksum(x[i, j] for i in nodes if i != j) == 1)

        # Each node must be exited exactly once
        for i in nodes:
            model.addConstr(quicksum(x[i, j] for j in nodes if i != j) == 1)

        # Subtour elimination constraints
        # We use the MTZ (Miller-Tucker-Zemlin) formulation
        u = {}
        for i in nodes:
            if i!=0:
                u[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"u_{i}")

        for i in nodes:
            for j in nodes:
                if i != j and i != depot and j != depot:
                    model.addConstr(u[i] - u[j] + 1 <= (1 - x[i, j]) * len(nodes))

        # Optimize the model
        model.Params.OutputFlag = 0
        model.optimize()

        # Extract the solution
        if model.status == GRB.OPTIMAL:
            tour = []
            current_node = depot
            while True:
                tour.append(current_node)
                next_node = None
                for j in nodes:
                    if j != current_node and x[current_node, j].x > 0.5:
                        next_node = j
                        break
                if next_node is None or next_node == depot:
                    break
                current_node = next_node
            tour.append(depot)
            #print(" -> ".join(map(str, tour)))
            #print(f"Total distance: {model.objVal}")
        else:
            print("No solution found")
        return model.getObjective().getValue()*w_dv, tour