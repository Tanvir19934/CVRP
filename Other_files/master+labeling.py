from collections import defaultdict
from config import *
import numpy as np
from gurobipy import Model, GRB, quicksum
import copy
import heapq
import itertools


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

def route_feasibility_check(temp_route):
   if type(temp_route)==str:
      temp_route = eval(temp_route)
   battery = 1
   q[0] = 0
   current_load = 0
   op_time = 0
   score = 0
   miles = 0
   for i in range(0,len(temp_route)-1):
      current_load += q[temp_route[i]]
      if current_load > Q_EV:
         return False
      else:
         battery = battery - (a[(temp_route[i],temp_route[i+1])]/EV_velocity)*(gamma+gamma_l*current_load)
         if battery < battery_threshold:
            return False
         else:
            op_time = op_time + (st[temp_route[i]]+(a[(temp_route[i],temp_route[i+1])]/EV_velocity))
         if op_time > T_max_EV:
            return False
   return True  

# Define a label structure
class Label:
    def __init__(self, node, resource_vector, parent=None):
        self.node = node  # Current node
        self.resource_vector = resource_vector  # Resources consumed up to the current node
        self.parent = parent  # Parent label (previous node)
    def __lt__(self, other):
        # Define less than operator for priority queue (heapq), e.g., based on some resource (distance)
        return self.resource_vector[0] < other.resource_vector[0]
    def __repr__(self):
        return f"Label(node={self.node}, resource_vector={self.resource_vector}, parent={self.parent})"

def generate_k_degree_coalition(N, k):
    # Generate all combinations of k nodes
    combinations = list(itertools.combinations(N, k))
    # Generate all possible routes starting and ending at depot 0
    degree_k_coalition = [tuple([0] + list(comb) + [0]) for comb in combinations]
    return degree_k_coalition

degree_2_coalition = generate_k_degree_coalition(N, 2)
degree_2_coalition_final = []
for item in degree_2_coalition:
    if a[item[0],item[1]] + a[item[1],item[2]] + a[item[2],item[0]] > a[item[0],item[2]] + a[item[2],item[1]] + a[item[1],item[0]]:
        degree_2_coalition_final.append(tuple([item[0],item[2],item[1],item[0]]))
    else: degree_2_coalition_final.append(tuple(item))
degree_2_coalition_initial = copy.deepcopy(degree_2_coalition_final)
degree_2_coalition=[]
for item in degree_2_coalition_initial:
    if route_feasibility_check(item):
        degree_2_coalition.append(item)
degree_2_coalition_cost = {}
for item in degree_2_coalition:
    route = copy.deepcopy(item)
    cost, rev = ev_travel_cost(route)
    if rev:
        degree_2_coalition_cost[item[::-1]] = cost
    else: degree_2_coalition_cost[item] = cost



#SETS and PARAMETERS
r_set = [[0, node, 0] for node in V if node != 0]
#r_set[0].insert(2,6)
#r_set.remove([0,6,0])
#r_set = [[0,2,4,0],[0,10,0],[0,7,0],[0,1,8,0],[0,9,5,0],[0,3,0],[0,6,0]]
r_set = r_set + list(degree_2_coalition) #+ list(degree_3_coalition) #+list(degree_4_coalition)+list(degree_5_coalition)
N.extend(['s','t'])
new_routes_record = [0,0,0,0,0]

iteration = 0
while True:

    ###########Gurobi model############
    mdl = Model('master_problem')

    r_set = set(tuple(route) for route in r_set)

    N.remove('s')
    N.remove('t')
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


    def get_vars(item,opt_route):
       vars = [var for var in opt_route.getVars() if f"{item}" in var.VarName]
       names = opt_route.getAttr('VarName', vars)
       values = opt_route.getAttr('X', vars)
       return dict(zip(names, values))
    y_r_result = get_vars('y_r',mdl)

    for item in y_r_result:
        if y_r_result[item]>0:
            print(f"{item}={y_r_result[item]}")




########################################### labeling / pricing subproblem ############################################
########################################### labeling / pricing subproblem ############################################
########################################### labeling / pricing subproblem ############################################
########################################### labeling / pricing subproblem ############################################
########################################### labeling / pricing subproblem ############################################



    current_routes = []
    for item in y_r_result:
        if y_r_result[item]>0:
            current_routes.append(list(eval(item.split('[')[1][:-1])))

    q[0]=0
    arcs ={item: a[item] for item in A}
    N.extend(['s','t'])


    def feasibility_check(curr_node, extending_node, curr_time = 0, curr_load = 0, curr_battery = 0, curr_distance = 0):

        #new_distance, new_load, new_battery, new_time

        if curr_node == 's':
            curr_node=0
        if extending_node == 't':
            extending_node=0
        new_load = curr_load + q[extending_node]
        if new_load>Q_EV:
            return None, None, None, None
        else:
            new_battery = curr_battery + (a[(curr_node,extending_node)]/EV_velocity)*(gamma+gamma_l*curr_load)
            if new_battery + (a[(0,extending_node)]/EV_velocity)*(gamma+gamma_l*new_load) > 1 - battery_threshold:
                return None, None, None, None
            else:
                new_time = curr_time + (st[curr_node]+(a[(curr_node,extending_node)]/EV_velocity))
                if new_time + (st[extending_node]+(a[(0,extending_node)]/EV_velocity)) > T_max_EV:
                    return None, None, None, None
        new_distance = curr_distance + a[(curr_node,extending_node)]

        return new_time, new_load, new_battery, new_distance 
    
    def calculate_reduced_cost(label):
        label_copy = copy.deepcopy(label)
        distance = label.resource_vector[0]
        reduced_cost = distance
        path = []
        while label:
            path.append(label.node)
            reduced_cost -= dual_values.get(label.node,0)
            label = label.parent     
        path = path[::-1]
        if reduced_cost<0:
            return True
        else: return False
    
    def partial_path(label,current_node):

        node = current_node
        path = []
        while label:
            path.append(label.node)
            label = label.parent
        if node in path[1:]:
            return True
        else: return False

    # Initialize the sets of labels
    U = []  # Priority queue for undominated labels
    L = defaultdict(list)  # Dictionary to store the sets of labels at each node

    # Step 1: Initialize with the starting node
    start_node = 's'
    initial_resource_vector = (0, 0, 0, 0)  # (distance, load, battery, time)
    initial_label = Label(start_node, initial_resource_vector, None)
    heapq.heappush(U, initial_label)
    
    # Step 2: Main loop for label setting
    while U:
        # 2a. Remove first label (label with the least resource cost in heap)
        current_label = heapq.heappop(U)
        current_node = current_label.node
        # 2c. Check for dominance and add label to the set of labels if not dominated
        is_dominated = False
        for label in L[current_node]:
            if current_label.resource_vector[0]>label.resource_vector[0] and \
            current_label.resource_vector[1]<label.resource_vector[1] and \
            current_label.resource_vector[2]>label.resource_vector[2] and \
            current_label.resource_vector[3]>label.resource_vector[3]:
                is_dominated = True
                break
            
        if not is_dominated and current_label not in L[current_node]:
                in_path_already = False
                in_path_already = partial_path(current_label,current_node)
                if in_path_already:
                    continue
                L[current_node].append(current_label)
                # 2c2. Extend the label along all arcs leaving the current node
                neigh = list(set(N)-set([current_node]))
                if current_node=='s':
                    neigh.remove('t')
                if current_node=='t':
                    continue
                
                for new_node in neigh:

                    if (new_node!='s' and current_node!='t'):
                        new_time, new_load, new_battery, new_distance   = feasibility_check(current_node, new_node, current_label.resource_vector[-1], current_label.resource_vector[1], current_label.resource_vector[2], current_label.resource_vector[0])
                    if new_node==2 and current_node==10 and new_distance:
                        pass
                        reduced_cost = False
                    if new_distance:
                        resource_vector = (new_distance, new_load, new_battery, new_time)
                        #new_resource_vector = tuple(map(sum, zip(current_label.resource_vector, resources)))
                        new_label = Label(new_node, resource_vector, current_label)
                        reduced_cost = calculate_reduced_cost(new_label)
                        if reduced_cost:
                            # 2c3. Add all feasible extensions to U (if no constraint violation)
                            heapq.heappush(U, new_label)

    # Step 3: Select the best label in L_t (sink node)
    sink_node = 't'
    best_label = min(L[sink_node], key=lambda x: x.resource_vector[0]) if L[sink_node] else None
    # Output the path corresponding to the best label

    def reconstruct_path(label):
        path = []
        while label:
            path.append(label.node)
            label = label.parent
        return path[::-1]

    #if best_label:
    #    best_path = reconstruct_path(best_label)
    #    print("Best Path:", best_path)
    #    print("Resource Vector:", best_label.resource_vector)
    #else:
    #    print("No feasible path found.")

    new_routes = []
    for item in L[sink_node]:
        new_routes.append(reconstruct_path(item))
    for item in new_routes:
        item[0]=0
        item[-1]=0

    def calculate_route_distance(route, distances):
        """Calculate the total distance of a given route."""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distances.get((route[i], route[i + 1]), float('inf'))
        return total_distance

    def find_best_routes(routes, distances):
        unique_routes = {}
        for route in routes:
            # Remove the start and end (0), work with middle nodes
            middle_nodes = tuple(sorted(route[1:-1]))

            # Generate all permutations of the middle nodes
            min_distance = float('inf')
            best_permutation = None

            for perm in itertools.permutations(middle_nodes):
                # Form the full route with current permutation
                full_route = [0] + list(perm) + [0]
                distance = calculate_route_distance(full_route, distances)

                if distance < min_distance:
                    min_distance = distance
                    best_permutation = full_route

            # Store the best route for the unique set of middle nodes
            if middle_nodes not in unique_routes or min_distance < unique_routes[middle_nodes][1]:
                unique_routes[middle_nodes] = (best_permutation, min_distance)

        # Extract and return the list of best routes
        return [(route, distance) for route, distance in unique_routes.values()]

    # Finding the best routes
    best_routes = find_best_routes(new_routes, a)

    # Print the result
    #for route, distance in best_routes:
    #    print(f"Route: {route}, Distance: {distance}")

    new_routes_to_add = [i[0] for i in best_routes]

    new_routes_record.append(new_routes_to_add)

    for item in new_routes_to_add: 
        r_set.add(tuple(item))
    
    print(dual_values)
    print(sum(dual_values.values()))

    iteration+=1

    def check_values(d):
        for key, value in d.items():
            if value != 0 and value!=1:
                return False
        return True

    if check_values(y_r_result):
        print("All non-zero values are 1, breaking the loop.")
        break

    if new_routes_record[-1]==new_routes_record[-2]==new_routes_record[-3]==new_routes_record[-4]==new_routes_record[-5]==new_routes_record[-6]:
        break

    if not new_routes_to_add:
        break
print(f"iteration count: {iteration}")
