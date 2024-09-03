from collections import defaultdict
from config import *
import heapq
import pickle
import copy
from itertools import permutations

objective='miles'
# Load the dictionary from the file
with open('dual_values.pkl', 'rb') as f:
    dual_values = pickle.load(f)
#with open('y_r_result.pkl', 'rb') as file:
#    y_r_result = pickle.load(file)
#current_routes = []
#for item in y_r_result:
#    if y_r_result[item]>0:
#        current_routes.append(list(eval(item.split('[')[1][:-1])))


q[0]=0
arcs ={item: a[item] for item in A}
N.extend(['s','t'])
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
def partial_path(label, current_node):
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
    print(len(U))
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
            in_path_already = partial_path(current_label, current_node)
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

if best_label:
    best_path = reconstruct_path(best_label)
    print("Best Path:", best_path)
    print("Resource Vector:", best_label.resource_vector)
else:
    print("No feasible path found.")

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
        
        for perm in permutations(middle_nodes):
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

with open('new_routes_to_add.pkl', 'wb') as file:
    pickle.dump(new_routes_to_add, file)