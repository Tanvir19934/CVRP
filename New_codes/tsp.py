from gurobipy import Model, GRB, quicksum

# Define the nodes and the depot
nodes = [0, 3, 2, 1, 6, 7]
depot = 0

# Define the pairwise distances as a dictionary
distances = {
    (0, 3): 3, (3, 0): 3,
    (0, 2): 4, (2, 0): 4,
    (0, 1): 1, (1, 0): 1,
    (0, 6): 6, (6, 0): 6,
    (0, 7): 7, (7, 0): 7,
    (3, 2): 2, (2, 3): 2,
    (3, 1): 5, (1, 3): 5,
    (3, 6): 8, (6, 3): 8,
    (3, 7): 9, (7, 3): 9,
    (2, 1): 3, (1, 2): 3,
    (2, 6): 7, (6, 2): 7,
    (2, 7): 8, (7, 2): 8,
    (1, 6): 4, (6, 1): 4,
    (1, 7): 5, (7, 1): 5,
    (6, 7): 1, (7, 6): 1
}

# Create the model
model = Model("TSP")

# Create decision variables
x = {}
for i in nodes:
    for j in nodes:
        if i != j:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# Set the objective function
model.setObjective(quicksum(distances[i, j] * x[i, j] for i in nodes for j in nodes if i != j), GRB.MINIMIZE)

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
    u[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"u_{i}")

for i in nodes:
    for j in nodes:
        if i != j and i != depot and j != depot:
            model.addConstr(u[i] - u[j] + len(nodes) * x[i, j] <= len(nodes) - 1)

# Optimize the model
model.optimize()

# Extract the solution
if model.status == GRB.OPTIMAL:
    print("Optimal tour:")
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
    #print("No solution found")
    pass

