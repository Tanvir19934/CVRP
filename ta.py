from gurobipy import Model, GRB

# Create a new Gurobi model
model = Model("closest_point")
import math

# Add decision variables (x1, x2, x3, x4)
x1 = model.addVar(vtype=GRB.CONTINUOUS, name="x1", lb=0)
x2 = model.addVar(vtype=GRB.CONTINUOUS, name="x2", lb=0)
x3 = model.addVar(vtype=GRB.CONTINUOUS, name="x3", lb=0)
x4 = model.addVar(vtype=GRB.CONTINUOUS, name="x4", lb=0)

# Update the model to integrate new variables
model.update()

# Objective function (Minimize squared distance)
obj = math.sqrt((x1 + 1)**2 + (x2 - 0)**2 + (x3 - 1)**2 + (x4 - 2)**2)
model.setObjective(obj, GRB.MINIMIZE)

# Add constraints
model.addConstr(x1 + 2 * x2 + 3 * x3 + 4 * x4 <= 5, "constraint1")
model.addConstr(x1 >= x2, "constraint2")
model.addConstr(x2 >= x3 + x4, "constraint3")

# Optimize the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution found:")
    print(f"x1: {x1.X}")
    print(f"x2: {x2.X}")
    print(f"x3: {x3.X}")
    print(f"x4: {x4.X}")
else:
    print("No optimal solution found.")
