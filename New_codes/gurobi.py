from gurobipy import Model, GRB

# Step 1: Create model
model = Model("wildcard_example")

# Step 2: Define sets
products = ['P1', 'P2']
sites = ['S1']
time_periods = ['T1', 'T2']
machines = ['M1', 'M2']

# Step 3: Create variable group dictionary
vars = {}

# Define assignment variable: assign[p, s, t, m]
vars["ASSIGN"] = model.addVars(
    products, sites, time_periods, machines,
    lb=0, ub=1, vtype=GRB.CONTINUOUS,
    name="ASSIGN"
)

# Define total load variable per (p, s, t)
vars["TOTAL_LOAD"] = model.addVars(
    products, sites, time_periods,
    lb=0, vtype=GRB.CONTINUOUS,
    name="TOTAL_LOAD"
)

# Step 4: Add constraints using wildcard sum
model.addConstrs(
    (
        vars["TOTAL_LOAD"][p, s, t] ==
        vars["ASSIGN"].sum(p, s, t, '*')  # sum over all machines
        for p in products
        for s in sites
        for t in time_periods
    ),
    name="TotalLoadConstraint"
)

#quicksum equivalent to sum in Python
from gurobipy import quicksum
model.addConstrs(
    (
        vars["TOTAL_LOAD"][p, s, t] ==
        quicksum(vars["ASSIGN"][p, s, t, m] for m in machines)
        for p in products
        for s in sites
        for t in time_periods
    ),
    name="TotalLoadConstraint"
)


print("Model constraints:")
model.update()
for constr in model.getConstrs():
    print(f"Name: {constr.ConstrName}, Expression: {model.getRow(constr)} {constr.Sense} {constr.RHS}")


# Step 5: Set a simple objective
model.setObjective(vars["TOTAL_LOAD"]['P1', 'S1', 0], GRB.MAXIMIZE)

# Step 6: Solve
model.optimize()

# Step 7: Print variable values
for key, var in vars["TOTAL_LOAD"].items():
    print(f"TOTAL_LOAD{key} = {var.X}")

for key, var in vars["ASSIGN"].items():
    print(f"ASSIGN{key} = {var.X}")
