import numpy as np
import math
import sys

rnd = np.random
rand_seed = 111
rnd.seed(42)


# ============================================================
# BASE / DEFAULT PARAMETERS
# ============================================================

NODES = 20

SEARCH_MODE = "heap"
run_dp = False

grid_size = 50

w_dv = 1.2
w_ev = 1
theta = 0.3
tol = 1e-4

# Demands and capacities
max_load = 7
min_load = 1

Q_EV = 10
Q_GV = Q_EV

# Other parameters
unlimited_EV = False
col_dp_cutoff = 1000000000000000

use_column_heuristic = False
always_generate_rows = False

plot_enabled = 0

MIP_start = 0

# Battery / vehicle parameters
battery_tech = 1

T_max_EV = 6800000 / 60
T_max_GV = 6800000 / 60

EV_velocity = 0.67    # miles per minute (40 mph)
GV_velocity = 0.67    # miles per minute (40 mph)

EV_cost = 3.5         # $/kWh
GV_cost = 1           # $/(ton*mile)

battery_threshold = 0.1
alpha = 0.1

allow_multi_trip = False


# ============================================================
# COMMAND-LINE PARAMETER OVERRIDES
#
# Example:
# python branch_and_price_coalition_new.py NODES 10
# python branch_and_price_coalition_new.py EV_cost 2.5
# ============================================================

PARAMS = {
    "NODES": int,
    "EV_cost": float,
    "battery_tech": float,
    "Q_EV": int,
    "w_ev": float,
}


if len(sys.argv) > 2:

    name = sys.argv[1]
    val = sys.argv[2]

    if name not in PARAMS:
        raise ValueError(
            f"Unknown parameter: {name}. "
            f"Available parameters: {list(PARAMS.keys())}"
        )

    globals()[name] = PARAMS[name](val)

    print(f"Overriding {name} -> {globals()[name]}")

elif len(sys.argv) == 2:

    raise ValueError(
        "Parameter name supplied without a value.\n"
        "Example: python branch_and_price_coalition_new.py NODES 10"
    )

else:

    print("Running with default parameters")


# ============================================================
# DERIVED PARAMETERS
#
# IMPORTANT:
# Everything below this point is calculated AFTER overrides.
# ============================================================

num_neighbors = min(round(NODES * 0.2), 4)

k = min(round(NODES * 0.5), 2)


# ============================================================
# NODE COORDINATES
# ============================================================

xc = np.random.uniform(
    low=-grid_size / 2,
    high=grid_size / 2,
    size=NODES + 1
)

yc = np.random.uniform(
    low=-grid_size / 2,
    high=grid_size / 2,
    size=NODES + 1
)

# Depot at origin
xc[0] = 0
yc[0] = 0


# ============================================================
# NODE SETS
# ============================================================

N = [i for i in range(1, NODES + 1)]

V = [0] + N


# ============================================================
# DEMANDS
# ============================================================

q = {
    i: rnd.randint(min_load, max_load)
    for i in N
}

total_dem = sum(q.values())

mean_dem = total_dem / NODES


# ============================================================
# VEHICLE SETS
# ============================================================

num_EV = math.ceil(NODES * 0.3)

if unlimited_EV:
    num_EV = NODES

num_clusters = num_EV

num_GV = len(N)

num_TV = num_EV + num_GV


# All vehicles
K = [
    i for i in range(1, num_TV + 1)
]

# Diesel vehicles
D = [
    i for i in range(1, num_GV + 1)
]

# EVs
E = [
    i for i in range(num_GV + 1, num_TV + 1)
]


# ============================================================
# ARCS AND DISTANCES
# ============================================================

A = [
    (i, j)
    for i in V
    for j in V
    if i != j
]

a = {
    (i, j): np.hypot(
        xc[i] - xc[j],
        yc[i] - yc[j]
    )
    for (i, j) in A
}

a[(0, 0)] = 0


# Customer-to-customer arcs
arc_set = [
    (i, j)
    for i in N
    for j in N
    if i != j
]

dist = {
    (i, j): np.hypot(
        xc[i] - xc[j],
        yc[i] - yc[j]
    )
    for (i, j) in arc_set
}


# ============================================================
# RECHARGING / SERVICE PARAMETERS
# ============================================================

r = {
    i: 1 if i == 0 else 0
    for i in V
}

st = {
    i: rnd.randint(20, 40)
    for i in N
}

st[0] = 0


# ============================================================
# BATTERY PARAMETERS
# ============================================================

gamma = (0.133 / 60) * battery_tech

gamma_l = (0.026 / 60) * battery_tech


# Battery level upon arriving at node j
b = [
    (i, j)
    for i in V
    for j in E
]


# ============================================================
# INITIAL / UPPER BOUND OBJECTIVE
# ============================================================

best_obj = 0

for i in range(1, NODES + 1):
    best_obj += 2 * w_dv * a[(0, i)]

# Just to be safe
best_obj *= 1.1


# ============================================================
# OTHER DERIVED VARIABLES
# ============================================================

n = NODES


# ============================================================
# OPTIONAL DEBUG PRINT
# ============================================================

print(
    f"Instance configuration: "
    f"NODES={NODES}, "
    f"Q_EV={Q_EV}, "
    f"EV_cost={EV_cost}, "
    f"battery_tech={battery_tech}, "
    f"w_ev={w_ev}, "
    f"num_EV={num_EV}, "
    f"num_GV={num_GV}"
)


"""
============================================================
EXAMPLE TERMINAL LOOPS
============================================================


# ----------------------------------------------------------
# w_ev
# ----------------------------------------------------------

for w_ev in 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7; do
    python branch_and_price_coalition_new.py w_ev $w_ev
done


for w_ev in $(seq 0 0.2 3); do
    python branch_and_price_coalition_new.py w_ev $w_ev
done


# ----------------------------------------------------------
# Q_EV
# ----------------------------------------------------------

for Q_EV in 5 8 10 12 15 18 20 25 30; do
    python branch_and_price_coalition_new.py Q_EV $Q_EV
done


for Q_EV in {1..30}; do
    python branch_and_price_coalition_new.py Q_EV $Q_EV
done


# ----------------------------------------------------------
# EV_cost
# ----------------------------------------------------------

for EV_cost in 0.8 1.43 2.286 2.57 2.8575 3.15 3.5 4.286 5.51 6.3 7.785 8.75 9.625 10.5; do
    python branch_and_price_coalition_new.py EV_cost $EV_cost
done


for EV_cost in $(seq 0 0.25 10); do
    python branch_and_price_coalition_new.py EV_cost $EV_cost
done


# ----------------------------------------------------------
# battery_tech
# ----------------------------------------------------------

for battery_tech in 0.25 0.5 0.75 0.9 1 1.1 1.25 1.5 1.75; do
    python branch_and_price_coalition_new.py battery_tech $battery_tech
done


for battery_tech in $(seq 0.1 0.1 3); do
    python branch_and_price_coalition_new.py battery_tech $battery_tech
done


# ----------------------------------------------------------
# NODES
# ----------------------------------------------------------

for NODES in $(seq 5 5 30); do
    python New_codes/branch_and_price_coalition_new.py NODES $NODES
done


for NODES in $(seq 50 25 225); do
    python New_codes/branch_and_price_coalition_new.py NODES $NODES
done

"""