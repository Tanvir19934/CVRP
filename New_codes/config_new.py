import numpy as np
import math
import sys
import time
rnd = np.random
rand_seed = 111
rnd.seed(42)

NODES = 10

num_neighbors = min(round(NODES*0.2),4)
SEARCH_MODE = "heap"
run_dp = False
k = min(round(NODES*0.5),2)
grid_size = 50                                                               #number of clients
time_limit = 3600
xc = np.random.uniform(low=- grid_size/2, high=grid_size/2, size=NODES+1)
yc = np.random.uniform(low=-grid_size/2, high=grid_size/2, size= NODES+1)
xc[0]=0
yc[0]=0
w_dv = 1.2
w_ev = 1
theta = 0.3
tol = 1e-4
N = [i for i in range(1,NODES+1)]                                            #set of customer nodes
V = [0] + N                                                                  #set of all nodes (customer+depot)

# Demands and capacities
max_load = 7
min_load = 1

Q_EV = 10                                                                    #capacity of each EV
Q_GV = Q_EV                                                                  #capacity of each GV
q = {i: rnd.randint(min_load,max_load) for i in N}                                         #demand for customers
total_dem = sum(q.values())                                                          #total demand
mean_dem = total_dem / NODES

#Other parameters
num_EV = math.ceil(NODES*0.3)
unlimited_EV = False
col_dp_cutoff = 1000000000000000

#use_column_heuristic = False
#always_generate_rows = True
use_column_heuristic = False
always_generate_rows = False
#use_column_heuristic = True
#always_generate_rows = False

plot_enabled = 0

if unlimited_EV:
    num_EV = NODES
num_clusters = num_EV #int(0.5*(total_dem/(num_EV*Q_EV)))
num_GV = len(N)
num_TV = num_EV+num_GV
K = [i for i in range(1,num_TV+1)]                                           #Set of all vehicles 
D = [i for i in range(1,num_GV+1)]                                           #Set of diesel vehicles
E = [i for i in range(num_GV+1,num_TV+1)]                                    #Set of EVs
A = [(i,j) for i in V for j in V  if i!=j]                                   #set of arcs in the network
a = {(i,j): np.hypot(xc[i]-xc[j], yc[i]- yc[j]) for (i,j) in A}              #eucledian distance


a[(0,0)] = 0
r = {i: 1 if i == 0 else 0 for i in V}                                       #recharge indicator for EVs at the depot
st = {i: rnd.randint(20,40) for i in N}                                      #service time at customer nodes
st[0]=0
MIP_start = 0

# Battery, speed, and time parameters
battery_tech = 1
gamma = (0.133/60) * battery_tech       #0.133                               #battery depletion rate for EVs without any load (0.133 per hour)
gamma_l =  (0.026/60) * battery_tech    #0.026                               #load dependent battery depletion rate for EVs   (0.026 per hour per ton)
b = [(i,j) for i in V for j in E]                                            #battery level upon arriving at node j
T_max_EV = 6800000/60                                                               #max operation time per EV 
T_max_GV = 6800000/60                                                              #max operation time per GV
EV_velocity = 0.67          # miles per minute (40 mph)
GV_velocity = 0.67        # miles per minute (40 mph)


EV_cost = 3.5       #$/kWh
GV_cost = 1         #$/(ton⋅mile)


# EV_cost = 2.3112 is on par with GV_cost = 1, i.e., equal cost


battery_threshold = 0.1
alpha = 0.1

arc_set = [(i,j) for i in N for j in N  if i!=j]
dist = {(i,j): np.hypot(xc[i]-xc[j], yc[i]- yc[j]) for (i,j) in arc_set}
best_obj = 0
for i in range(1, NODES+1):
    best_obj+= 2*w_dv*a[(0,i)]
best_obj = best_obj * 1.1 #just to be safe




gamma = (0.133/60) * battery_tech       #0.133                               #battery depletion rate for EVs without any load (0.133 per hour)
gamma_l =  (0.026/60) * battery_tech    #0.026                               #load dependent battery depletion rate for EVs   (0.026 per hour per ton)
n = NODES
allow_multi_trip = False



PARAMS = {
    "NODES": int,
    "EV_cost": float,
    "battery_tech": float,
    "Q_EV": int,
    "w_ev": float
}

if len(sys.argv) > 2:
    name, val = sys.argv[1], sys.argv[2]
    if name not in PARAMS:
        raise ValueError(f"Unknown parameter: {name}")
    globals()[name] = PARAMS[name](val)
    print(f"Overriding {name} → {globals()[name]}")
else:
    print("Running with default parameters")



"""

for w_ev in 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7; do
    python branch_and_price_coalition_new.py w_ev $w_ev
done

for w_ev in $(seq 0 0.2 3); do
    python branch_and_price_coalition_new.py w_ev $w_ev
done


for Q_EV in 5 8 10 12 15 18 20 25 30; do
    python branch_and_price_coalition_new.py Q_EV $Q_EV
done

for Q_EV in {1..30}; do
    python branch_and_price_coalition_new.py Q_EV $Q_EV
done


0.25 0.45 0.8 0.9 1 1.1 1.225 1.5 1.75 2 2.25 2.5 2.75 3
for EV_cost in 0.8 1.43 2.286 2.57 2.8575 3.15 3.5 4.286 5.51 6.3 7.785 8.75 9.625 10.5; do
    python branch_and_price_coalition_new.py EV_cost $EV_cost
done


for EV_cost in $(seq 0 0.25 10); do
    python branch_and_price_coalition_new.py EV_cost $EV_cost
done



for battery_tech in 0.25 0.5 0.75 0.9 1 1.1 1.25 1.5 1.75; do
    python branch_and_price_coalition_new.py battery_tech $battery_tech
done

for battery_tech in $(seq 0.1 0.1 3); do
    python branch_and_price_coalition_new.py battery_tech $battery_tech
done

for NODES in $(seq 5 5 30); do
    python hsc_ALNS.py NODES $NODES
done

for NODES in $(seq 50 25 225); do
    python hsc_ALNS.py NODES $NODES
done

"""