import numpy as np
import math
rnd = np.random
rand_seed = 111
rnd.seed(42)


NODES = 25
SEARCH_MODE = "heap"
k = min(round(NODES*0.5),2)
grid_size = 50                                                               #number of clients
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
Q_EV = 10                                                                    #capacity of each EV
Q_GV = 10                                                                    #capacity of each GV
max_load = 7
min_load = 1
q = {i: rnd.randint(min_load,max_load) for i in N}                                         #demand for customers
total_dem = sum(q)                                                           #total demand

#Other parameters
num_EV = math.ceil(NODES*0.3)
unlimited_EV = False
col_dp_cutoff = 1000

#use_column_heuristic = False
#always_generate_rows = True
#use_column_heuristic = False
#always_generate_rows = False
use_column_heuristic = True
always_generate_rows = False

dom_heuristic = False
plot_enabled = 0

if unlimited_EV:
    num_EV = NODES
num_clusters = int(0.5*(total_dem/(num_EV*Q_EV)))
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
time_limit = 6000
MIP_start = 0

# Battery, speed, and time parameters
gamma = 0.133/60    #0.133                                                   #battery depletion rate for EVs without any load (0.133 per hour)
gamma_l =  0.026/60 #0.026                                                   #load dependent battery depletion rate for EVs   (0.026 per hour per ton)
b = [(i,j) for i in V for j in E]                                            #battery level upon arriving at node j
T_max_EV = 68000                                                               #max operation time per EV 
T_max_GV = 68000                                                              #max operation time per GV
EV_velocity = 0.67 
GV_velocity = 0.67
EV_cost = 14.5 # 0.3    #
#GV_cost = 0.58  # 
GV_cost = 5   

EV_cost = 3.5
GV_cost = 1


# EV_cost = 2.3112 is on par with GV_cost = 1, i.e., equal cost


battery_threshold = 0.1
alpha = 0.1

arc_set = [(i,j) for i in N for j in N  if i!=j]
dist = {(i,j): np.hypot(xc[i]-xc[j], yc[i]- yc[j]) for (i,j) in arc_set}
best_obj = 0
for i in range(1, NODES+1):
    best_obj+= 2*w_dv*a[(0,i)]
best_obj = best_obj * 1.1 #just to be safe