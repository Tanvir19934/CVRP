import numpy as np
rnd = np.random
rnd.seed(10)

# Grid and coordinates
n = 7
grid_size = 50                                                               #number of clients
xc = np.random.uniform(low=- grid_size/2, high=grid_size/2, size=n+1)
yc = np.random.uniform(low=-grid_size/2, high=grid_size/2, size=n+1)
xc[0]=0
yc[0]=0

#nodes
N = [i for i in range(1,n+1)]                                                #set of customer nodes
V = [0] + N                                                                  #set of all nodes (customer+depot)

# Demands and capacities
Q_EV = 10                                                                    #capacity of each EV
Q_GV = 15                                                                    #capacity of each GV
q = {i: rnd.randint(1,7) for i in N}                                         #demand for customers

#q = {i: np.random.choice([1, 20]) for i in N}
total_dem = sum(q)                                                           #total demand

#Other parameters
num_EV = int(n*0.2) 
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
time_limit = 60
MIP_start = 1

# Battery, speed, and time parameters
gamma = 0.133/60    #0.133                                                   #battery depletion rate for EVs without any load (0.133 per hour)
gamma_l =  0.026/60 #0.026                                                   #load dependent battery depletion rate for EVs   (0.026 per hour per ton)
b = [(i,j) for i in V for j in E]                                            #battery level upon arriving at node j
T_max_EV = 600                                                               #max operation time per EV 
T_max_GV = 600                                                               #max operation time per GV
EV_velocity = 0.67
GV_velocity = 0.67
GV_cost = 0.58 #4.5/6.5  #0.58 #0.25 per ton mile
EV_cost = 0.38 # 0.3    #0.38  #$/kWh   or 0.035 per ton mile
M = 4
battery_threshold = 0.0
alpha = 0.1

arc_set = [(i,j) for i in N for j in N  if i!=j]
dist = {(i,j): np.hypot(xc[i]-xc[j], yc[i]- yc[j]) for (i,j) in arc_set}      #eucledian distance
