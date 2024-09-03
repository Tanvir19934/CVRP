import numpy as np
from matplotlib import pyplot as plt
import gurobipy as gp
import networkx as nx
from gurobipy import Model, GRB, quicksum
import copy
import time
import re
from matplotlib.lines import Line2D
import random


rnd = np.random
rnd.seed(12)
n = 10                                                                        #number of clients
xc = rnd.rand(n+1)*50
yc = rnd.rand(n+1)*50

N = [i for i in range(1,n+1)]                                                #set of customer nodes
V = [0] + N                                                                  #set of all nodes (customer+depot)
Q_EV = 10                                                                    #capacity of each EV
Q_GV = 15                                                                    #capacity of each GV
q = {i: rnd.randint(1,10) for i in N}                                        #demand for customers
total_dem = sum(q)                                                           #total demand
num_EV = 2                
num_GV = len(N)
num_TV = num_EV+num_GV 
K = [i for i in range(1,num_TV+1)]                                           #Set of all vehicles 
D = [i for i in range(1,num_GV+1)]                                           #Set of diesel vehicles
E = [i for i in range(num_GV+1,num_TV+1)]                                    #Set of EVs
A = [(i,j) for i in V for j in V  if i!=j]                                   #set of arcs in the network
a = {(i,j): np.hypot(xc[i]-xc[j], yc[i]-yc[j]) for (i,j) in A}               #eucledian distance
r = {i: 1 if i == 0 else 0 for i in V}                                       #recharge indicator for EVs at the depot
st = {i: rnd.randint(20,40) for i in N}                                      #service time at customer nodes
st[0]=0
gamma = 0.133/60    #0.133                                                   #battery depletion rate for EVs without any load (0.133 per hour)
gamma_l =  0.026/60 #0.026                                                   #load dependent battery depletion rate for EVs   (0.026 per hour per ton)
b = [(i,j) for i in V for j in E]                                            #battery level upon arriving at node j
T_max_EV = 600                                                               #max operation time per EV 
T_max_GV = 480                                                               #max operation time per GV
EV_velocity = 0.67
GV_velocity = 0.67
GV_cost = 6.5*4.5
EV_cost = GV_cost/1
M = 4

mdl = Model('hsc')
x_e = {}
for item in E:
    for element in A:
        x_e[item,element] = mdl.addVar(vtype=GRB.BINARY, name=f"x_e[{item},{element}]")

#eliminate some x_e variables based on range and distance (i,j)


x_d = mdl.addVars(((item, element) for item in D for element in A), vtype = GRB.BINARY, name = "x_d")
z = mdl.addVars(((item, element) for item in K for element in V), vtype = GRB.CONTINUOUS, name = "z")
l = mdl.addVars(((item, element) for item in K for element in V), vtype = GRB.INTEGER, name = "l")
b = mdl.addVars(((item, element) for item in E for element in V), vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name = "b")
b0 = mdl.addVars(((item, 0, i) for item in E for i in N), vtype=GRB.CONTINUOUS, name = "b0")
y = mdl.addVars(((item, 0, i) for item in E for i in N), vtype=GRB.CONTINUOUS, name = "y")

ind = mdl.addVars(((item, element) for item in E for element in V), vtype = GRB.BINARY, name = "ind")


mdl.addConstrs((x_d[j,(0,j)] + (quicksum(x_e[e,(i,j)] for e in E for i in V if i!=j))== 1) for j in N)
mdl.addConstrs(((x_d[d,(0,j)])-(x_d[d,(j,0)]) == 0) for d in D for j in N)
mdl.addConstrs(((x_d[j,(0,i)]) == 0) for i in N for j in N if i!=j)
mdl.addConstrs((quicksum(x_d[d,(i,j)] for i in N for j in N if i!=j)==0) for d in D)
mdl.addConstrs(((quicksum(x_e[e,(i,j)] for i in V if i!=j)-quicksum(x_e[e,(j,i)] for i in V if i!=j) == 0) for j in V for e in E), name='s')


mdl.addConstrs((l[(e,i)]  <= Q_EV) for i in N  for e in E)
mdl.addConstrs((q[j]*x_d[d,(0,j)] <= Q_GV) for d in D for j in N)
mdl.addConstrs((l[(e,j)]* x_e[e,(i,j)] ==  x_e[e,(i,j)]*(l[(e,i)]+q[j])) for e in E for j in N for i in V if i!=j)
mdl.addConstrs((l[(d,j)]* x_d[d,(i,j)] ==  x_d[d,(i,j)]*(l[(d,i)]+q[j])) for d in D for j in N for i in V if i!=j)
mdl.addConstrs((l[(k,0)] == 0) for k in K)


mdl.addConstrs(((b[e,j]*x_e[e,(i,j)] == x_e[e,(i,j)] * (b[e,i]*(1-r[i])+r[i]-(a[(i,j)]/EV_velocity)*(gamma+gamma_l*l[(e,j)]))) for i in V for j in N for e in E if i!=j), name='ssss')
mdl.addConstrs((b[e,j]*x_e[e,(i,j)] >= x_e[e,(i,j)]*(a[(j,0)]/EV_velocity)*(gamma+gamma_l*l[(e,j)])) for i in N for j in N for e in E if i!=j)
mdl.addConstrs((b[e,0] == 1) for e in E)
#mdl.addConstrs((b[e,i] >= 0.0) for e in E for i in N)
#mdl.addConstrs((b[e,i] <= 1) for e in E for i in N)


mdl.addConstrs((z[d,j]*x_d[d,(0,j)] == 2*x_d[d,(0,j)]*(st[j]+(a[0,j]/GV_velocity))) for j in N for d in D)

for j in N:
      for e in E:
         mdl.addGenConstrPWL(b0[e,0,j], y[e,0,j], [0, 0.8, 1], [300, 120, 0],  "myPWLConstr")

mdl.addConstrs((z[e,j]*x_e[e,(i,j)] == x_e[e,(i,j)]*(z[e,i]+st[i]+(a[(i,j)]/EV_velocity))) for i in V for j in N for e in E if i!=j)
mdl.addConstrs((b0[e,0,j]*x_e[e,(j,0)] == x_e[e,(j,0)] * (b[e,j] - (a[(j,0)]/EV_velocity)*(gamma+gamma_l*l[(e,j)]))) for j in N for e in E)
mdl.addConstrs((z[d,0] == 0) for d in D)
mdl.addConstrs((z[e,0] == 0) for e in E)
mdl.addConstrs((quicksum(z[d,j]*x_d[d,(0,j)] for j in N)<= T_max_GV) for d in D)


#h = mdl.addVars((item for item in E), vtype=GRB.CONTINUOUS, name = "h")
#mdl.addConstrs((h[e]==gp.max_((z[e,j]) for j in V)) for e in E)
#mdl.addConstrs((h[e] <= T_max) for e in E)
#mdl.addConstrs((z[e,i] -z[e,0]<= T_max_EV) for e in E for i in V  if i!=0)
mdl.addConstrs((quicksum(x_e[e,(j,0)]* (z[e,j]+y[e,0,j]+(a[(j,0)]/EV_velocity)) for j in N) <= T_max_EV) for e in E)


#mdl.addConstrs((quicksum(x_e[e,(0,j)] for j in N) <= (quicksum(x_e[e-1,(0,j)] for j in N)) for e in E if e!=min(E)))       #var_elim constraint. employ smaller labeled EV first
#mdl.addConstrs((quicksum(x_d[d,(0,j)] for j in N) <= (quicksum(x_d[d-1,(0,j)] for j in N)) for d in D if d!=min(D)))      #might be incorrect. var_elim constraint. employ smaller labeled GV first


#mdl.addConstrs(((x_e[e,(i,j)] - b[e,j]) >= 0) for e in E for i in V for j in N  if i!=j)



mdl.modelSense = GRB.MINIMIZE
mdl.setObjective((quicksum(x_d[d,(0,j)]*a[(0,j)]*GV_cost for d in D for j in V if j!=0))+(quicksum(x_d[d,(j,0)]*a[(j,0)]*GV_cost for j in V for d in D if j!=0))+(quicksum(x_e[e,(i,j)]*a[(i,j)]*EV_cost for i in V for j in V for e in E if i!=j))+ 0*quicksum(z[e,j] for e in E for j in N) + 0*quicksum(x_e[e,(0,j)]*e for e in E for j in N))

mdl.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/hsc.lp")

mdl.Params.MIPGap = 0.05
mdl.params.NonConvex = 2
#mdl.Params.TimeLimit = 2000 #seconds
mdl.optimize()

""" try:
    mdl.computeIIS()
    mdl.write("model.ilp")
except: 2 """

def get_vars(item):
   vars = [var for var in mdl.getVars() if f"{item}" in var.VarName]
   names = mdl.getAttr('VarName', vars)
   values = mdl.getAttr('X', vars)
   return dict(zip(names, values))

x_d_result  = get_vars('x_d')
x_e_result = get_vars('x_e')
z_result = get_vars('z')
b_result = get_vars('b')
b0_result = get_vars('b0')
l_result = get_vars('l')
y_result = get_vars('y')


G = nx.DiGraph(directed=True)
label_pos_dict={}
pos_dict = {}
offset = 2
legend_elements_ev = []
legend_elements_gv = []

color_dict = {i: (random.random(), random.random(), random.random()) for i in range(1, num_TV + 1)}

for i in range(len(xc)):
    pos_dict[i] = (xc[i], yc[i])
for i in range(len(xc)):
    label_pos_dict[i] = (xc[i]+offset, yc[i]+offset)

for item in x_e_result:
   if x_e_result[item]>0.99:
      temp = [int(match.group()) for match in re.finditer(r'\b\d+\b', item)]
      G.add_edge(temp[1], temp[2], color=color_dict[temp[0]])
      s = f"l[{temp[0]},{temp[2]}]"
      #nx.draw_networkx_labels(G, pos=label_pos_dict, labels={temp[2]: int(l_result[s])}, font_color='red', font_weight='bold',font_size=8)
      #nx.draw_networkx_labels(G, pos=label_pos_dict, labels={temp[2]: int(z_result[f"z[{temp[0]},{temp[2]}]"])}, font_color='black', font_weight='bold',font_size=8)
      #nx.draw_networkx_labels(G, pos=label_pos_dict, labels={temp[2]: round((y_result[f"y[{temp[0]},({temp[1]}, {temp[2]})]"]),4)}, font_color='red', font_weight='bold',font_size=8)
      nx.draw_networkx_labels(G, pos=label_pos_dict, labels={temp[2]: round((b_result[f"b[{temp[0]},{temp[2]}]"]),3)}, font_color='black', font_weight='bold',font_size=8)

      if color_dict[temp[0]] not in legend_elements_ev:
        legend_elements_ev.append(color_dict[temp[0]])

for item in x_d_result:
   if x_d_result[item]>0.99:
      temp = [int(match.group()) for match in re.finditer(r'\b\d+\b', item)]
      G.add_edge(temp[1], temp[2], color=color_dict[temp[0]])
      s = f"l[{temp[0]},{temp[2]}]"
      #nx.draw_networkx_labels(G, pos=label_pos_dict, labels={temp[2]: int(l_result[s])}, font_color='red', font_weight='bold',font_size=8)
      nx.draw_networkx_labels(G, pos=label_pos_dict, labels={temp[2]: int(z_result[f"z[{temp[0]},{temp[2]}]"])}, font_color='black', font_weight='bold',font_size=8)
      if color_dict[temp[0]] not in legend_elements_gv:
        legend_elements_gv.append(color_dict[temp[0]])

lev = []
lgv = []
for i, j in enumerate(legend_elements_ev):
   lev.append(Line2D([0], [0], color=j, label=f'EV{i+1}'))
for i, j in enumerate(legend_elements_gv):
   lgv.append(Line2D([0], [0], color=j, label=f'GV{i+1}'))

all_legends = lev+lgv   

# Draw the graph
plt.xlim(-10, max(pos_dict.values(), key=lambda x: x[0])[0]+20)  # Adjust the limits based on your node coordinates
plt.ylim(-10, max(pos_dict.values(), key=lambda x: x[1])[1]+20)  # Adjust the limits based on your node coordinates

pos = nx.spring_layout(G)
edge_colors = [data['color'] for _, _, data in G.edges(data=True)]

nx.draw(G, pos=pos_dict, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color=edge_colors)
plt.legend(handles=all_legends, loc='upper right')
plt.show()

2