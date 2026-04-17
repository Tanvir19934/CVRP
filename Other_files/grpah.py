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
n = 10                                                                       #number of clients
xc = rnd.rand(n+1)*200
yc = rnd.rand(n+1)*100
#plt.plot(xc[0],yc[0], c = 'r', marker = 's')                                #depot
#plt.scatter(xc[1:], yc[1:], c = 'b')                                        #clients
#for i, j in zip(xc, yc):
#   plt.text(i, j+1, '({}, {})'.format(i, j))

N = [i for i in range(1,n+1)]                                               #set of customer nodes
V = [0] + N                                                                 #set of all nodes (customer+depot)
Q = 10                                                                      #capacity of each vehicle
q = {i: rnd.randint(1,10) for i in N}                                       #demand for customers
total_dem = sum(q)                                                          #total demand
num_EV = 1                
num_GV = 1
num_TV = num_EV+num_GV
K = [i for i in range(1,num_TV+1)]                                          #Set of all vehicles 
D = [i for i in range(1,num_GV+1)]                                          #Set of diesel vehicles
E = [i for i in range(num_GV+1,num_TV+1)]                                   #Set of EVs
A = [(i,j) for i in V for j in V  if i!=j]                                  #set of arcs in the network
c = {(i,j): np.hypot(xc[i]-xc[j], yc[i]-yc[j]) for (i,j) in A}              #eucledian distance
st = {i: rnd.randint(20,40) for i in N}                                     #service time at customer nodes
st[0]=0
gamma = 0.2                                                                   #battery depletion rate for EVs without any load
gamma_l = 0.2                                                                 #load dependent battery depletion rate for EVs
b = [(i,j) for i in V for j in E]                                           #battery level upon arriving at node i
T_max_EV = 533                                                                #max operation time per vehicle 
T_max_GV = 500
#r = [i for i in E]                                                         #recharge indicator for EVs at the depot




mdl = Model('hsc')
x_e = {}
for item in E:
    for element in A:
        x_e[item,element] = mdl.addVar(vtype=GRB.BINARY, name=f"x_e[{item},{element}]")

#eliminate some x_e variables based on range and distance (i,j)


x_d = mdl.addVars(((item, element) for item in D for element in A), vtype = GRB.BINARY, name = "x_d")
#x_k = mdl.addVars(((item, element) for item in K for element in A), vtype = GRB.BINARY, name = "x_k")
r = mdl.addVars(E, vtype = GRB.BINARY, name = "r")            #recharge indicator for EVs at the depot
z = mdl.addVars(((item, element) for item in K for element in V), vtype = GRB.CONTINUOUS, name = "z")
l = mdl.addVars(((item, element) for item in K for element in V), vtype = GRB.CONTINUOUS, name = "l")
b = mdl.addVars(((item, element) for item in E for element in V), vtype=GRB.CONTINUOUS, name = "b")



#mdl.addConstrs((quicksum(x_d[d,(i,j)] for d in D for i in V if i!=j) + (quicksum(x_e[e,(i,j)] for e in E for i in V if i!=j))== 1) for j in V if j!=0)
mdl.addConstrs((quicksum(x_d[d,(0,j)] for d in D) + (quicksum(x_e[e,(i,j)] for e in E for i in V if i!=j))== 1) for j in V if j!=0)

#mdl.addConstrs((quicksum(x_d[d,(i,j)] for i in V if i!=j)-quicksum(x_d[d,(j,i)] for i in V if i!=j) == 0) for j in V for d in D)
mdl.addConstrs(((x_d[d,(0,j)])-(x_d[d,(j,0)]) == 0) for j in V for d in D if j!=0)

mdl.addConstrs((quicksum(x_d[d,(i,j)] for i in N for j in N if i!=j)==0) for d in D)

mdl.addConstrs((quicksum(x_e[e,(i,j)] for i in V if i!=j)-quicksum(x_e[e,(j,i)] for i in V if i!=j) == 0) for j in V for e in E)

#mdl.addConstrs((quicksum(x_e[e,(0,j)] for j in N) == 1) for e in E)


#mdl.addConstrs((l[(e,j)]  <= Q) for j in N  for e in E)
mdl.addConstrs((quicksum(q[j]*x_e[e,(i,j)] for i in V for j in N if i!=j) <= Q) for e in E)
mdl.addConstrs((q[j]*x_d[d,(0,j)] <= Q) for d in D for j in N)


mdl.addConstrs((l[(e,j)] >=  x_e[e,(i,j)]*(l[(e,i)]+q[j])) for e in E for j in N for i in V if i!=j)
mdl.addConstrs((l[(d,j)] >=  x_d[d,(i,j)]*(l[(d,i)]+q[j])) for d in D for j in N for i in V if i!=j)
mdl.addConstrs((l[(k,0)] == 0) for k in K)
#mdl.addConstr((quicksum(l[(k,j)] for j in V for k in K) == sum(q.values())))





#mdl.addConstrs((b[e,j] == x_e[e,(i,j)]*((1-r[e])*b[e,i]+r[e]-c[i,j]*gamma*l[(e,j)])+b[e,i]*(1-x_e[e,(i,j)])) for j in N for i in V for e in E if i!=j)
mdl.addConstrs((b[e,j] >= x_e[e,(i,j)] * (b[e,i]-c[i,j]*(gamma+gamma_l*l[(e,j)]))) for i in V for j in N for e in E if i!=j)
mdl.addConstrs((b[e,j] >= x_e[e,(i,j)]*c[j,0]*(gamma+gamma_l*l[(e,j)])) for i in V for j in N for e in E if i!=j)
mdl.addConstrs((b[e,0] == 1) for e in E)

mdl.addConstrs((z[d,j] >= x_d[d,(i,j)]*(z[d,i]+st[i]+c[i,j])) for i in V for j in N for d in D if i!=j)
mdl.addConstrs((z[e,j] >= x_e[e,(i,j)]*(z[e,i]+st[i]+c[i,j])) for i in V for j in N for e in E if i!=j)
mdl.addConstrs((z[d,0] == 0) for d in D)
mdl.addConstrs((z[e,0] == 0) for e in E)
mdl.addConstrs((quicksum(z[d,j] for j in V)<= T_max_GV) for d in D)


#h = mdl.addVars((item for item in E), vtype=GRB.CONTINUOUS, name = "h")
#mdl.addConstrs((h[e]==gp.max_((z[e,j]) for j in V)) for e in E)
#mdl.addConstrs((h[e] <= T_max) for e in E)
#mdl.addConstrs((z[e,i] -z[e,0]<= T_max_EV) for e in E for i in V  if i!=0)
mdl.addConstrs((quicksum(x_e[e,(j,0)]*z[e,j] for j in V if j!=0) <= T_max_EV) for e in E)


mdl.modelSense = GRB.MINIMIZE
mdl.setObjective((quicksum(x_d[d,(0,j)]*c[(0,j)]*100 for d in D for j in V if j!=0))+(quicksum(x_d[d,(j,0)]*c[(j,0)]*100 for j in V for d in D if j!=0))+(quicksum(x_e[e,(i,j)]*c[(i,j)] for i in V for j in V for e in E if i!=j)))

mdl.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/hsc.lp")





mdl
mdl.Params.MIPGap = 0.01
#mdl.params.NonConvex = 2

mdl.Params.TimeLimit = 1000 #seconds
mdl.optimize()
#for v in mdl.getVars():
#    print(f"{v.VarName} = {v.X}")
""" try:
    mdl.computeIIS()
    mdl.write("model.ilp")
except: 2 """
x_d_vars = [var for var in mdl.getVars() if "x_d" in var.VarName]
names = mdl.getAttr('VarName', x_d_vars)
values = mdl.getAttr('X', x_d_vars)
x_d_result = dict(zip(names, values))

x_e_vars = [var for var in mdl.getVars() if "x_e" in var.VarName]
names = mdl.getAttr('VarName', x_e_vars)
values = mdl.getAttr('X', x_e_vars)
x_e_result = dict(zip(names, values))

z_vars = [var for var in mdl.getVars() if "z" in var.VarName]
names = mdl.getAttr('VarName', z_vars)
values = mdl.getAttr('X', z_vars)
z_result = dict(zip(names, values))

b_vars = [var for var in mdl.getVars() if "b" in var.VarName]
names = mdl.getAttr('VarName', b_vars)
values = mdl.getAttr('X', b_vars)
b_result = dict(zip(names, values))

l_vars = [var for var in mdl.getVars() if "l" in var.VarName]
names = mdl.getAttr('VarName', l_vars)
values = mdl.getAttr('X', l_vars)
l_result = dict(zip(names, values))


#plt.plot(xc[0],yc[0], c = 'r', marker = 's')                                #depot
#plt.scatter(xc[1:], yc[1:], c = 'b')                                        #clients
#for i, j in zip(xc, yc):
#   plt.text(i, j+1, '({}, {})'.format(i, j))



G = nx.DiGraph(directed=True)
label_pos_dict={}
pos_dict = {}
offset = 5
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
      nx.draw_networkx_labels(G, pos=label_pos_dict, labels={temp[2]: int(z_result[f"z[{temp[0]},{temp[2]}]"])}, font_color='black', font_weight='bold',font_size=8)
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
