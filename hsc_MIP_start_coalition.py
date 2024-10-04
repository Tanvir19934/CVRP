import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from gurobipy import Model, GRB, quicksum
import re
from matplotlib.lines import Line2D
import random
import pickle
from config import *
import itertools
import copy
#import subprocess
#import json
#import hsc_ALNS_IFB

rnd = np.random
rnd.seed(10)

#override some config parameters
q[0] = 0
#T_max_EV = 700
#print(T_max_EV)
adjustment = 0.8 #artificially make degree 2 coalition lucrative to get unstable results

###########Gurobi model############
mdl = Model('hsc')

def generate_k_degree_coalition(N, k):
    # Generate all combinations of k nodes
    combinations = list(itertools.combinations(N, k))
    # Generate all possible routes starting and ending at depot 0
    degree_k_coalition = [tuple([0] + list(comb) + [0]) for comb in combinations]
    return degree_k_coalition

degree_2_coalition = generate_k_degree_coalition(N, 2)
degree_2_coalition_final = []
degree_2_coalition_route = []

for item in degree_2_coalition:
    if a[item[0],item[1]] + a[item[1],item[2]] + a[item[2],item[0]] > a[item[0],item[2]] + a[item[2],item[1]] + a[item[1],item[0]]:
        degree_2_coalition_route.append(tuple([item[0],item[2],item[1],item[0]]))
        degree_2_coalition_final.append(tuple([item[2],item[1]]))
    else:
      degree_2_coalition_route.append(tuple([item[0],item[1],item[2],item[0]]))
      degree_2_coalition_final.append(tuple([item[1],item[2]]))

def ev_travel_cost(route):
    rev=False
    b = 1
    l = 0
    for i in range(len(route)-1):
        l+=q[route[i]]
        b = b - (a[route[i],route[i+1]]/EV_velocity)*(gamma+gamma_l*l) 
    clockwise_cost = 260*EV_cost*(1-b)
    b = 1
    l = 0
    route = list(route)
    route.reverse()
    for i in range(len(route)-1):
        l+=q[route[i]]
        b = b - (a[route[i],route[i+1]]/EV_velocity)*(gamma+gamma_l*l) 
    counter_clockwise_cost = 260*EV_cost*(1-b)
    if clockwise_cost>counter_clockwise_cost:
        rev=True
    return min(clockwise_cost,counter_clockwise_cost),rev

degree_2_coalition_cost = {}
for item in degree_2_coalition:
    route = copy.deepcopy(item)
    cost, rev = ev_travel_cost(route)
    if rev:
        degree_2_coalition_cost[tuple([item[2],item[1]])] = cost
    else: degree_2_coalition_cost[tuple([item[1],item[2]])] = cost

standalone_cost_degree_2 = {}
for item in degree_2_coalition_cost:
   standalone_cost_degree_2[item] = {}
for item in degree_2_coalition_cost:
   standalone_cost_degree_2[item][item[0]] = adjustment*degree_2_coalition_cost[item]* (a[item[0],0]*GV_cost+a[item[0],0]*GV_cost*q[item[0]])/((a[item[0],0]*GV_cost+a[item[0],0]*GV_cost*q[item[0]])+(a[item[1],0]*GV_cost+a[item[1],0]*GV_cost*q[item[1]]))
   standalone_cost_degree_2[item][item[1]] = adjustment*degree_2_coalition_cost[item]* (a[item[1],0]*GV_cost+a[item[1],0]*GV_cost*q[item[1]])/((a[item[0],0]*GV_cost+a[item[0],0]*GV_cost*q[item[0]])+(a[item[1],0]*GV_cost+a[item[1],0]*GV_cost*q[item[1]]))
standalone_cost_degree_2_copy = copy.deepcopy(standalone_cost_degree_2)
for item in standalone_cost_degree_2_copy:
   standalone_cost_degree_2[(item[1],item[0])]=standalone_cost_degree_2[item]


# Decision variables
x_e = {}
p = {}
e_IR = {}
e_S = {}
sc_e = {}
rc_e = {}
for item in E:
    for element in A:
        x_e[item,element] = mdl.addVar(vtype=GRB.BINARY, name=f"x_e[{item},{element}]")
x_d = mdl.addVars(((item, element) for item in D for element in A), vtype = GRB.BINARY, name = "x_d")
z = mdl.addVars(((item, element) for item in K for element in V), vtype = GRB.CONTINUOUS, name = "z")
l = mdl.addVars(((item, element) for item in K for element in V), vtype = GRB.INTEGER, name = "l")
b = mdl.addVars(((item, element) for item in E for element in V), vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name = "b")
b0 = mdl.addVars(((item, (i, 0)) for item in E for i in N), vtype=GRB.CONTINUOUS, name = "b0")
y = mdl.addVars(((item, (i, 0)) for item in E for i in N), vtype=GRB.CONTINUOUS, name = "y")
e_BB = mdl.addVar(vtype=GRB.CONTINUOUS, name = "e_BB")
tol = mdl.addVar(vtype=GRB.CONTINUOUS, name = "tol")

for i in N:
   p[i] = mdl.addVar(vtype=GRB.CONTINUOUS, name = f"p{i}")
   e_IR[i] = mdl.addVar(vtype=GRB.CONTINUOUS, name = f"e_IR{i}")
   e_S[i] = mdl.addVar(vtype=GRB.CONTINUOUS, name = f"e_S{i}")
for e in E:
   sc_e[e] = mdl.addVar(vtype=GRB.CONTINUOUS, name=f"sc_e[{e}]")
   rc_e[e] = mdl.addVar(vtype=GRB.CONTINUOUS, name=f"rc_e[{e}]")

mdl.update()
#ind = mdl.addVars(((item, element) for item in E for element in V), vtype = GRB.BINARY, name = "ind")


#Constraints
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


mdl.addConstrs(((b[e,j]*x_e[e,(i,j)] == x_e[e,(i,j)] * (b[e,i]*(1-r[i])+r[i]-(a[(i,j)]/EV_velocity)*(gamma+gamma_l*l[(e,i)]))) for i in V for j in N for e in E if i!=j), name='ssss')
mdl.addConstrs((b[e,j]*x_e[e,(i,j)] >= x_e[e,(i,j)]*(a[(j,0)]/EV_velocity)*(gamma+gamma_l*l[(e,j)])) for i in N for j in N for e in E if i!=j)
mdl.addConstrs((b[e,0] == 1) for e in E)
#mdl.addConstrs((b[e,i] >= 0.0) for e in E for i in N)
#mdl.addConstrs((b[e,i] <= 1) for e in E for i in N)


mdl.addConstrs((z[d,j]*x_d[d,(0,j)] == 2*x_d[d,(0,j)]*(st[j]+(a[0,j]/GV_velocity))) for j in N for d in D)

for j in N:
      for e in E:
         mdl.addGenConstrPWL(b0[e,(j,0)], y[e,(j,0)], [0, 0.8, 1], [300, 120, 0],  "myPWLConstr")

mdl.addConstrs((z[e,j]*x_e[e,(i,j)] == x_e[e,(i,j)]*(z[e,i]+st[i]+(a[(i,j)]/EV_velocity))) for i in V for j in N for e in E if i!=j)
mdl.addConstrs((b0[e,(j,0)]*x_e[e,(j,0)] == x_e[e,(j,0)] * (b[e,j] - (a[(j,0)]/EV_velocity)*(gamma+gamma_l*l[(e,j)]))) for j in N for e in E)
#mdl.addConstrs((b0[e,(j,0)]*x_e[e,(j,0)] >= battery_threshold*x_e[e,(j,0)]) for j in N for e in E)
mdl.addConstrs((b0[e,(j,0)]*x_e[e,(j,0)] >= battery_threshold*x_e[e,(j,0)] ) for j in N for e in E)
mdl.addConstrs((z[d,0] == 0) for d in D)
mdl.addConstrs((z[e,0] == 0) for e in E)
mdl.addConstrs((quicksum(z[d,j]*x_d[d,(0,j)] for j in N)<= T_max_GV) for d in D)


#h = mdl.addVars((item for item in E), vtype=GRB.CONTINUOUS, name = "h")
#mdl.addConstrs((h[e]==gp.max_((z[e,j]) for j in V)) for e in E)
#mdl.addConstrs((h[e] <= T_max) for e in E)
#mdl.addConstrs((z[e,i] -z[e,0]<= T_max_EV) for e in E for i in V  if i!=0)
mdl.addConstrs((quicksum(x_e[e,(j,0)]* (z[e,j]+y[e,(j,0)]+(a[(j,0)]/EV_velocity)) for j in N) <= T_max_EV) for e in E)


#mdl.addConstrs((quicksum(x_e[e,(0,j)] for j in N) <= (quicksum(x_e[e-1,(0,j)] for j in N)) for e in E if e!=min(E)))       #var_elim constraint. employ smaller labeled EV first
#mdl.addConstrs((quicksum(x_d[d,(0,j)] for j in N) <= (quicksum(x_d[d-1,(0,j)] for j in N)) for d in D if d!=min(D)))      #might be incorrect. var_elim constraint. employ smaller labeled GV first


#mdl.addConstrs(((x_e[e,(i,j)] - b[e,j]) >= 0) for e in E for i in V for j in N  if i!=j)

#coalition constraints


#IR
mdl.addConstrs((p[i]<=a[i,0]*GV_cost*q[i]+a[i,0]*GV_cost+e_IR[i]) for i in N)

#BB
mdl.addConstr(quicksum(p[i] for i in N)+(quicksum(e_IR[i] for i in N)) + (quicksum(e_S[i] for i in N))+ e_BB >= (quicksum((1-b0[e,(i,0)])*260*EV_cost*x_e[e,(i,0)] for i in N for e in E)))

for i in N:
   for j in N:
      if i!=j:
         mdl.addConstr(p[i]<=standalone_cost_degree_2[i,j][i]+e_S[i],name="stability")

#no payment if someone uses GV
for d in D:
   for j in N:
      mdl.addConstr(p[j]*x_d[d,(0,j)]<=1-x_d[d,(0,j)],name="no_payment")


#proportional cost allocation

#for e in E:
#   mdl.addConstr(sc_e[e] == quicksum(x_e[e,(j,i)]*a[j,i]*GV_cost*q[i] +  x_e[e,(j,i)]*a[j,i]*GV_cost for i in N for j in V if i!=j))
#   mdl.addConstr(rc_e[e] == quicksum((1-b0[e,(i,0)])*260*EV_cost*x_e[e,(i,0)] for i in N))
#
#for e in E:
#   for i in N:
#      mdl.addConstr(p[i] * sc_e[e] == a[0,i]*GV_cost*rc_e[e])

mdl.addConstrs((p[i]*x_e[e,(j,i)]>=(a[(i,j)]/EV_velocity)*(gamma+gamma_l*q[i])*260*EV_cost*x_e[e,(j,i)] ) for e in E for i in N for j in V if i!=j)





def variable_elimination():

   #mdl.addConstrs((quicksum(x_e[e,(0,j)] for j in N))<= (quicksum(x_e[e-1,(0,j)] for j in N)) for e in E if e!=min(E))  #employ smaller labeled truck first
   
   #elimination rule that concerns the load capacity limitation for EV
   count = 0
   for i in V:
      for j in V:
         if q[i]+q[j] > Q_EV and i!=j:
            mdl.addConstrs(x_e[e,(i,j)]==0 for e in E)
            count+=1
   #elimination rule that concerns the load capacity limitation for GV
   for j in N:
      if q[j] > Q_GV:
         mdl.addConstrs(x_d[d,(0,j)]==0 for d in D)
         mdl.addConstrs(x_d[d,(j,0)]==0 for d in D)
         count+=2

   #daily working time limitation for EV
   for i in N:
      for j in N:
         if i!=j and (a[(0,i)] + a[(i,j)] + a[(j,0)])/(EV_velocity) >= T_max_EV:
            mdl.addConstrs(x_e[e,(i,j)]==0 for e in E)
            count+=1
   
   #A vehicle can not travel on an arc if it is more than the battery capacity limitation
   for i in N:
      for j in N:
         if i!=j and ((a[(0,i)]/EV_velocity)*gamma + (a[(i,j)]/EV_velocity)*(gamma+gamma_l*q[i]) + (a[(j,0)]/EV_velocity)*(gamma+gamma_l*(q[i]+q[j]))) > 1-battery_threshold:
            mdl.addConstrs(x_e[e,(i,j)]==0 for e in E)
            count+=len(E)

def valid_inequality():
   #valid inequality: If the total distance of two successive arcs is larger than γ2G, a vehicle can not travel through both of the arcs
   for i in V:
      for j in N:
         for k in V:
            if i!=j and j!=k and ((a[(i,j)]/EV_velocity)*(gamma+gamma_l*q[i]) + (a[(j,k)]/EV_velocity)*(gamma+gamma_l*(q[i]+q[j]))) > 1-battery_threshold:
               mdl.addConstrs(x_e[e,(i,j)]+x_e[e,(j,k)] <= 1 for e in E)
            if i!=j and j!=k and (((a[(i,j)]/EV_velocity) + (a[(j,k)]/EV_velocity) + st[j] + st[k]) > T_max_EV):
               mdl.addConstrs(x_e[e,(i,j)]+x_e[e,(j,k)] <= 1 for e in E)

   #cuts based on load capacity limitation. If a subset C′ of the customers’ cargo load is more than a vehicle’s 
   # load capacity, those subsets of customers can not be served by a vehicle.
   for i in V:
      for j in N:
         for k in V:
            if i!=j and j!=k and q[i]+q[j]+q[k] > Q_EV:
               mdl.addConstrs(x_e[e,(i,j)]+x_e[e,(j,k)] <= 1 for e in E)

variable_elimination()
valid_inequality()

mdl.NumStart = 1
mdl.update()

#subprocess.run(['python3', 'hsc_ALNS_IFB.py'], check=True)  #or you could import hsc_ALNS_IFB in this file and change
# __name__ == "__main__" in the hsc_ALNS_IFB file to __name__ == "hsc_ALNS_IFB"

if not MIP_start:
  
  filenames = ['x_d_df.pkl', 'x_e_df.pkl', 'l_df.pkl']
  # Dictionary to hold the loaded dataframes
  loaded_dataframes = {}
  # Loop through the filenames and load each dataframe
  for filename in filenames:
     with open(filename, 'rb') as file:
        loaded_dataframes[filename[:-4]] = pickle.load(file)  # Remove .pkl extension for key
  # Access the loaded dataframes
  x_d_df = loaded_dataframes['x_d_df']
  x_e_df = loaded_dataframes['x_e_df']
  l_df = loaded_dataframes['l_df']

  # Set MIP start values for x_d

  for idx, row in x_d_df.iterrows():
      item = (row['index'])
      value = row['Value']
      mdl.getVarByName(f'x_d[{item[0]},{item[1]}]').Start = value
  # Set MIP start values for x_e
  for idx, row in x_e_df.iterrows():
      item = (row['index'])
      value = row['Value']
      mdl.getVarByName(f'x_e[{item[0]},{item[1]}]').Start = value

mdl.update()
mdl.modelSense = GRB.MINIMIZE

#Set objective
#mdl.setObjective((quicksum(x_d[d,(0,j)]*a[(0,j)]*GV_cost for d in D for j in V if j!=0))+(quicksum(x_d[d,(j,0)]*a[(j,0)]*GV_cost for j in V for d in D if j!=0))+(quicksum(x_e[e,(i,j)]*a[(i,j)]*EV_cost for i in V for j in V for e in E if i!=j))+ 0*quicksum(z[e,j] for e in E for j in N) + 0*quicksum(x_e[e,(0,j)]*e for e in E for j in N))

#mdl.setObjective((quicksum(x_d[d,(0,j)]*a[(0,j)] for j in N for d in D )) + (quicksum(x_d[d,(j,0)]*a[(j,0)] for j in N for d in D))+(quicksum(x_e[e,(i,j)]*a[(i,j)] for i in V for j in V for e in E if i!=j)))

mdl.setObjective((quicksum(x_d[d,(0,j)]*a[(0,j)]*2 for j in N for d in D )) +0.1*(e_BB + (quicksum(e_IR[i] for i in N)) + (quicksum(e_S[i] for i in N)))+(quicksum(x_e[e,(i,j)]*a[(i,j)] for i in V for j in V for e in E if i!=j)))

#mdl.setObjective((quicksum(x_d[d,(0,j)]*a[(0,j)]*GV_cost for j in N for d in D ))+ (e_BB + (quicksum(e_IR[i] for i in N)) + (quicksum(e_S[i] for i in N))) + (quicksum(x_d[d,(j,0)]*a[(j,0)] for j in N for d in D))+(quicksum(x_e[e,(i,j)]*a[(i,j)] for i in V for j in V for e in E if i!=j)))

#mdl.setObjective((quicksum(x_d[d,(0,j)]*a[(0,j)]*q[j]*GV_cost for d in D for j in N))+(quicksum(x_d[d,(j,0)]*a[(j,0)]*GV_cost for j in V for d in D if j!=0))+(quicksum((1-b0[e,(j,0)])*260*EV_cost*x_e[e,(j,0)] for j in N for e in E)))


mdl.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/hsc.lp")

mdl.Params.MIPGap = 0.05
mdl.params.NonConvex = 2
#mdl.Params.TimeLimit = 2000 #seconds
mdl.optimize()

#try:
#    mdl.computeIIS()
#    mdl.write("model.ilp")
#except: 2

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
p_result = get_vars('p')
e_S_result = get_vars('e_S')
e_BB_result = get_vars('e_BB')
e_IR_result = get_vars('e_IR')
rc_e_result = get_vars('rc_e')
def visualize_routes():
   G = nx.DiGraph(directed=True)
   label_pos_dict={}
   pos_dict = {}
   offset = 3
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
         if temp[-1]==0:
            nx.draw_networkx_labels(G, pos=label_pos_dict, labels={temp[1]: round((b0_result[f"b0[{temp[0]},{temp[1],0}]"]),3)}, font_color='red', font_weight='bold',font_size=8)

         if color_dict[temp[0]] not in legend_elements_ev:
            legend_elements_ev.append(color_dict[temp[0]])

   for item in x_d_result:
      if x_d_result[item]>0.9:
         temp = [int(match.group()) for match in re.finditer(r'\b\d+\b', item)]
         G.add_edge(temp[1], temp[2], color=color_dict[temp[0]])
         s = f"l[{temp[0]},{temp[2]}]"
         #nx.draw_networkx_labels(G, pos=label_pos_dict, labels={temp[2]: int(l_result[s])}, font_color='red', font_weight='bold',font_size=8)
         #nx.draw_networkx_labels(G, pos=label_pos_dict, labels={temp[2]: int(z_result[f"z[{temp[0]},{temp[2]}]"])}, font_color='black', font_weight='bold',font_size=8)
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
   plt.xlim(-grid_size/1.5, max(pos_dict.values(), key=lambda x: x[0])[0]+20)  # Adjust the limits based on your node coordinates
   plt.ylim(-grid_size/1.5, max(pos_dict.values(), key=lambda x: x[1])[1]+20)  # Adjust the limits based on your node coordinates

   pos = nx.spring_layout(G)
   edge_colors = [data['color'] for _, _, data in G.edges(data=True)]

   nx.draw(G, pos=pos_dict, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color=edge_colors)
   plt.legend(handles=all_legends, loc='upper right')
   plt.show()

def cost_calculation(x_e_result,x_d_result,b0_result):
   miles_EV = 0
   miles_GV = 0

   for item in x_e_result:
      if x_e_result[item]>=0.99:
         temp = [int(match.group()) for match in re.finditer(r'\b\d+\b', item)]
         miles_EV += a[(temp[1],temp[2])]

   for item in x_d_result:
      if x_d_result[item]>=0.99:
         temp = [int(match.group()) for match in re.finditer(r'\b\d+\b', item)]
         miles_GV += a[(temp[1],temp[2])]

   total_miles = miles_EV + miles_GV
   
   no_collab_total_miles = 0
   for i in range(1,n+1):
      no_collab_total_miles += a[(0,i)]*2

   miles_saved = no_collab_total_miles-total_miles
   percentage_miles_saved = (miles_saved/(no_collab_total_miles))*100

   route = {}
   for item in E:
      route[item] = []
   for item in x_e_result:
      if x_e_result[item]>0.99:
         temp = [int(match.group()) for match in re.finditer(r'\b\d+\b', item)]
         route[temp[0]].append(temp[2])
   #a1 = copy.deepcopy(a)
   #a1[(0,0)]=0
   #for item in route:
   #   num_nodes = len(item)
   #   for i in range(0,num_nodes-1):
   #      2
   battery_consumed = 0
   for item in x_e_result:
      if x_e_result[item]>0.99:
         temp = [int(match.group()) for match in re.finditer(r'\b\d+\b', item)]
         if temp[-1]==0:
            battery_consumed+= (1-b0_result[f"b0{item[3:]}"])
   cost_EV = 260*EV_cost*battery_consumed
   cost_GV = 0
   for item in x_d_result:
      if x_d_result[item]>0.99:
         temp = [int(match.group()) for match in re.finditer(r'\b\d+\b', item)]
         if temp[-1]==0:
            cost_GV+= a[temp[1],temp[2]]*GV_cost
         else:
            cost_GV+=a[temp[1],temp[2]]*GV_cost*q[temp[1]]
   total_cost = cost_EV+cost_GV

   no_collab_total_cost = 0
   for i in range(1,n+1):
      no_collab_total_cost += a[(i,0)]*q[i]*GV_cost + a[(0,i)]*GV_cost
      
   cost_saved = no_collab_total_cost-total_cost
   percentage_cost_saved = (cost_saved/no_collab_total_cost)*100
   2
   




   return cost_EV 

cost_EV = cost_calculation(x_e_result,x_d_result,b0_result)
visualize_routes()

print(f"total payment= {sum(p_result.values()),cost_EV}")
print(f"stability subsidy= {sum(e_S_result.values())}")
print(f"IR subsidy= {sum(e_IR_result.values())}")
print(f"BB subsidy= {sum(e_BB_result.values())}")
print(f"payments= {p_result}")
print(f"stability= {e_S_result}")

2
#miles_saved, percentage_miles_saved, cost_saved, percentage_cost_saved =  cost_calculation(x_e_result,x_d_result,b0_result)