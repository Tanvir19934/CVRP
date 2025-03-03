import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from gurobipy import Model, GRB, quicksum
import re
from matplotlib.lines import Line2D
import random
import pickle
from config_new import N, V, Q_EV, q, a, NODES, grid_size, xc, yc, w_dv, w_ev, theta, MIP_start, gamma, gamma_l, battery_threshold, EV_cost, GV_cost, T_max_EV, T_max_GV, EV_velocity, GV_velocity, Q_GV, num_TV, num_GV, num_EV, E, D, K, A, r, st
from utils_new_auto import generate_all_possible_routes, ev_travel_cost, tsp_tour, gv_tsp_cost, save_to_excel, update_config, refresh_config
import pandas as pd
import time
import importlib

def main():
   rnd = np.random
   #rnd.seed(10)
   # [[0, 2, 4, 6, 0], [0, 1, 9, 7, 0], [0, 5, 12, 10, 11, 0]]
   #override some config parameters
   n = NODES
   q[0] = 0
   #T_max_EV = 700

   ###########Gurobi model############
   mdl = Model('mip')

   # Decision variables
   x_e = {}
   p = {}

   all_routes = generate_all_possible_routes(N)
   mr = {}
   for item in all_routes:

      
      mr[item] = gv_tsp_cost(item)


   for item in E:
      for element in A:
         x_e[item,element] = mdl.addVar(vtype=GRB.BINARY, name=f"x_e[{item},{element}]")
   x_d = mdl.addVars(((item, element) for item in D for element in A), vtype = GRB.BINARY, name = "x_d")
   #z = mdl.addVars(((item, element) for item in K for element in V), vtype = GRB.CONTINUOUS, name = "z")
   l = mdl.addVars(((item, element) for item in K for element in V), vtype = GRB.INTEGER, name = "l")
   b = mdl.addVars(((item, element) for item in E for element in V), vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name = "b")
   b0 = mdl.addVars(((item, (i, 0)) for item in E for i in N), vtype=GRB.CONTINUOUS, name = "b0")
   y = mdl.addVars(((item, (i, 0)) for item in E for i in N), vtype=GRB.CONTINUOUS, name = "y")
   tol = mdl.addVar(vtype=GRB.CONTINUOUS, name = "tol")

   for i in N:
      p[i] = mdl.addVar(vtype=GRB.CONTINUOUS, name = f"p{i}")


   mdl.update()


   #Constraints
   mdl.addConstrs((x_d[j,(0,j)] + (quicksum(x_e[e,(i,j)] for e in E for i in V if i!=j))== 1) for j in N)
   mdl.addConstrs(((x_d[d,(0,j)])-(x_d[d,(j,0)]) == 0) for d in D for j in N)
   mdl.addConstrs(((x_d[j,(0,i)]) == 0) for i in N for j in N if i!=j) #use own GV
   mdl.addConstrs((quicksum(x_d[d,(i,j)] for i in N for j in N if i!=j)==0) for d in D) #forbids intra-GV travel
   mdl.addConstrs(((quicksum(x_e[e,(i,j)] for i in V if i!=j)-quicksum(x_e[e,(j,i)] for i in V if i!=j) == 0) for j in V for e in E), name='s')


   mdl.addConstrs((l[(e,i)]  <= Q_EV) for i in N  for e in E) #load capacity constraint for EV
   mdl.addConstrs((q[j]*x_d[d,(0,j)] <= Q_GV) for d in D for j in N) #load capacity constraint for GV
   mdl.addConstrs((l[(e,j)]* x_e[e,(i,j)] ==  x_e[e,(i,j)]*(l[(e,i)]+q[j])) for e in E for j in N for i in V if i!=j) #load upon arriving at j is equal to j's total load+all previous loads
   mdl.addConstrs((l[(d,j)]* x_d[d,(i,j)] ==  x_d[d,(i,j)]*(l[(d,i)]+q[j])) for d in D for j in N for i in V if i!=j) #load upon arriving at j is equal to j's total load+all previous loads
   mdl.addConstrs((l[(k,0)] == 0) for k in K)


   mdl.addConstrs(((b[e,j]*x_e[e,(i,j)] == x_e[e,(i,j)] * (b[e,i]*(1-r[i])+r[i]-(a[(i,j)]/EV_velocity)*(gamma+gamma_l*l[(e,i)]))) for i in V for j in N for e in E if i!=j), name='ssss') #tracks rem battery capacity for each intermediate node
   mdl.addConstrs((b[e,j]*x_e[e,(i,j)] >= x_e[e,(i,j)]*(a[(j,0)]/EV_velocity)*(gamma+gamma_l*l[(e,j)])) for i in N for j in N for e in E if i!=j) #enough battery to go back to depot
   mdl.addConstrs((b[e,0] == 1) for e in E) #starts with full battery



   #mdl.addConstrs((z[d,j]*x_d[d,(0,j)] == 2*x_d[d,(0,j)]*(st[j]+(a[0,j]/GV_velocity))) for j in N for d in D) #tracks time for GV

   #for j in N:
   #      for e in E:
   #         mdl.addGenConstrPWL(b0[e,(j,0)], y[e,(j,0)], [0, 0.8, 1], [300, 120, 0],  "myPWLConstr")

   #mdl.addConstrs((z[e,j]*x_e[e,(i,j)] == x_e[e,(i,j)]*(z[e,i]+st[i]+(a[(i,j)]/EV_velocity))) for i in V for j in N for e in E if i!=j) #tracks time for EV
   mdl.addConstrs((b0[e,(j,0)]*x_e[e,(j,0)] == x_e[e,(j,0)] * (b[e,j] - (a[(j,0)]/EV_velocity)*(gamma+gamma_l*l[(e,j)]))) for j in N for e in E)
   mdl.addConstrs((b0[e,(j,0)]*x_e[e,(j,0)] >= battery_threshold*x_e[e,(j,0)] ) for j in N for e in E)
   #mdl.addConstrs((z[d,0] == 0) for d in D) #starts at time 0
   #mdl.addConstrs((z[e,0] == 0) for e in E) #starts at time 0
   #mdl.addConstrs((quicksum(z[d,j]*x_d[d,(0,j)] for j in N)<= T_max_GV) for d in D) #daily working time limitation for GV
   #mdl.addConstrs((quicksum(x_e[e,(j,0)]* (z[e,j]+y[e,(j,0)]+(a[(j,0)]/EV_velocity)) for j in N) <= T_max_EV) for e in E)

   # To ensure singleton cannot be an EV
   mdl.addConstrs(x_e[e,(j,0)] + x_e[e,(0,j)] <= 1 for e in E for j in N) 

   # To ensure one EV makes one trip
   mdl.addConstrs(quicksum(x_e[e,(j,0)] for j in N) <= 1 for e in E) 


   #coalition constraints

   #IR
   mdl.addConstrs((p[i]<=a[i,0]*GV_cost*q[i]+a[i,0]*GV_cost) for i in N)


   #BB
   mdl.addConstr(quicksum(p[i] for i in N) <= (quicksum((1-b0[e,(i,0)])*260*EV_cost*x_e[e,(i,0)] for i in N for e in E)))
   #no payment if someone uses GV
   for d in D:
      for j in N:
         mdl.addConstr(p[j]*x_d[d,(0,j)]<=1-x_d[d,(0,j)],name="no_payment")


   # Stability
   mdl.addConstrs((quicksum(p[i] for i in N if i in route) <= mr[route]) for route in all_routes if len(route)>3)


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
      #for i in N:
      #   for j in N:
      #      if i!=j and (a[(0,i)] + a[(i,j)] + a[(j,0)])/(EV_velocity) >= T_max_EV:
      #         mdl.addConstrs(x_e[e,(i,j)]==0 for e in E)
      #         count+=1
      
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
   valid_inequality()      #[[0, 8, 6, 4, 0], [0, 1, 7, 2, 0]]

   #mdl.NumStart = 1
   mdl.update()


   if MIP_start:
   
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
   mdl.setParam('TimeLimit', 3000)  # Set a 60-second time limit

   #Set objective
   #mdl.setObjective((quicksum(x_d[d,(0,j)]*a[(0,j)]*2 for j in N for d in D )) +0.1*(e_BB + (quicksum(e_IR[i] for i in N)) + (quicksum(e_S[i] for i in N)))+(quicksum(x_e[e,(i,j)]*a[(i,j)] for i in V for j in V for e in E if i!=j)))

   mdl.setObjective((quicksum(x_d[d,(0,j)]*a[(0,j)]*2 for j in N for d in D ))*w_dv + (quicksum(x_e[e,(i,j)]*a[(i,j)] for i in V for j in V for e in E if i!=j))*w_ev \
                  +  theta * ((quicksum((1-b0[e,(i,0)])*260*EV_cost*x_e[e,(i,0)] for i in N for e in E)) - quicksum(p[i] for i in N))) 


   mdl.Params.MIPGap = 0.05
   #mdl.params.NonConvex = 2
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

   for item in p_result:
      p_result[item] = round(p_result[item],2)

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

      print(f"total miles= {total_miles}")
      print(f"miles_EV= {miles_EV}")
      print(f"miles_GV= {miles_GV}")
      
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
      
      return cost_EV, miles_EV, miles_GV

   def route_construction(x_e_result):
      routes = {}
      for item in E:
         routes[f"{item}"] = []
      for item in x_e_result:
         if x_e_result[item]>0.99:
            temp = [int(match.group()) for match in re.finditer(r'\b\d+\b', item)]
            if temp[1]==0:
               routes[f"{temp[0]}"].append([temp[1],temp[2]])
      for item in routes:
         for elem in routes[item]:
            while elem[-1]!=0:
               for res in x_e_result:
                  if x_e_result[res]>0.99:
                     temp =  [int(match.group()) for match in re.finditer(r'\b\d+\b', res)]
                     if temp[0]==int(item) and temp[1]==elem[-1]:
                        elem.append(temp[-1])
      return routes



   ev_routes = route_construction(x_e_result)
   ev_routes_list = []
   for item in ev_routes:
      for elem in ev_routes[item]:
         ev_routes_list.append(elem)
   cost_EV=0
   for item in ev_routes_list:
      if True:#len(item)!=30:
         cost = ev_travel_cost(item)
         cost_EV+=cost

   #visualize_routes()
   cost_EV, miles_EV, miles_GV = cost_calculation(x_e_result,x_d_result,b0_result)
   print(f"total payment, EV tour cost (manual), EV tour cost = {sum(p_result.values()),cost_EV, sum(p_result.values()) + (mdl.getObjective().getValue()-(miles_EV*w_ev+miles_GV*w_dv))/theta}")
   print(f"Total subsidy: {cost_EV - sum(p_result.values())}")
   print(f"payments= {p_result}")
   print(ev_routes_list)
   return mdl.getObjective().getValue(), miles_EV+miles_GV, miles_EV, sum(p_result.values()), cost_EV - sum(p_result.values()), p_result, ev_routes_list

if __name__=="__main__":
    
    nodes = [i for i in range(5, 16)]

    for item in nodes:
        
        update_config(item)
        time.sleep(10)
        import config_new
        importlib.reload(config_new)  # Reload the module to update V, q, a, Q_EV
        globals().update({k: getattr(config_new, k) for k in dir(config_new) if not k.startswith("__")})



        # Run the branching logic
        start = time.perf_counter()
        obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes = main()
        end = time.perf_counter()
        print(f"Execution time for nodes={item}: {end - start}")
        Execution_time = end - start
        data = {
            "Nodes": [item],
            "Obj": [obj],
            "Total Miles": [total_miles],
            "EV miles": [EV_miles],
            "Total payments": [Total_payments],
            "Subsidy": [Subsidy],
            "Execution time (sec.)": [Execution_time]
        }
        df = pd.DataFrame(data)
        file_name = "New_codes/results.xlsx"
        save_to_excel(file_name, "Sheet3", df)
        data = {
            "Nodes": [item],
            "Payments": [payments],
            "Solution routes": [solution_routes]
        }
        df = pd.DataFrame(data)
        file_name = "New_codes/results.xlsx"
        save_to_excel(file_name, "Sheet4", df)

#miles_saved, percentage_miles_saved, cost_saved, percentage_cost_saved =  cost_calculation(x_e_result,x_d_result,b0_result)