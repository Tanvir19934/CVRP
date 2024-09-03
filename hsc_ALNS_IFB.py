import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import copy
import re
import random
import pandas as pd
import pickle
from config import *
from gurobipy import Model, GRB, quicksum

rnd = np.random
rnd.seed(10)

def visualize_routes(EV_dict):
   G = nx.DiGraph(directed=True)
   pos = nx.spring_layout(G)
   label_pos_dict={}
   pos_dict = {}
   offset = 0
   color_dict = {i: (random.random(), random.random(), random.random()) for i in range(1, num_TV + 1)}

   for i in range(len(xc)):
      pos_dict[i] = (xc[i], yc[i])
   for i in range(len(xc)):
      label_pos_dict[i] = (xc[i]+offset, yc[i]+offset)

   for item in EV_dict:
      for element in EV_dict[item]['route']:
         num_nodes = len(element)
         if num_nodes!=1:
            for i in range(0,num_nodes-1):
               G.add_edge(element[i],element[i+1],color=color_dict[item])
   edge_colors = [data['color'] for _, _, data in G.edges(data=True)]

   nx.draw(G, pos=pos_dict, with_labels=True, node_size=500, node_color="skyblue", font_size=8, font_color="black", font_weight="bold", edge_color=edge_colors)
   plt.show()

def k_means_3D(x,y,z,num_clusters):
   X = np.column_stack((x[1:], y[1:], [1/z[i] for i in range(1, len(z) + 1)]))
   kmeans = KMeans(num_clusters)                   # Number of clusters == 3
   kmeans = kmeans.fit(X)                          # Fitting the input data
   labels = kmeans.predict(X)                      # Getting the cluster labels
   centroids = kmeans.cluster_centers_             # Centroid values
   # print("Centroids are:", centroids)              # From sci-kit learn

   fig = plt.figure(figsize=(10,10))
   ax = fig.add_subplot(projection = '3d')

   x = np.array(labels==0)
   y = np.array(labels==1)
   z = np.array(labels==2)

   ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c="black",s=150,label="Centers",alpha=1)
   ax.scatter(X[x,0],X[x,1],X[x,2],c="blue",s=40,label="C1")
   ax.scatter(X[y,0],X[y,1],X[y,2],c="yellow",s=40,label="C2")
   ax.scatter(X[z,0],X[z,1],X[z,2],c="red",s=40,label="C3")
   return labels, centroids , X

def k_means_transformed(x,y,z,num_cluster):
   x = copy.deepcopy(xc)
   y = copy.deepcopy(yc)
   max_dem = max(z)
   for item in z:
      x[item] = x[item]*(z[item]/max_dem)
      y[item] = y[item]*(z[item]/max_dem)
   X = np.column_stack((x[1:],y[1:]))
   kmeans = KMeans(n_clusters=num_cluster)                   # Number of clusters == 3
   kmeans = kmeans.fit(X)                          # Fitting the input data
   labels = kmeans.predict(X)                      # Getting the cluster labels
   centroids = kmeans.cluster_centers_             # Centroid values
   # print("Centroids are:", centroids)              # From sci-kit learn

   # Visualize clusters
   plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data Points')
   plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=100, c='red', label='Centroids')
   plt.xlabel('Distance')
   plt.ylabel('Additional Feature (e.g., q)')
   plt.title('K-Means Clustering')
   plt.legend()
   plt.show()
   
   plt.scatter(xc[1:], yc[1:], c=labels, cmap='viridis', label='Data Points')
   for item in q:
      plt.text(xc[item] + 0.2, yc[item] + 0.2, str(q[item]), fontsize=8, color='black')
   plt.text(0,0,"DC",fontsize=10, color='red')
   plt.show()
   return labels, centroids , X

def k_means_transformed_best_k(x,y,z,num_clusters):
   x = copy.deepcopy(xc)
   y = copy.deepcopy(yc)
   max_dem = max(z.values())
   inertias = []
   # this groups higher loads-higher distances and lower loads-lower distances together 
   for item in z:
      x[item] *= (max_dem/(z[item]+0.00001))
      y[item] *= (max_dem/(z[item]+0.00001))
   # this groups higher loads-lower distances and lower loads-higher distances together 
   #for item in z:
   #   x[item] = x[item]*(z[item]/max_dem)
   #   y[item] = y[item]*(z[item]/max_dem)
   #or we could not transform at all and keep x=xc, y=yc
   X = np.column_stack((x[1:],y[1:]))
   for i in range(2,len(z)):
      kmeans = KMeans(n_clusters=i)
      kmeans.fit(X)
      inertias.append(kmeans.inertia_)
   plt.plot(range(2,len(z)), inertias, marker='o')
   plt.title('Elbow method')
   plt.xlabel('Number of clusters')
   plt.ylabel('Inertia')
   plt.show()

   #num_clusters = 5 #based on intertia plot
   kmeans = KMeans(n_clusters=num_clusters)                   # Number of clusters == 3
   kmeans = kmeans.fit(X)                          # Fitting the input data
   labels = kmeans.predict(X)                      # Getting the cluster labels
   centroids = kmeans.cluster_centers_             # Centroid values
   # print("Centroids are:", centroids)              # From sci-kit learn

   # Visualize clusters
   plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='RdYlGn', label='Data Points',s=110)
   plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=80, c='red', label='Centroids')
   plt.text(0,0,"DC",fontsize=10, color='red')
   for item in q:
      #plt.text(xc[item] + 3, yc[item] + 3, str(q[item]), fontsize=8, color='red')
      plt.text(X[item-1][0], X[item-1][1], str(item), fontsize=10, color='black')
   plt.xlabel('Distance')
   plt.ylabel('Additional Feature (e.g., q)')
   plt.title('K-Means Clustering')
   plt.legend()
   plt.show()
   
   plt.scatter(xc[1:], yc[1:], c=labels, cmap='RdYlGn', label='Data Points',s=110)
   for item in q:
      plt.text(xc[item] + 3, yc[item] + 3, str(q[item]), fontsize=8, color='red')
      plt.text(xc[item] , yc[item], str(item), fontsize=10, color='black')
   plt.text(0,0,"DC",fontsize=10, color='red')
   plt.show()
   return labels, centroids, X

def k_means_2D(node_attr,z,num_cluster):
   dist_values = np.array([entry['dist'] for entry in node_attr.values()])
   X = np.column_stack((dist_values[0:], [1/z[i] for i in range(1, len(z) + 1)]))
   kmeans = KMeans(n_clusters=num_cluster)                   # Number of clusters == 3
   kmeans = kmeans.fit(X)                          # Fitting the input data
   labels = kmeans.predict(X)                      # Getting the cluster labels
   centroids = kmeans.cluster_centers_             # Centroid values
   # print("Centroids are:", centroids)              # From sci-kit learn

   # Visualize clusters
   plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data Points')
   plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=100, c='red', label='Centroids')
   plt.xlabel('Distance')
   plt.ylabel('Additional Feature (e.g., q)')
   plt.title('K-Means Clustering')
   plt.legend()
   plt.show()

   plt.scatter(xc[1:], yc[1:], c=labels, cmap='viridis', label='Data Points')
   for item in q:
      plt.text(xc[item] + 0.2, yc[item] + 0.2, str(q[item]), fontsize=8, color='black')
   plt.text(0,0,"DC",fontsize=10, color='red')
   plt.show()
   return labels, centroids , X

def get_node_att():
   node_attr = {}
   for i in range(1,len(N)+1):
      node_attr[i] = {}
      node_attr[i]["dem"] = q[i]
      node_attr[i]["dist"] = a[(0,i)]
   #labels, centroids, X = k_means_3D(xc,yc,q,num_EV)
   #labels, centroids, X= k_means_2D(node_attr,q,num_EV)
   #labels, centroids, X = k_means_transformed(xc,yc,q,num_EV) #better
   labels, centroids, X = k_means_transformed_best_k(xc,yc,q,num_clusters)  #better
   for item in node_attr:
      node_attr[item]["transformed_dist"] = np.hypot(xc[0]-X[int(item)-1][0], yc[0]- X[int(item)-1][1])
   return labels, centroids, X, node_attr

def battery_level(x,y,current_load, current_battery):
   flag_b = True
   flag_l = True
   current_load += q[y]
   battery_consumption = (a[(x,y)]/EV_velocity)*(gamma+gamma_l*current_load) 
   rem_battery = current_battery - battery_consumption
   rem_battery_home = current_battery - (battery_consumption + (a[(y,0)]/EV_velocity)*(gamma+gamma_l*current_load))
   if (current_load > Q_EV) and (rem_battery_home>=battery_threshold):
      current_load -= q[y]
      flag_l = False
      rem_battery += battery_consumption
      return rem_battery, current_load, flag_b, flag_l 
   if (rem_battery_home < battery_threshold) and (current_load <= Q_EV):
      current_load -= q[y]
      rem_battery += battery_consumption
      flag_b = False
      return rem_battery, current_load, flag_b, flag_l
   if (rem_battery_home < battery_threshold) and (current_load > Q_EV):
      current_load -= q[y]
      rem_battery += battery_consumption
      flag_b = False
      flag_l = False
      return rem_battery, current_load, flag_b, flag_l  
   return rem_battery, current_load, flag_b, flag_l 

def operation_time(x,y,current_time):
   flag_o = True
   time_consumption = st[y]+(a[(x,y)]/EV_velocity)
   rem_time = current_time - time_consumption
   rem_time_home = current_time - time_consumption - (a[(y,0)]/EV_velocity)
   if rem_time_home < 0:
      flag_o = False
      rem_time += time_consumption
      return rem_time, flag_o
   return rem_time, flag_o

def IFB(labels, centroids, X, node_attr):
   label_set = list(set(labels))
   cluster_info = {}
   for item in label_set:
      cluster_info[item] = {}
      cluster_info[item]["nodes"] = {}
      for element in list(j+1 for j in range(0,len(labels)) if labels[j]==item):
         cluster_info[item]["nodes"][element] = {}
         cluster_info[item]["nodes"][element]["transformed_coord"] = X[element-1]
         cluster_info[item]["nodes"][element]["actual_coord"] = [xc[element],yc[element]]
         cluster_info[item]["nodes"][element]["transformed_dist_from_DC"] = np.hypot(xc[0]-cluster_info[item]["nodes"][element]["transformed_coord"][0],yc[0]-cluster_info[item]["nodes"][element]["transformed_coord"][1])
         cluster_info[item]["nodes"][element]["actual_dist_from_DC"] = np.hypot(xc[0]-cluster_info[item]["nodes"][element]["actual_coord"][0],yc[0]-cluster_info[item]["nodes"][element]["actual_coord"][1])
      cluster_info[item]["centroid"] = centroids[item]
      cluster_info[item]["centroid_dist"] = np.hypot(xc[0]-cluster_info[item]["centroid"][0], yc[0]- cluster_info[item]["centroid"][1])
      cluster_info[item]["nodes"] = dict(sorted(cluster_info[item]["nodes"].items(), key=lambda item: item[1]['transformed_dist_from_DC']))
   ordered_cluster = {}
   for item in cluster_info:
      ordered_cluster[item] = cluster_info[item]["centroid_dist"]
   ordered_cluster = dict(sorted(ordered_cluster.items(), key=lambda x: x[1]))
   for item in node_attr:
      for element in cluster_info:
         if item in list(cluster_info[element]["nodes"].keys()):
            node_attr[item]["transformed_coord"] = cluster_info[element]["nodes"][item]["transformed_coord"]
            node_attr[item]["actual_coord"] = cluster_info[element]["nodes"][item]["actual_coord"]

   EV_dict = {}
   for i in range(E[0],E[-1]+1):
      EV_dict[i] = {}
      EV_dict[i]["route"] = [[0]]
      EV_dict[i]["battery"] = [[1]]
      EV_dict[i]["time"] = [T_max_EV]
      EV_dict[i]["curr_load"] = [[0]]
   GV_dict = {}
   for i in range(1,num_GV+1):
      GV_dict[i] = {}
      GV_dict[i]["route"] = []
      GV_dict[i]["time"] = []
   nodes = {}
   for item in cluster_info:
      nodes[item] = list(cluster_info[item]["nodes"].keys())
   nodes_original = copy.deepcopy(nodes)
   assigned_nodes = []
   unassigned_nodes = copy.deepcopy(list(q.keys()))
   for ev in EV_dict:
      for item in range(0,len(ordered_cluster.keys())):
         for element in nodes[list(ordered_cluster.keys())[item]]:
            if element == 6:
               2
            if element in unassigned_nodes:
               b, l, flag_b, flag_l = battery_level(EV_dict[ev]['route'][-1][-1],element,EV_dict[ev]['curr_load'][-1][-1],EV_dict[ev]['battery'][-1][-1])
               o, flag_o = operation_time(EV_dict[ev]['route'][-1][-1],element,EV_dict[ev]["time"][-1])
               if flag_o==True:
                  if flag_b==True and flag_l==True:
                     EV_dict[ev]['route'][-1].append(element)
                     EV_dict[ev]['battery'][-1].append(b)
                     EV_dict[ev]['time'].append(o)
                     EV_dict[ev]['curr_load'][-1].append(l)
                     assigned_nodes.append(element)
                     unassigned_nodes.remove(element)
                     if random.uniform (0, 1) <= alpha:         #make a trip back to depot anyway with alpha% chance to replenish the battery
                        b = b - (a[(EV_dict[ev]['route'][-1][-1],0)]/EV_velocity)*(gamma+gamma_l*l)  #travel back to depot
                        o = o - (a[(EV_dict[ev]['route'][-1][-1],0)]/EV_velocity)
                        EV_dict[ev]['route'][-1].append(0)
                        EV_dict[ev]['curr_load'][-1].append(0)
                        EV_dict[ev]['battery'][-1].append(b)
                        if b < 0.8:
                           t = a[(EV_dict[ev]['route'][-1][-1],0)]/EV_velocity + 225*(0.8-b) + 120
                        elif b > 0.8:
                           t = a[(EV_dict[ev]['route'][-1][-1],0)]/EV_velocity + (b-0.8)*600 + 180
                        EV_dict[ev]['time'].append(o-t)
                        EV_dict[ev]['route'].append([0])
                        EV_dict[ev]['curr_load'].append([0])
                        EV_dict[ev]['battery'].append([1])
               
                  elif (flag_b==False and flag_l==True) or (flag_b==True and flag_l==False) or (flag_b==False and flag_l==False):

                     b = b - (a[(EV_dict[ev]['route'][-1][-1],0)]/EV_velocity)*(gamma+gamma_l*l)   #travel back to depot
                     o = o + (st[element]+(a[(EV_dict[ev]['route'][-1][-1],element)]/EV_velocity)) #compensate time since element is not added to the trip
                     EV_dict[ev]['route'][-1].append(0)
                     EV_dict[ev]['curr_load'][-1].append(0)
                     EV_dict[ev]['battery'][-1].append(b)
                     if b < 0.8:
                        t = a[(EV_dict[ev]['route'][-1][-1],0)]/EV_velocity + 225*(0.8-b) + 120
                     elif b > 0.8:
                        t = a[(EV_dict[ev]['route'][-1][-1],0)]/EV_velocity + (b-0.8)*600 + 180
                     EV_dict[ev]['time'].append(o-t)
                     t2 = a[(0,element)]/EV_velocity + st[element]
                     t3 = a[(element,0)]/EV_velocity
                     b2 = (a[(0,element)]/EV_velocity)*(gamma+gamma_l*0)
                     b3 = (a[(element,0)]/EV_velocity)*(gamma+gamma_l*q[element])
                     EV_dict[ev]['route'].append([0])
                     EV_dict[ev]['curr_load'].append([0])
                     EV_dict[ev]['battery'].append([1])
                     if 1 - (b2+b3) >= battery_threshold and o-(t+t2+t3) >= 0:
                        EV_dict[ev]['route'][-1].append(element)
                        EV_dict[ev]['curr_load'][-1].append(q[element])
                        EV_dict[ev]['battery'][-1].append(1-b2)
                        EV_dict[ev]['time'].append(o-t-t2)
                        assigned_nodes.append(element)
                        unassigned_nodes.remove(element)
                     else:
                        #if item!=len(ordered_cluster.keys())-1 and element not in nodes[list(ordered_cluster.keys())[item+1]]:
                        #   nodes[list(ordered_cluster.keys())[item+1]].append(element)
                        
                        if ev!=len(EV_dict):
                           potential_cluster = list(ordered_cluster.keys())
                           potential_cluster.remove(potential_cluster[item])
                           dist = {}
                           for j in potential_cluster:
                              dist[j] = np.hypot(cluster_info[j]["centroid"][0]-node_attr[element]["transformed_coord"][0],cluster_info[j]["centroid"][1]-node_attr[element]["transformed_coord"][1])
                              #dist[j] = np.hypot(0-node_attr[element]["actual_coord"][0],0-node_attr[element]["actual_coord"][1])
                           dist = dict(sorted(dist.items(), key=lambda x: x[1]))
                           nodes[list(dist.keys())[0]].append(element)


   for item in EV_dict:
      for element in EV_dict[item]["route"]:
         if element[-1]!=0:
            b = EV_dict[item]["battery"][-1][-1] - (a[(element[-1],0)]/EV_velocity)*(gamma+gamma_l*EV_dict[item]["curr_load"][-1][-1])
            EV_dict[item]["battery"][-1].append(b)
            EV_dict[item]["curr_load"][-1].append(0)
            if b < 0.8:
               t = a[(EV_dict[ev]['route'][-1][-1],0)]/EV_velocity + 225*(0.8-b) + 120
            elif b > 0.8:
               t = a[(EV_dict[ev]['route'][-1][-1],0)]/EV_velocity + (b-0.8)*600 + 180
            z = (a[(element[-1],0)]/EV_velocity)
            EV_dict[item]["time"].append(EV_dict[item]["time"][-1]-z-t)
            element.append(0)
   2
   return EV_dict, unassigned_nodes, cluster_info, nodes_original

def cost_calculation(GV_assignment,EV_assignment,EV_dict_copy):
   cost_GV = 0
   miles_GV = 0
   cost_EV = 0
   miles_EV = 0
   for item in GV_assignment:
      miles_GV+=a[(0,item)]*2
      GV_assignment[item]['miles'] = a[(0,item)]*2
   for item in EV_dict_copy:
      for element in EV_dict_copy[item]["route"]:
         num_nodes = len(element)
         if num_nodes!=1 and num_nodes!=3:
            for i in range(0,num_nodes-1):
               miles_EV+=a[(element[i],element[i+1])]
   miles_IFB = miles_EV+miles_GV
   for item in GV_assignment:
      cost_GV+=a[(0,item)]*GV_cost + a[(item,0)]*GV_cost*q[item]
      GV_assignment[item]['cost'] = a[(0,item)]*GV_cost + a[(item,0)]*GV_cost*q[item]
   battery_consumed = 0
   for item in EV_dict_copy:
      for element in EV_dict_copy[item]["battery"]:
         battery_consumed+=(1-element[-1])
   for item in EV_assignment:
      for element in EV_assignment[item]:
         EV_assignment[item][element]['cost'] = 260*EV_cost*(1-EV_assignment[item][element]['battery'])
         r = [int(e) for e in element[1:-1].split(', ')]
         mile = 0
         for i in range(0,len(r)-1):
            mile += a[(r[i],r[i+1])]
         EV_assignment[item][element]['miles'] = mile

   cost_EV = 260*EV_cost*battery_consumed
   cost_IFB= cost_EV + cost_GV
   return cost_IFB, miles_IFB, GV_assignment, EV_assignment

def EV_GV_assignment(EV_dict, unassigned_nodes):
   GV_assignment = {}
   EV_assignment = {}
   for item in EV_dict:
      for element in EV_dict[item]["route"]:
         num_nodes = len(element)
         if num_nodes>3:
            EV_assignment[item] = {}
   for item in EV_dict:
      for idx,element in enumerate(EV_dict[item]["route"]):
         num_nodes = len(element)
         if num_nodes==3:
            GV_assignment[element[1]]={}
            GV_assignment[element[1]]['cost'] = 0
            GV_assignment[element[1]]['miles'] = 0
         elif num_nodes>3:
            EV_assignment[item][f'{element}'] = {}
            EV_assignment[item][f'{element}']['cost'] = 0
            EV_assignment[item][f'{element}']['miles'] = 0
            EV_assignment[item][f'{element}']['battery'] = EV_dict[item]['battery'][idx][-1]
   if unassigned_nodes:
      for item in unassigned_nodes:
         GV_assignment[item]={}
         GV_assignment[item]['cost'] = 0
         GV_assignment[item]['miles'] = 0

   return GV_assignment, EV_assignment

def get_vars(item,opt_route):
   vars = [var for var in opt_route.getVars() if f"{item}" in var.VarName]
   names = opt_route.getAttr('VarName', vars)
   values = opt_route.getAttr('X', vars)
   return dict(zip(names, values))

def optimize_routes(A, V, E):
   mdl = Model('optimize_routes')
   N = V[1:]
   x_e = {}
   for item in E:
      for element in A:
        x_e[item,element] = mdl.addVar(vtype=GRB.BINARY, name=f"x_e[{item},{element}]")
   z = mdl.addVars(((item, element) for item in E for element in V), vtype = GRB.CONTINUOUS, name = "z")
   l = mdl.addVars(((item, element) for item in E for element in V), vtype = GRB.INTEGER, name = "l")
   b = mdl.addVars(((item, element) for item in E for element in V), vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name = "b")
   b0 = mdl.addVars(((item, 0, i) for item in E for i in N), vtype=GRB.CONTINUOUS, name = "b0")
   y = mdl.addVars(((item, 0, i) for item in E for i in N), vtype=GRB.CONTINUOUS, name = "y")

   ind = mdl.addVars(((item, element) for item in E for element in V), vtype = GRB.BINARY, name = "ind")


   mdl.addConstrs((quicksum(x_e[e,(i,j)] for e in E for i in V if i!=j)== 1) for j in N)
   mdl.addConstrs(((quicksum(x_e[e,(i,j)] for i in V if i!=j)-quicksum(x_e[e,(j,i)] for i in V if i!=j) == 0) for j in V for e in E), name='s')


   mdl.addConstrs((l[(e,i)]  <= Q_EV) for i in N  for e in E)
   mdl.addConstrs((l[(e,j)]* x_e[e,(i,j)] ==  x_e[e,(i,j)]*(l[(e,i)]+q[j])) for e in E for j in N for i in V if i!=j)
   mdl.addConstrs((l[(e,0)] == 0) for e in E)


   mdl.addConstrs(((b[e,j]*x_e[e,(i,j)] == x_e[e,(i,j)] * (b[e,i]*(1-r[i])+r[i]-(a[(i,j)]/EV_velocity)*(gamma+gamma_l*l[(e,i)]))) for i in V for j in N for e in E if i!=j), name='ssss')
   mdl.addConstrs((b[e,j]*x_e[e,(i,j)] >= x_e[e,(i,j)]*(a[(j,0)]/EV_velocity)*(gamma+gamma_l*l[(e,j)])) for i in N for j in N for e in E if i!=j)
   mdl.addConstrs((b[e,0] == 1) for e in E)

   for j in N:
         for e in E:
            mdl.addGenConstrPWL(b0[e,0,j], y[e,0,j], [0, 0.8, 1], [300, 120, 0],  "myPWLConstr")

   mdl.addConstrs((z[e,j]*x_e[e,(i,j)] == x_e[e,(i,j)]*(z[e,i]+st[i]+(a[(i,j)]/EV_velocity))) for i in V for j in N for e in E if i!=j)
   mdl.addConstrs((b0[e,0,j]*x_e[e,(j,0)] == x_e[e,(j,0)] * (b[e,j] - (a[(j,0)]/EV_velocity)*(gamma+gamma_l*l[(e,j)]))) for j in N for e in E)
   mdl.addConstrs((b0[e,0,j]*x_e[e,(j,0)] >= battery_threshold*x_e[e,(j,0)]) for j in N for e in E)
   mdl.addConstrs((z[e,0] == 0) for e in E)


   mdl.addConstrs((quicksum(x_e[e,(j,0)]* (z[e,j]+y[e,0,j]+(a[(j,0)]/EV_velocity)) for j in N) <= T_max_EV) for e in E)


   mdl.modelSense = GRB.MINIMIZE
   #mdl.setObjective((quicksum(x_d[d,(0,j)]*a[(0,j)]*GV_cost for d in D for j in V if j!=0))+(quicksum(x_d[d,(j,0)]*a[(j,0)]*GV_cost for j in V for d in D if j!=0))+(quicksum(x_e[e,(i,j)]*a[(i,j)]*EV_cost for i in V for j in V for e in E if i!=j))+ 0*quicksum(z[e,j] for e in E for j in N) + 0*quicksum(x_e[e,(0,j)]*e for e in E for j in N))

   mdl.setObjective(quicksum(x_e[e,(i,j)]*a[(i,j)] for i in V for j in V for e in E if i!=j))

   #mdl.setObjective(quicksum((1-b0[e,0,j])*260*EV_cost*x_e[e,(j,0)] for j in N for e in E))


   mdl.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/optimized_routes.lp")

   mdl.Params.MIPGap = 0.001
   #mdl.params.NonConvex = 2
   #mdl.Params.TimeLimit = 2000 #seconds
   mdl.optimize()

   """ try:
      mdl.computeIIS()
      mdl.write("model.ilp")
   except: 2 """
   x_e_result = get_vars('x_e',mdl)
   z_result = get_vars('z',mdl)
   b_result = get_vars('b',mdl)
   b0_result = get_vars('b0',mdl)
   l_result = get_vars('l',mdl)
   y_result = get_vars('y',mdl)

   opt_route = []
   break_outer = False
   temp=0
   for element in V:
      for item in x_e_result:
         if x_e_result[item]>0.99 and [int(match.group()) for match in re.finditer(r'\b\d+\b', item)][2]==temp:
            tmp = [int(match.group()) for match in re.finditer(r'\b\d+\b', item)]
            opt_route.append(tmp[2])
            temp = tmp[1]
         if len(opt_route)==len(V):
            break_outer = True
            break
      if break_outer==True:
         break
   opt_route.append(0)

   for item in b0_result:
      if b0_result[item]!=0.0 and b0_result[item]!=0.8:
         rem_battery= b0_result[item]
   return opt_route, rem_battery

def preprocess_and_optimize(EV_dict):

   EV_dict_optimized = copy.deepcopy(EV_dict)
   for item in EV_dict:
      A = []
      for i, elements in enumerate(EV_dict[item]["route"]):
         num_nodes = len(elements)
         if num_nodes!=1 and num_nodes!=3 and num_nodes!=4:
            V = elements[:-1]
            A = [(j,k) for j in V for k in V  if j!=k] 
            opt_route, rem_battery = optimize_routes(A,V,[item])
            EV_dict_optimized[item]["route"].remove(elements)
            EV_dict_optimized[item]["route"].append(opt_route)
            EV_dict_optimized[item]["battery"].pop(i)
            EV_dict_optimized[item]["battery"].append([rem_battery])
   return EV_dict_optimized

def MIP_start_IFB(EV_dict_optimized, unassigned_nodes):

   x_d = {(item, element): 0 for item in D for element in A}
   x_e = {(item, element): 0 for item in E for element in A}
   z = {(item, element): 0 for item in K for element in V}
   l = {(item, element): 0 for item in K for element in V}
   b = {(item, element): 1 for item in E for element in V}
   b0 = {(item, 0, i): 1 for item in E for i in N}
   y = {(item, 0, i): 0 for item in E for i in N}
   
   for item in unassigned_nodes:
       x_d[item, (0, item)] = 1 
       x_d[item, (item, 0)] = 1
       z[item, item] = st[item] + (a[(0, item)] / GV_velocity)
   
   for items in EV_dict_optimized:
       for r in range(0,len(EV_dict_optimized[items]['route'])):
            for i in range(1, len(EV_dict_optimized[items]['route'][r]) - 1):
                if len(EV_dict_optimized[items]['route'][r])>1:
                   z[items, EV_dict_optimized[items]['route'][r][i]] = z[items, EV_dict_optimized[items]['route'][r][i - 1]] + st[i] + (a[(i - 1, i)] / EV_velocity)

   for items in EV_dict_optimized:
       for r in range(0,len(EV_dict_optimized[items]['route'])):
            for i in range(0, len(EV_dict_optimized[items]['route'][r])-1):
                if len(EV_dict_optimized[items]['route'][r])>1:
                   x_e[items, (EV_dict_optimized[items]['route'][r][i], EV_dict_optimized[items]['route'][r][i + 1])] = 1


   for items in EV_dict_optimized:
       for r in range(0,len(EV_dict_optimized[items]['route'])):
            for i in range(1, len(EV_dict_optimized[items]['route'][r]) - 1):
                if len(EV_dict_optimized[items]['route'][r])>1:
                  l[items, EV_dict_optimized[items]['route'][r][i]] = l[items, EV_dict_optimized[items]['route'][r][i-1]]+q[i]
                  #b[items, EV_dict_optimized[items]['route'][r][i]] = b[items, EV_dict_optimized[items]['route'][r][i-1]]- (a[EV_dict_optimized[items]['route'][r][i-1],EV_dict_optimized[items]['route'][r][i]]/EV_velocity)*(gamma+gamma_l*l[items,i-1]) 
   
   #for items in EV_dict_optimized:
   #    for r in range(0,len(EV_dict_optimized[items]['route'])):
   #         for i in range(1, len(EV_dict_optimized[items]['route'][r]) - 1):
   #             if len(EV_dict_optimized[items]['curr_load'][r])>1:
   #                z[items, EV_dict_optimized[items]['route'][r][i]] = z[items, EV_dict_optimized[items]['route'][r][i - 1]] + st[i] + (a[(i - 1, i)] / EV_velocity)
   
   # Convert dictionaries to DataFrames
   x_d_df = pd.DataFrame.from_dict(x_d, orient='index', columns=['Value']).reset_index()
   x_e_df = pd.DataFrame.from_dict(x_e, orient='index', columns=['Value']).reset_index()
   #z_df = pd.DataFrame.from_dict(z, orient='index', columns=['Value']).reset_index()
   l_df = pd.DataFrame.from_dict(l, orient='index', columns=['Value']).reset_index()
   #b_df = pd.DataFrame.from_dict(b, orient='index', columns=['Value']).reset_index()
   

   dataframes = {
      'x_d_df': x_d_df,
      'x_e_df': x_e_df,
      'l_df': l_df
   }
   # Loop through the dictionary and save each dataframe to a pickle file
   for filename, dataframe in dataframes.items():
      with open(f'{filename}.pkl', 'wb') as file:
         pickle.dump(dataframe, file)

def solution_cost(sol1, sol2):
    sol1_cost = sum(sol1[item]['objective'] for item in sol1)
    sol2_cost = sum(sol2[item]['objective'] for item in sol2)
    return (sol1_cost, sol1) if sol1_cost < sol2_cost else (sol2_cost, sol2)

if __name__ == "__main__":

   labels, centroids, X, node_attr = get_node_att()
      
   EV_dict, unassigned_nodes, cluster_info, nodes_original = IFB(labels, centroids, X, node_attr)
   
   visualize_routes(EV_dict)
   
   EV_dict_optimized = preprocess_and_optimize(EV_dict)

   GV_assignment_IFB, EV_assignment_IFB = EV_GV_assignment(EV_dict_optimized, unassigned_nodes)

   #x_d, x_e, z, l, b = MIP_start_IFB(EV_dict_optimized, unassigned_nodes)
   MIP_start_IFB(EV_dict_optimized, unassigned_nodes)
   2