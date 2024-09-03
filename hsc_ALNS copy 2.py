import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import gurobipy as gp
import networkx as nx
from gurobipy import Model, GRB, quicksum
import copy
import time
import re
from matplotlib.lines import Line2D
import random
import math
import itertools
from itertools import combinations


rnd = np.random
rnd.seed(10)

n = 8
grid_size = 50                                                                        #number of clients
xc = np.random.uniform(low=- grid_size/2, high=grid_size/2, size=n+1)
yc = np.random.uniform(low=-grid_size/2, high=grid_size/2, size=n+1)
xc[0]=0
yc[0]=0

N = [i for i in range(1,n+1)]                                                #set of customer nodes
V = [0] + N                                                                  #set of all nodes (customer+depot)
Q_EV = 10                                                                    #capacity of each EV
Q_GV = 15                                                                    #capacity of each GV
q = {i: rnd.randint(1,7) for i in N}                                         #demand for customers
#q = {i: np.random.choice([1, 20]) for i in N}

total_dem = sum(q)                                                           #total demand
num_EV = 3 #int(n*0.2) 
num_clusters = 4 # int(total_dem/(num_EV*Q_EV))            
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
battery_threshold = 0.1
alpha = 0.1
arc_set = [(i,j) for i in N for j in N  if i!=j]
dist = {(i,j): np.hypot(xc[i]-xc[j], yc[i]- yc[j]) for (i,j) in arc_set}              #eucledian distance
time_limit = 60

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
      x[item] = x[item]*(max_dem/z[item])
      y[item] = y[item]*(max_dem/z[item])
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
      rem_battery = rem_battery + battery_consumption
      return rem_battery, current_load, flag_b, flag_l 
   if (rem_battery_home < battery_threshold) and (current_load <= Q_EV):
      current_load -= q[y]
      rem_battery = rem_battery + battery_consumption
      flag_b = False
      return rem_battery, current_load, flag_b, flag_l
   if (rem_battery_home < battery_threshold) and (current_load > Q_EV):
      current_load -= q[y]
      rem_battery = rem_battery + battery_consumption
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
      rem_time = rem_time + time_consumption
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

""" GV_assignment, EV_assignment = EV_GV_assignment(EV_dict, unassigned_nodes)

EV_dict_copy = copy.deepcopy(EV_dict)

for item in EV_dict:
   for i, elements in enumerate(EV_dict[item]["route"]):
      num_nodes = len(elements)
      if num_nodes==1 or num_nodes==3:
         EV_dict_copy[item]["battery"].remove(EV_dict[item]["battery"][i])
         EV_dict_copy[item]["curr_load"].remove(EV_dict[item]["curr_load"][i])
         EV_dict_copy[item]["route"].remove(elements)

cost_IFB, miles_IFB, GV_assignment, EV_assignment = cost_calculation(GV_assignment,EV_assignment,EV_dict_copy)

no_collab_total_miles=0
no_collab_total_cost=0
for i in range(1,n+1):
   no_collab_total_miles += a[(0,i)]*2
for i in range(1,n+1):
   no_collab_total_cost += a[(0,i)]*GV_cost + a[(i,0)]*GV_cost*q[i]
miles_saved_IFB = no_collab_total_miles-miles_IFB
percentage_miles_saved_IFB = (miles_saved_IFB/(no_collab_total_miles))*100
cost_saved_IFB = no_collab_total_cost-cost_IFB
percentage_cost_saved_IFB = (cost_saved_IFB/no_collab_total_cost)*100 """

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

def convert_solution_to_EVdict(solution):

   EV_dict = {}
   for key, value in solution.items():
      if value['type']!='GV':  # Skip GV entries
         ev_type = value['type']
         EV_dict[ev_type] = {'route': [],'battery': [],'time': [],'curr_load': []}
   for key, value in solution.items():
      if value['type']!='GV':  # Skip GV entries
         ev_type = value['type']         
         EV_dict[ev_type]['route'].append(eval(key))
         EV_dict[ev_type]['battery'].append([value.get('battery', None)])
         EV_dict[ev_type]['time'].append(value.get('op_time', None))
         EV_dict[ev_type]['curr_load'].append(None)  # You can add the current load value here if needed

   return EV_dict


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

def final_tsp_optimization(best_overall_solution):
   best_EV_dict = convert_solution_to_EVdict(best_overall_solution)
   best_optimized_EV_dict = preprocess_and_optimize(best_EV_dict)
   unassigned_nodes = [eval(item)[1] for item in best_overall_solution if len(eval(item))==3]
   GV_assignment, EV_assignment = EV_GV_assignment(best_optimized_EV_dict, unassigned_nodes)

   cost, miles, GV_assignment, EV_assignment = cost_calculation(GV_assignment,EV_assignment,best_optimized_EV_dict)

   if objective=="miles":
      return miles,best_optimized_EV_dict 
   else: return cost,best_optimized_EV_dict 


def set_ALNS_obj(EV_assignment_IFB, GV_assignment_IFB, objective):
   for item in EV_assignment_IFB:
      for element in EV_assignment_IFB[item]:
         if objective=='miles':
            EV_assignment_IFB[item][element]['objective'] = EV_assignment_IFB[item][element]['miles']
         else:
            EV_assignment_IFB[item][element]['objective'] = EV_assignment_IFB[item][element]['cost']
   for item in GV_assignment_IFB:
      if objective=='miles':
         GV_assignment_IFB[item]['objective'] = GV_assignment_IFB[item]['miles']
      else:
         GV_assignment_IFB[item]['objective'] = GV_assignment_IFB[item]['cost']
   if objective=='cost':
      IFB_objective = miles_IFB_opt
   else:
      IFB_objective = cost_IFB_opt
   current_best_local_search = IFB_objective
   return EV_assignment_IFB, GV_assignment_IFB, IFB_objective, current_best_local_search

def all_routes_construction(EV_assignment_IFB, GV_assignment_IFB):
   all_routes = {}
   EV_only_route = {}
   for item in EV_assignment_IFB:
      for element in EV_assignment_IFB[item]:
         all_routes[element] = EV_assignment_IFB[item][element]
         EV_only_route[element] = EV_assignment_IFB[item][element]
         all_routes[element]['type'] = item
   for item in GV_assignment_IFB:
      i = [0,item,0]
      all_routes[f'{i}'] = GV_assignment_IFB[item]
      all_routes[f'{i}']['type'] = 'GV'
   return all_routes

def convert_keys_into_array(k):
   k = k[1:-1].split(', ')
   k = [int(element) for element in k]
   return k

def check_score(temp_route):
   battery = 1
   q[0] = 0
   current_load = 0
   op_time = T_max_EV
   score = 0
   miles = 0
   for i in range(0,len(temp_route)-1):
      current_load += q[temp_route[i]]
      battery = battery - (a[(temp_route[i],temp_route[i+1])]/EV_velocity)*(gamma+gamma_l*current_load)
      op_time = op_time - (st[temp_route[i]]+(a[(temp_route[i],temp_route[i+1])]/EV_velocity))
   if objective=='cost':
      score = 260*EV_cost*(1-battery)
   else:
      for i in range(0,len(temp_route)-1):
         score = score + a[(temp_route[i],temp_route[i+1])]
   for i in range(0,len(temp_route)-1):
      miles = miles + a[(temp_route[i],temp_route[i+1])]
   cost = 260*EV_cost*(1-battery)
   return score, current_load, battery, op_time, miles , cost 

def check_GV_score(temp_route):
   miles = 2*a[0,temp_route[1]]
   cost = a[(0,temp_route[1])]*GV_cost + a[(temp_route[1],0)]*GV_cost*q[temp_route[1]]
   if objective=='cost':
      score = cost
   else: score = miles
   return score, miles , cost

def feasibility_check(temp_route):
   if type(temp_route)==str:
      temp_route = eval(temp_route)
   battery = 1
   q[0] = 0
   current_load = 0
   op_time = 0
   score = 0
   miles = 0
   for i in range(0,len(temp_route)-1):
      current_load += q[temp_route[i]]
      if current_load > Q_EV:
         return score, None, None, None, None, None
      else:
         battery = battery - (a[(temp_route[i],temp_route[i+1])]/EV_velocity)*(gamma+gamma_l*current_load)
         if battery < battery_threshold:
            return score, None, None, None, None, None
         else:
            op_time = op_time + (st[temp_route[i]]+(a[(temp_route[i],temp_route[i+1])]/EV_velocity))
         if op_time > T_max_EV:
            return score, None, None, None, None, None
         
   #update score if feasible
   if objective=='cost':
      score = 260*EV_cost*(1-battery)
   else:
      for i in range(0,len(temp_route)-1):
         score = score + a[(temp_route[i],temp_route[i+1])]
   for i in range(0,len(temp_route)-1):
      miles = miles + a[(temp_route[i],temp_route[i+1])]
   cost = 260*EV_cost*(1-battery)
   return score, current_load, battery, op_time, miles, cost  

def EV_multiple_tour_feasibility_check(all_routes, new_route, EV, k1, f=2):
   #o = 0
   #temp_route = [0, 49, 9, 38, 48, 0]
   #for i in range(0,5):
   #   o += (st[temp_route[i]]+(a[(temp_route[i],temp_route[i+1])]/EV_velocity))
   #b = 
   #if b<0.8:
   #   a=225*(0.8-b) + 120
   #else: 
   #   a = (b-0.8)*600 + 180
   tours = []
   charging_time = 0
   for items in all_routes:
      if all_routes[items]['type']==EV:
         tours.append(convert_keys_into_array(items))
   tours.append(new_route)
   try:
      tours.remove(k1)
   except: pass
   tours = [list(t) for t in set(tuple(row) for row in tours)]
   tour_info = {}
   for item in tours:
      score, current_load, battery, op_time, miles, cost = feasibility_check(item)
      if score!=0:
         tour_info[f'{item}'] = {}
         tour_info[f'{item}']['cost'] = cost
         tour_info[f'{item}']['objective'] = score
         tour_info[f'{item}']['battery'] = battery
         tour_info[f'{item}']['miles'] = miles
         tour_info[f'{item}']['op_time'] = op_time
         tour_info[f'{item}']['type'] = EV

      else:
         if f==5:
            pass 
         return False, 0
   sorted_data = dict(sorted(tour_info.items(), key=lambda item: item[1]['battery'], reverse=True))
   total_op_time = 0
   for items in sorted_data:
      total_op_time += sorted_data[items]["op_time"]
   if total_op_time>T_max_EV:
      if f==5:
         pass
      return False, 0
   #else:
   #   for index, key in enumerate(tour_info):
   #      if index < len(tour_info) - 1:
   #         if tour_info[key]['battery']<0.8:
   #            charging_time+=225*(0.8-tour_info[key]['battery']) + 120
   #         else: 
   #            charging_time+= (tour_info[key]['battery']-0.8)*600 + 180
   #   
      #lowest_battery = min(tour_info.values(), key=lambda x: x['battery'])['battery']
      #lowest_battery =  tour_info[list(tour_info.keys())[-1]]['battery']
      #if lowest_battery <0.8:
      #   charging_time-=225*(0.8-lowest_battery ) + 120
      #else: 
      #   charging_time-= (lowest_battery -0.8)*600 + 180 

   #if total_op_time + charging_time > T_max_EV:
   #   if f==5:
   #      pass
   #   return False, 0
   else:   #it is feasible
      total_op_time_with_charging = 0
      for key in tour_info:
         if tour_info[key]['battery']<0.8:
            tour_info[key]['charging_time']=225*(0.8-tour_info[key]['battery']) + 120
         else: 
            tour_info[key]['charging_time'] = (tour_info[key]['battery']-0.8)*600 + 180
         total_op_time_with_charging+= tour_info[key]['charging_time'] + tour_info[key]['op_time']
      if total_op_time_with_charging - max(tour_info.values(), key=lambda x: x['charging_time'])['charging_time']>T_max_EV+1:
         if f==5:
               pass
         return False, 0

      total_score = tour_info[f'{new_route}']['objective']
      return tour_info, total_score

def shift(all_routes):
   shift_solution = copy.deepcopy(all_routes)
   k1 = random.choice(list(all_routes.keys()))
   k2 = list(all_routes.keys())
   k2.remove(k1)
   k2 = random.choice(k2)
   k1 = convert_keys_into_array(k1)
   k2 = convert_keys_into_array(k2)
   n1 = random.choice(k1[1:-1])
   n2 = random.choice(k2[1:-1])


   # for testing k1 = [0,3,0]
   # for testing k2 = [0,7,0]
   # for testing n1 = random.choice(k1[1:-1])
   # for testing n2 = random.choice(k2[1:-1])

   w = q[n2]
   for i in k1[1:-1]:
      w+=q[i]
   if w > Q_EV:
      return False, False, 0

   score_dict = {}
   shift_solution_copy = copy.deepcopy(shift_solution)

   if all_routes[f'{k1}']['type'] == 'GV' and all_routes[f'{k2}']['type'] == 'GV':
      for i in (1,len(k1)-1):
         temp_route = copy.deepcopy(k1)
         temp_route.insert(i,n2)
         for item in E:
            tour_info, total_score = EV_multiple_tour_feasibility_check(all_routes, temp_route, item, k1)
            if tour_info:
               score_dict[(f'{temp_route}',item)] = tour_info[f'{temp_route}']
               score_dict[(f'{temp_route}',item)]['type'] = item
      score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1]['objective'], reverse=False))
         
      if len(score_dict)!=0:
         shift_solution_copy[next(iter(score_dict))[0]] = score_dict[next(iter(score_dict))]
         score = score_dict[next(iter(score_dict))]['objective'] - shift_solution_copy[f'{k2}']['objective'] - shift_solution_copy[f'{k1}']['objective']
         shift_solution_copy.pop(f'{k1}')
         shift_solution_copy.pop(f'{k2}')
      else: return False, False, 0
   #GV becomes an EV. Choose the best EV. The second GV disappears
     
   #for testing k1 = [0,9,5,0]
   #for testing k2 = [0,10,0]
   #for testing n1 = random.choice(k1[1:-1])
   #for testing n2 = random.choice(k2[1:-1])
   #for testing q = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2}
      
   if (all_routes[f'{k1}']['type'] == 'GV' and all_routes[f'{k2}']['type'] != 'GV') or (all_routes[f'{k1}']['type'] != 'GV' and all_routes[f'{k2}']['type'] != 'GV'):
      #GV becomes an EV. It might be the same EV. Choose the best EV. The second on can also become a GV
      temp_route2 = copy.deepcopy(k2)
      temp_route2.remove(n2)
      if len(temp_route2)>3:
         score2, current_load2, battery2, op_time2, miles2, cost2 = check_score(temp_route2)
         shift_solution_copy[f'{temp_route2}'] = {}
         shift_solution_copy[f'{temp_route2}']['type'] = all_routes[f'{k2}']['type']
         shift_solution_copy[f'{temp_route2}']['cost'] = cost2
         shift_solution_copy[f'{temp_route2}']['miles'] = miles2
         shift_solution_copy[f'{temp_route2}']['battery'] = battery2
         shift_solution_copy[f'{temp_route2}']['objective'] = score2
         shift_solution_copy[f'{temp_route2}']['op_time'] = op_time2  ###
      else:
         score2, miles2 , cost2 = check_GV_score(temp_route2)
         shift_solution_copy[f'{temp_route2}'] = {}
         shift_solution_copy[f'{temp_route2}']['type'] = 'GV'
         shift_solution_copy[f'{temp_route2}']['cost'] = cost2
         shift_solution_copy[f'{temp_route2}']['miles'] = miles2
         shift_solution_copy[f'{temp_route2}']['objective'] = score2
      shift_solution_copy.pop(f'{k2}')

      for i in (1,len(k1)-1):
         temp_route = copy.deepcopy(k1)
         temp_route.insert(i,n2)
         for item in E:
            tour_info, total_score = EV_multiple_tour_feasibility_check(shift_solution_copy, temp_route, item, k1)
            if tour_info:
               score_dict[(f'{temp_route}',item)] = tour_info[f'{temp_route}']
               score_dict[(f'{temp_route}',item)]['type'] = item
      score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1]['objective'], reverse=False))
         
      if len(score_dict)!=0:
         shift_solution_copy[next(iter(score_dict))[0]] = score_dict[next(iter(score_dict))]
         score = score_dict[next(iter(score_dict))]['objective'] + shift_solution_copy[f'{temp_route2}']['objective'] - all_routes[f'{k2}']['objective'] - all_routes[f'{k1}']['objective']
         shift_solution_copy.pop(f'{k1}')
      else: return False, False, 0


   if all_routes[f'{k1}']['type'] != 'GV' and all_routes[f'{k2}']['type'] == 'GV':
   #EV remains the same, GV disappears
      for i in (1,len(k1)-1):
         temp_route = copy.deepcopy(k1)
         temp_route.insert(i,n2)
         tour_info, total_score = EV_multiple_tour_feasibility_check(all_routes, temp_route, all_routes[f'{k1}']['type'], k1)
         if tour_info:
            score_dict[f'{temp_route}'] = tour_info[f'{temp_route}']
            score_dict[f'{temp_route}']['type'] = all_routes[f'{k1}']['type']
      if len(score_dict)==0:
         return False, False, 0
      score_dict = dict(sorted(score_dict.items(), key=lambda item: item[1]['objective'], reverse=False))
      score = score_dict[next(iter(score_dict))]['objective'] - shift_solution_copy[f'{k2}']['objective'] - shift_solution_copy[f'{k1}']['objective']
      shift_solution_copy.pop(f'{k1}')
      shift_solution_copy.pop(f'{k2}')
      shift_solution_copy[next(iter(score_dict))] = score_dict[next(iter(score_dict))]

   if score < 0:
      return shift_solution_copy, True, score
   elif score > 0: 
      return shift_solution_copy, False, score

def switch(all_routes):
   switch_solution = copy.deepcopy(all_routes)
   k1 = random.choice(list(all_routes.keys()))
   k2 = list(all_routes.keys())
   k2.remove(k1)
   k2 = random.choice(k2)
   k1 = convert_keys_into_array(k1)
   k2 = convert_keys_into_array(k2)
   n1 = random.choice(k1[1:-1])
   n2 = random.choice(k2[1:-1])

   q[0] = 0
   #cannot be the same EV
   if all_routes[f'{k1}']['type'] != 'GV' and all_routes[f'{k2}']['type'] != 'GV':
      if all_routes[f'{k1}']['type'] != all_routes[f'{k2}']['type']:
         score_dict = {}
         score_dict2 = {}
         score_dict3 = {}
         corresponding_route = {}
         switch_solution_copy = copy.deepcopy(switch_solution) 
         for i in (1,len(k1)-2):
            temp_route = copy.deepcopy(k1)
            temp_route2 = copy.deepcopy(k2)
            temp_route2[k2.index(n2)] = k1[i]
            temp_route[i] = n2
            corresponding_route[f'{temp_route}'] = f'{temp_route2}'
            w1 = sum(q[key] for key in temp_route[1:-1])
            w2 = sum(q[key] for key in temp_route2[1:-1])

            if w1 > Q_EV or w2 > Q_EV:
               return False, False, 0

            tour_info, total_score = EV_multiple_tour_feasibility_check(all_routes, temp_route, all_routes[f'{k1}']['type'], k1)
            tour_info2, total_score2 = EV_multiple_tour_feasibility_check(all_routes, temp_route2, all_routes[f'{k2}']['type'], k2)
            if tour_info and tour_info2:
               score_dict[f'{temp_route}'] = tour_info[f'{temp_route}']
               score_dict[f'{temp_route}']['type'] = all_routes[f'{k1}']['type']
               score_dict2[f'{temp_route2}'] = tour_info2[f'{temp_route2}']
               score_dict2[f'{temp_route2}']['type'] = all_routes[f'{k2}']['type']
         if len(score_dict)==0 or len(score_dict2)==0:
            return False, False, 0
         for element in score_dict:
            score_dict3[f'{element}']= score_dict[element]['objective'] + score_dict2[corresponding_route[f'{element}']]['objective']

         score_dict3 = dict(sorted(score_dict3.items(), key=lambda item: item[1], reverse=False))

         switch_solution_copy[next(iter(score_dict3))] = score_dict[next(iter(score_dict3))]
         switch_solution_copy[corresponding_route[next(iter(score_dict3))]] = score_dict2[corresponding_route[next(iter(score_dict3))]]

         score = score_dict[next(iter(score_dict3))]['objective'] + switch_solution_copy[corresponding_route[next(iter(score_dict3))]]['objective'] - switch_solution_copy[f'{k2}']['objective'] - switch_solution_copy[f'{k1}']['objective']
         switch_solution_copy.pop(f'{k1}')
         switch_solution_copy.pop(f'{k2}')

         if score < 0:
            return switch_solution_copy, True, score
         elif score>0: 
            return switch_solution_copy, False, score
   

   if all_routes[f'{k1}']['type'] != 'GV' and all_routes[f'{k2}']['type'] == 'GV':
      score_dict = {}
      score_dict2 = {}
      score_dict3 = {}
      corresponding_route = {}
      switch_solution_copy = copy.deepcopy(switch_solution) 
      for i in (1,len(k1)-2):
         temp_route = copy.deepcopy(k1)
         temp_route2 = copy.deepcopy(k2)
         temp_route2[1] = k1[i]
         temp_route[i] = n2
         corresponding_route[f'{temp_route}'] = f'{temp_route2}'
         w1 = sum(q[key] for key in temp_route[1:-1])
         w2 = temp_route2[1]

         if w1 > Q_EV or w2 > Q_EV:
            return False, False, 0

         tour_info, total_score = EV_multiple_tour_feasibility_check(all_routes, temp_route, all_routes[f'{k1}']['type'], k1)
         score_GV, miles_GV , cost_GV =  check_GV_score(temp_route2)
         if tour_info:
            score_dict[f'{temp_route}'] = tour_info[f'{temp_route}']
            score_dict[f'{temp_route}']['type'] = all_routes[f'{k1}']['type']
            score_dict2[f'{temp_route2}'] = {'cost':cost_GV,'miles':miles_GV,'objective':score_GV,'type':'GV'}
      if len(score_dict)==0 or len(score_dict2)==0:
         return False, False, 0
      for element in score_dict:
         score_dict3[f'{element}']= score_dict[element]['objective'] + score_dict2[corresponding_route[f'{element}']]['objective']
      score_dict3 = dict(sorted(score_dict3.items(), key=lambda item: item[1], reverse=False))
      switch_solution_copy[next(iter(score_dict3))] = score_dict[next(iter(score_dict3))]
      switch_solution_copy[corresponding_route[next(iter(score_dict3))]] = score_dict2[corresponding_route[next(iter(score_dict3))]]
      score = score_dict[next(iter(score_dict3))]['objective'] + switch_solution_copy[corresponding_route[next(iter(score_dict3))]]['objective'] - switch_solution_copy[f'{k2}']['objective'] - switch_solution_copy[f'{k1}']['objective']
      switch_solution_copy.pop(f'{k1}')
      switch_solution_copy.pop(f'{k2}')
      
      if score < 0:
         return switch_solution_copy, True, score
      elif score>0: 
         return switch_solution_copy, False, score
   else: return False, False, 0

def two_opt(all_routes):
   two_opt_solution = copy.deepcopy(all_routes)
   EV_only_route = {}
   for items in all_routes:
      if all_routes[items]['type'] != 'GV':
         EV_only_route[items] = all_routes[items]
   k1 = random.choice(list(EV_only_route.keys()))
   k1 = convert_keys_into_array(k1)
   n1 = random.choice(k1[1:-1])
   k2 = copy.deepcopy(k1)
   k2.remove(n1)
   n2 = random.choice(k2[1:-1])
   temp_route = copy.deepcopy(k1)
   temp_route[k1.index(n1)] = n2
   temp_route[k1.index(n2)] = n1
 
   tour_info, total_score = EV_multiple_tour_feasibility_check(all_routes, temp_route, all_routes[f'{k1}']['type'], k1)
   if tour_info:
      two_opt_solution[f'{temp_route}'] = tour_info[f'{temp_route}']
      two_opt_solution[f'{temp_route}']['type'] = all_routes[f'{k1}']['type']
      score = two_opt_solution[f'{temp_route}']['objective'] - two_opt_solution[f'{k1}']['objective']
      two_opt_solution.pop(f'{k1}')
      if score < 0:
         return two_opt_solution, True, score
      elif score>=0: 
         return two_opt_solution, False, score
   else: return False, False, 0

def two_opt_prime(all_routes):
   two_opt_prime_solution = copy.deepcopy(all_routes)
   EV_only_route = {}
   for items in all_routes:
      if all_routes[items]['type'] != 'GV':
         EV_only_route[items] = all_routes[items]
   if len(EV_only_route)>1:
      k1 = random.choice(list(EV_only_route.keys()))
      k2 = list(EV_only_route.keys())
      k2.remove(k1)
      k2 = random.choice(k2)
      k1 = convert_keys_into_array(k1)
      k2 = convert_keys_into_array(k2)
      n1 = random.choice(range(1,len(k1)-1))
      n2 = random.choice(range(1,len(k2)-1))
      temp_route = k1[:n1] + k2[n2:]
      temp_route2= k2[:n2] + k1[n1:]
   else: return False, False, 0

   w1 = sum(q[key] for key in temp_route[1:-1])
   w2 = sum(q[key] for key in temp_route2[1:-1])
   if w1 > Q_EV or w2 > Q_EV:
      return False, False, 0
   
   if all_routes[f'{k1}']['type'] != all_routes[f'{k2}']['type']:
      if len(temp_route)==3:
         tour_info, total_score = EV_multiple_tour_feasibility_check(all_routes, temp_route2, all_routes[f'{k2}']['type'], k2)
         if tour_info:
            two_opt_prime_solution[f'{temp_route2}'] = tour_info[f'{temp_route2}']
            two_opt_prime_solution[f'{temp_route2}']['type'] = all_routes[f'{k2}']['type']
            score_GV, miles_GV , cost_GV =  check_GV_score(temp_route)
            two_opt_prime_solution[f'{temp_route}'] = {'cost':cost_GV,'miles':miles_GV,'objective':score_GV,'type':'GV'}
            score = two_opt_prime_solution[f'{temp_route}']['objective'] + two_opt_prime_solution[f'{temp_route2}']['objective'] - two_opt_prime_solution[f'{k2}']['objective'] - two_opt_prime_solution[f'{k1}']['objective']
            if temp_route!=k2:
               two_opt_prime_solution.pop(f'{k1}')
               two_opt_prime_solution.pop(f'{k2}')
            return two_opt_prime_solution , True, score
         else: return False, False, 0
            
      elif len(temp_route2)==3:
         tour_info, total_score = EV_multiple_tour_feasibility_check(all_routes, temp_route, all_routes[f'{k1}']['type'], k1)
         if tour_info:
            two_opt_prime_solution[f'{temp_route}'] = tour_info[f'{temp_route}']
            two_opt_prime_solution[f'{temp_route}']['type'] = all_routes[f'{k1}']['type']
            score_GV, miles_GV , cost_GV =  check_GV_score(temp_route2)
            two_opt_prime_solution[f'{temp_route2}'] = {'cost':cost_GV,'miles':miles_GV,'objective':score_GV,'type':'GV'}
            score = two_opt_prime_solution[f'{temp_route}']['objective'] + two_opt_prime_solution[f'{temp_route2}']['objective'] - two_opt_prime_solution[f'{k2}']['objective'] - two_opt_prime_solution[f'{k1}']['objective']
            if temp_route!=k2:
               two_opt_prime_solution.pop(f'{k1}')
               two_opt_prime_solution.pop(f'{k2}')
            return two_opt_prime_solution , True, score
         else: return False, False, 0
      
      elif len(temp_route)!=3 and len(temp_route2)!=3:
         tour_info, total_score = EV_multiple_tour_feasibility_check(all_routes, temp_route, all_routes[f'{k1}']['type'], k1)
         tour_info2, total_score2 = EV_multiple_tour_feasibility_check(all_routes, temp_route2, all_routes[f'{k2}']['type'], k2)
         if tour_info and tour_info2:
            two_opt_prime_solution[f'{temp_route}'] = tour_info[f'{temp_route}']
            two_opt_prime_solution[f'{temp_route}']['type'] = all_routes[f'{k1}']['type']
            two_opt_prime_solution[f'{temp_route2}'] = tour_info2[f'{temp_route2}']
            two_opt_prime_solution[f'{temp_route2}']['type'] = all_routes[f'{k2}']['type']
            score = two_opt_prime_solution[f'{temp_route}']['objective'] + two_opt_prime_solution[f'{temp_route2}']['objective'] - two_opt_prime_solution[f'{k2}']['objective'] - two_opt_prime_solution[f'{k1}']['objective']
            if temp_route!=k2:
               two_opt_prime_solution.pop(f'{k1}')
               two_opt_prime_solution.pop(f'{k2}')
            return two_opt_prime_solution , True, score
         else: return False, False, 0
   
   else: return False, False, 0

   #tour_info, total_score, total_op_time = EV_multiple_tour_feasibility_check(all_routes, temp_route, all_routes[f'{k1}']['type'], k1)
   #two_opt_prime_solution[f'{temp_route}'] = tour_info[f'{temp_route}']
   #two_opt_prime_solution[f'{temp_route}']['type'] = all_routes[f'{k1}']['type']
   #score = two_opt_prime_solution[f'{temp_route}']['objective'] - two_opt_prime_solution[f'{k1}']['objective']
   #two_opt_prime_solution.pop(f'{k1}')
   #if score < 0:
   #   return two_opt_prime_solution, True, score
   #elif score>0: 
   #   return two_opt_prime_solution, False, 0

def roulette_wheel_selection(items, weights):
    total_weight = sum(weights)
    pick = random.uniform(0, total_weight)
    current_weight = 0
    for item, weight in zip(items, weights):
        current_weight += weight
        if current_weight >= pick:
            return item

def sim_anneal(score,r1,T,alpha1,solution,best_solution,local_incumbent_solution):
   if score > 0:
      #sim_anneal
      p = (math.e)**(-(score)/T)
      if random.random() < p:
         local_incumbent_solution = copy.deepcopy(solution)
   elif score<0: 
      local_incumbent_solution = copy.deepcopy(solution)
   
   obj_sol = 0
   obj_best = 0
   for items in solution:
      obj_sol += solution[items]['objective']
   for items in best_solution:
      obj_best += best_solution[items]['objective']
   if obj_sol < obj_best:
      best_solution = copy.deepcopy(solution)
   T = alpha1*T
   
   return best_solution, local_incumbent_solution, T

def ALNS_local_search(EV_dict, EV_assignment, GV_assignment, local_search_iterations, all_routes):
   local_incumbent_solution = copy.deepcopy(all_routes)
   best_solution = copy.deepcopy(all_routes)
   r1 = 0.5
   T = 10
   alpha1 = 0.9999
   i = 0
   local_search_weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
   local_search_used = np.zeros(4)

   for i in range(0, local_search_iterations):
      local_search = roulette_wheel_selection(['shift','switch','two_opt','two_opt_prime'], local_search_weights)

      if local_search=='shift':
         shift_solution, better, score = shift(local_incumbent_solution)
         if score!=0:
            best_solution, local_incumbent_solution,T=sim_anneal(score,r1,T,alpha1,shift_solution,best_solution,local_incumbent_solution)
            local_search_used[0]+=1
            #i=i+1
         if score < 0:
            local_search_weights[0] = local_search_weights[0]*(1-r1)+r1*(abs(score)/local_search_used[0])
      
      elif local_search=='switch':
         switch_solution, better, score = switch(local_incumbent_solution)
         if score!=0:
            best_solution, local_incumbent_solution, T=sim_anneal(score,r1,T,alpha1,switch_solution,best_solution,local_incumbent_solution)
            local_search_used[1]+=1
            #i=i+1
         if score < 0:
            local_search_weights[1] = local_search_weights[1]*(1-r1)+r1*(abs(score)/local_search_used[1])
      
      elif local_search=='two_opt':
         two_opt_solution, better, score = two_opt(local_incumbent_solution)
         if score!=0:
            best_solution, local_incumbent_solution, T = sim_anneal(score,r1,T,alpha1,two_opt_solution,best_solution,local_incumbent_solution)
            local_search_used[2]+=1
            #i=i+1
         if score < 0:
            local_search_weights[2] = local_search_weights[2]*(1-r1)+r1*(abs(score)/local_search_used[2])
      
      elif local_search=='two_opt_prime':
         two_opt_prime_solution, better, score = two_opt_prime(local_incumbent_solution)
         if score!=0:
            best_solution, local_incumbent_solution, T = sim_anneal(score,r1,T,alpha1,two_opt_prime_solution,best_solution,local_incumbent_solution)
            local_search_used[3]+=1
            #i=i+1
         if score < 0:
            local_search_weights[3] = local_search_weights[3]*(1-r1)+r1*(abs(score)/local_search_used[3])
   return best_solution

def destroy(original_solution, num_destroy):
   random_solution = copy.deepcopy(original_solution)
   destroy_operator_weights = np.array([1, 1, 1], dtype=float)
   destroy_operator = roulette_wheel_selection(['random','related','worst'], destroy_operator_weights)
   nodes_removed = []
   if destroy_operator=='random':
      while len(nodes_removed)<num_destroy:
         routes = []
         for item in random_solution:         #only consider those EV routes such that deleting a node still makes it an EV route, does not become a GV route 
            r = eval(item)
            if len(r)!=4:
               routes.append(r)
         old_route = random.choice(routes)
         #routes.remove(old_route)
         if len(old_route)==3:  #if its a GV, then delete the route
            try:
               del random_solution[f'{old_route}']
               nodes_removed.append(old_route[1])
            except: pass
            #continue
         elif len(old_route)>4:  #only consider those EV routes such that deleting a node still makes it an EV route, does not become a GV route 
            new_route = copy.deepcopy(old_route)
            node_index_removed = random.randint(1, len(old_route) - 2)  # Random index from 1 to len(arr) - 2
            nodes_removed.append(new_route[node_index_removed])
            new_route.pop(node_index_removed)
            type = random_solution[f'{old_route}']['type']
            tour_info, aa = EV_multiple_tour_feasibility_check(random_solution, new_route, type, old_route, f=5)
            #try:
            del random_solution[f'{old_route}']
            #except: pass
            for item in tour_info:
               random_solution[item] = tour_info[item]
               random_solution[item]['type'] =  type
      return random_solution, nodes_removed
   
   
   if destroy_operator=='related':
      related_solution = copy.deepcopy(original_solution)
      same_route = []
      nodes_removed = []
      routes = []
      nodes_del = []
      for item in related_solution:         #only consider those EV routes such that deleting a node still makes it an EV route, does not become a GV route 
         r = eval(item)
         if len(r)!=4:
            routes.append(r)
         else: 
            for j in r:
               nodes_del.append(j)
      nodes_del = list(set(nodes_del))
      #chek if the pair belongs to the same route or not
      for elem in routes:
         if len(elem)>3:
            same_route.append(list(combinations(elem[1:-1], 2)))
      same_route = [item for sublist in same_route for item in sublist]
      same_route.extend([(t[1], t[0]) for t in same_route])



      while len(nodes_removed)<num_destroy:
         start_time = time.time()
         d = {}
         relatedness_measure = {}
         outer_break = False
         while True:
            first_node = random.choice(random.choice(routes))   #randomly choose a node from filtered routes
            if first_node!=0 and first_node not in nodes_removed and first_node not in nodes_del:
               #nodes_removed.append(first_node)
               break
         for items in dist:
            if items[0]==first_node:
               d[items] = dist[items]
         d_max = max(d.values())


         for items in d:
            if items not in same_route:
               relatedness_measure[items] = 1/((d[items]/d_max))
         relatedness_measure = dict(sorted(relatedness_measure.items(), key=lambda x: x[1], reverse=True))

         for item in relatedness_measure:
            flag = 1
            combo = []
            combo_list = nodes_removed + [item[0]] + [item[1]]
            combo = list(combinations(combo_list, 2))

            for elem in combo:
               if elem in same_route or item[1] in nodes_del or item[1] in nodes_removed:
                  flag = 0
                  break
            if flag==1:
               nodes_removed.append(item[0])
               nodes_removed.append(item[1])
               del relatedness_measure[item]
               outer_break = True
            if outer_break:
               break
         if time.time()-start_time>60:
            break



  

      for items in nodes_removed:
         for elem in related_solution:
            if items in eval(elem):
               old_route=eval(elem)
               if len(old_route)==3:  #if its a GV, then delete the route
                  try:
                     del related_solution[f'{old_route}']
                     break
                  except: pass
               elif len(old_route)>4:  #only consider those EV routes such that deleting a node still makes it an EV route, does not become a GV route 
                  new_route = copy.deepcopy(old_route)
                  new_route.remove(items)
                  type = related_solution[f'{old_route}']['type']
                  tour_info, aa = EV_multiple_tour_feasibility_check(related_solution, new_route, type, old_route, f=5)
                  try:
                     del related_solution[f'{old_route}']
                  except: pass
                  for item in tour_info:
                     related_solution[item] = tour_info[item]
                     related_solution[item]['type'] =  type
                  break
      return related_solution, nodes_removed
   

   if destroy_operator=='worst':
      worst_solution = copy.deepcopy(original_solution)
      routes = []
      same_route = []
      for item in original_solution:         #only consider those EV routes such that deleting a node still makes it an EV route, does not become a GV route 
         r = convert_keys_into_array(item)
         if len(r)!=4:
            routes.append(r)
      for elem in routes:
         if len(elem)>3:
            same_route.append(list(combinations(elem[1:-1], 2)))
      same_route = [item for sublist in same_route for item in sublist]
      same_route.extend([(t[1], t[0]) for t in same_route])
      nodes_removed = []
      detour = {}
      for item in routes:
         for i,j in enumerate(item):
            if j!=0:
               detour[j] = a[(item[i-1],j)] + a[(j,item[i+1])] - a[(item[i-1],item[i+1])]
      detour = dict(sorted(detour.items(), key=lambda x: x[1], reverse=True))
      
      while len(nodes_removed)<num_destroy:
         start = time.time()
         if not detour:
            break
         for item in detour:
            flag = 1
            combo = []
            combo_list = nodes_removed + [item]
            combo = list(combinations(combo_list, 2))
            if len(combo)==0:
               nodes_removed.append(item)
               del detour[item]
               break
            for elem in combo: 
               if elem in same_route:
                  flag = 0
                  del detour[item]
                  break
            if flag==1:
               nodes_removed.append(item)
               del detour[item]
               break
            else: break
         if time.time()-start>60:
            break       
         
         
      for items in nodes_removed:
         for elem in worst_solution:
            if items in eval(elem):
               old_route=eval(elem)
               if len(old_route)==3:  #if its a GV, then delete the route
                  try:
                     del worst_solution[f'{old_route}']
                     break
                  except: pass
                  #continue
               elif len(old_route)>4:  #only consider those EV routes such that deleting a node still makes it an EV route, does not become a GV route 
                  new_route = copy.deepcopy(old_route)
                  new_route.remove(items)
                  type = worst_solution[f'{old_route}']['type']
                  tour_info, aa = EV_multiple_tour_feasibility_check(worst_solution, new_route, type, old_route, f=5)
                  try:
                     del worst_solution[f'{old_route}']
                  except: pass
                  for item in tour_info:
                     worst_solution[item] = tour_info[item]
                     worst_solution[item]['type'] =  type
                  break

      return worst_solution, nodes_removed

def repair(best_local_solution, destroyed_solution, nodes_removed, repair_iterations):
   #brute force cheapest insertion among EV routes (polynomial even though brute force)
   cost = 0
   for items in best_local_solution:
      cost+= best_local_solution[items]['objective']
   for i in range(1,repair_iterations):
      if i%9000==0:
         2
         #print(i)
      continue_outer = False
      new_cost = 0
      #new_EV_routes = {key: {'type': value['type']} for key, value in destroyed_solution.items()}
      new_EV_routes = copy.deepcopy(destroyed_solution)
      for items in destroyed_solution:
         new_EV_routes[items]['old_route'] = eval(items)
      for items in nodes_removed:
         older = random.choice(list(new_EV_routes.keys()))
         outer_idx = eval(older)
         inner_idx = random.randint(1, len(outer_idx) - 2)
         outer_idx.insert(inner_idx, items)
         new_EV_routes[str(outer_idx)] = {}
         new_EV_routes[str(outer_idx)]['type'] = new_EV_routes[older]['type']
         new_EV_routes[str(outer_idx)]['old_route'] = eval(older)
         #if len(outer_idx)==len(eval(older)):
         #   pass
         
         del new_EV_routes[older]

      
      for items in new_EV_routes:
         if type(new_EV_routes[items]['old_route'])==str:
            old_route = eval(new_EV_routes[items]['old_route'])
         else: old_route = new_EV_routes[items]['old_route']
         
         if len(eval(items))!= len(old_route):
            tour_info, aa = EV_multiple_tour_feasibility_check(new_EV_routes, eval(items), new_EV_routes[items]['type'], old_route)
            if tour_info:
               for item in tour_info:
                  new_EV_routes[item]['cost'] =  tour_info[item]['cost']
                  new_EV_routes[item]['objective'] =  tour_info[item]['objective']
                  new_EV_routes[item]['battery'] =  tour_info[item]['battery']
                  new_EV_routes[item]['miles'] =  tour_info[item]['miles']
                  new_EV_routes[item]['op_time'] =  tour_info[item]['op_time']

            else:
               continue_outer = True   #tour is not feasible, try again
               break
      if continue_outer:
         continue

      for items in new_EV_routes:
         new_cost+= new_EV_routes[items]['objective']
      #print(new_cost)
      
      if new_cost<cost:
         print(f"better cost found during repair: {new_cost}")
         return new_EV_routes

   return best_local_solution


   #return repair_solution


def solution_cost(sol1, sol2):
    sol1_cost = sum(sol1[item]['objective'] for item in sol1)
    sol2_cost = sum(sol2[item]['objective'] for item in sol2)
    return (sol1_cost, sol1) if sol1_cost < sol2_cost else (sol2_cost, sol2)

def ALNS(best_local_solution):
   #print(best_local_solution)
   destroyed_solution = copy.deepcopy(best_local_solution)
   destroyed_solution, nodes_removed = destroy(destroyed_solution, num_destroy)
   repair_solution = repair(best_local_solution, destroyed_solution, nodes_removed,repair_iterations)
   best_repair_solution_cost, best_repair_solution= solution_cost(best_local_solution, repair_solution)
   return best_repair_solution_cost, best_repair_solution



if __name__ == "__main__":

   start =time.perf_counter()
   repair_iterations = 300*n
   ALNS_iterations = 10
   local_search_iterations = 10000
   num_destroy = 2 #max(int(n*0.05),2)
   objective = "miles"  # or "miles"
   

   
   labels, centroids, X, node_attr = get_node_att()
   
   EV_dict, unassigned_nodes, cluster_info, nodes_original = IFB(labels, centroids, X, node_attr)

   visualize_routes(EV_dict)

   EV_dict_optimized = preprocess_and_optimize(EV_dict)

   GV_assignment_IFB, EV_assignment_IFB = EV_GV_assignment(EV_dict_optimized, unassigned_nodes)
   
   
   EV_dict_only = copy.deepcopy(EV_dict_optimized)

   for item in EV_dict_optimized:
      for i, elements in enumerate(EV_dict_optimized[item]["route"]):
         num_nodes = len(elements)
         if num_nodes==1 or num_nodes==3:
            EV_dict_only[item]["battery"].remove(EV_dict_optimized[item]["battery"][i])
            EV_dict_only[item]["curr_load"].remove(EV_dict_optimized[item]["curr_load"][i])
            EV_dict_only[item]["route"].remove(elements)
   cost_IFB_opt, miles_IFB_opt, GV_assignment_IFB, EV_assignment_IFB = cost_calculation(GV_assignment_IFB,EV_assignment_IFB,EV_dict_only)

   EV_assignment_IFB, GV_assignment_IFB, IFB_objective, current_best_local_search = set_ALNS_obj(EV_assignment_IFB, GV_assignment_IFB, objective)

   #no_collab_total_miles=0
   #no_collab_total_cost=0
   #for i in range(1,n+1):
   #   no_collab_total_miles += a[(0,i)]*2
   #for i in range(1,n+1):
   #   no_collab_total_cost += a[(0,i)]*GV_cost + a[(i,0)]*GV_cost*q[i]
   #miles_saved_IFB_opt = no_collab_total_miles-miles_IFB_opt
   #percentage_miles_saved_IFB_opt = (miles_saved_IFB_opt/(no_collab_total_miles))*100
   #cost_saved_IFB_opt = no_collab_total_cost-cost_IFB_opt
   #percentage_cost_saved_IFB_opt = (cost_saved_IFB_opt/no_collab_total_cost)*100

   all_routes = all_routes_construction(EV_assignment_IFB, GV_assignment_IFB)
   best_overall_solution = {}
   best_overall_solution_cost = float('inf')
   


   for i in range(ALNS_iterations):
      best_local_solution = ALNS_local_search(EV_dict_optimized, EV_assignment_IFB, GV_assignment_IFB, local_search_iterations, all_routes)
      print(f"local search cost:{sum(best_local_solution[item]['objective'] for item in best_local_solution)}")
      best_repair_solution_cost, best_repair_solution = ALNS(best_local_solution)
      if best_local_solution!=best_repair_solution:
         print("better solution found")
      best_local_solution_cost, best_local_solution = solution_cost(best_local_solution, best_repair_solution)
      if best_local_solution_cost<best_overall_solution_cost:
         best_overall_solution = copy.deepcopy(best_local_solution)
         best_overall_solution_cost = best_local_solution_cost



   best_optimized_cost, best_optimized_EV_dict = final_tsp_optimization(best_overall_solution)
   

   end =time.perf_counter()
   print('\n')
   print('\n')
   print(f"best overall cost: {sum(best_overall_solution[item]['objective'] for item in best_overall_solution)}")
   print('\n')
   print(f"cost after subtour optimization: {best_optimized_cost}")
   print('\n')
   print(f"execution time: {end-start} seconds")
   print('\n')
   print(f"number of clusters: {num_clusters}")
   print('\n')
   2