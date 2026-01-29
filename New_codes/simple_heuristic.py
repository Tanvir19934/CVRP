import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import copy
import random
import time
from config_new import *
from gurobipy import Model, GRB, quicksum
from utils_new import  unpack_result, print_solution, create_columns_from_EV_dict
from pricing_coalition_new import  column_generation
rnd = np.random
from itertools import permutations

rand_seed = 111
rnd.seed(42)

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
   #plt.show()
   
   plt.scatter(xc[1:], yc[1:], c=labels, cmap='viridis', label='Data Points')
   for item in q:
      plt.text(xc[item] + 0.2, yc[item] + 0.2, str(q[item]), fontsize=8, color='black')
   plt.text(0,0,"DC",fontsize=10, color='red')
   #plt.show()
   return labels, centroids , X

def k_means_transformed_best_k(x,y,z,num_clusters):
   x = copy.deepcopy(xc)
   y = copy.deepcopy(yc)
   max_dem = max(z.values())
   inertias = []
   # this groups higher loads-higher distances and lower loads-lower distances together 
   for item in z:
      x[item] *= (max_dem/(z[item]+0.00000000001))
      y[item] *= (max_dem/(z[item]+0.00000000001))
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
   #plt.show()

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
   kmeans = kmeans.fit(X)                                    # Fitting the input data
   labels = kmeans.predict(X)                                # Getting the cluster labels
   centroids = kmeans.cluster_centers_                       # Centroid values
   # print("Centroids are:", centroids)                      # From sci-kit learn

   # Visualize clusters
   plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data Points')
   plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=100, c='red', label='Centroids')
   plt.xlabel('Distance')
   plt.ylabel('Additional Feature (e.g., q)')
   plt.title('K-Means Clustering')
   plt.legend()
   #plt.show()

   plt.scatter(xc[1:], yc[1:], c=labels, cmap='viridis', label='Data Points')
   for item in q:
      plt.text(xc[item] + 0.2, yc[item] + 0.2, str(q[item]), fontsize=8, color='black')
   plt.text(0,0,"DC",fontsize=10, color='red')
   #plt.show()
   return labels, centroids , X

def get_node_att():
   node_attr = {}
   for i in range(1,len(N)+1):
      node_attr[i] = {}
      node_attr[i]["dem"] = q[i]
      node_attr[i]["dist"] = a[(0,i)]
   #labels, centroids, X = k_means_3D(xc,yc,q,num_EV)
   #labels, centroids, X= k_means_2D(node_attr,q,num_EV)
   labels, centroids, X = k_means_transformed(xc,yc,q,num_EV) #best for simple heuristic
   #labels, centroids, X = k_means_transformed_best_k(xc,yc,q,num_clusters)  #better
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
   if rem_battery_home < battery_threshold:
      flag_b = False
   if current_load > Q_EV:
      flag_l = False 
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

def travel_back_to_depot(EV_dict, ev, a, EV_velocity, gamma, gamma_l, l, b, o, st=None, element=None, compensate=False):
    """Handle EV trip back to depot and battery/time updates."""
    last_node = EV_dict[ev]['route'][-1][-1]

    # Battery and time update
    b -= (a[(last_node, 0)] / EV_velocity) * (gamma + gamma_l * l)

    if compensate and st is not None and element is not None:
        # Compensate for skipped node time
        o += st[element] + (a[(last_node, element)] / EV_velocity)

    EV_dict[ev]['route'][-1].append(0)
    EV_dict[ev]['curr_load'][-1].append(0)
    EV_dict[ev]['battery'][-1].append(b)

    # Compute recharge/idle time
    if b < 0.8:
        t = a[(last_node, 0)] / EV_velocity + 225 * (0.8 - b) + 120
    else:
        t = a[(last_node, 0)] / EV_velocity + (b - 0.8) * 600 + 180

    EV_dict[ev]['time'].append(o - t)

    # Start new route segment
    EV_dict[ev]['route'].append([0])
    EV_dict[ev]['curr_load'].append([0])
    EV_dict[ev]['battery'].append([1])

    return b, o, t

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
            if element in unassigned_nodes:
               b, l, flag_b, flag_l = battery_level(EV_dict[ev]['route'][-1][-1],element,EV_dict[ev]['curr_load'][-1][-1],EV_dict[ev]['battery'][-1][-1])
               if flag_b==True and flag_l==True:
                  EV_dict[ev]['route'][-1].append(element)
                  EV_dict[ev]['battery'][-1].append(b)
                  EV_dict[ev]['curr_load'][-1].append(l)
                  assigned_nodes.append(element)
                  unassigned_nodes.remove(element)
               else:
                   continue

   for item in EV_dict:
      for element in EV_dict[item]["route"]:
         if element[-1]!=0:
            b = EV_dict[item]["battery"][-1][-1] - (a[(element[-1],0)]/EV_velocity)*(gamma+gamma_l*EV_dict[item]["curr_load"][-1][-1])
            EV_dict[item]["battery"][-1].append(b)
            EV_dict[item]["curr_load"][-1].append(0)
            element.append(0)
   2
   return EV_dict, unassigned_nodes, cluster_info, nodes_original

if __name__ == "__main__":
   start = time.perf_counter()
   labels, centroids, X, node_attr = get_node_att()
      
   EV_dict, unassigned_nodes, cluster_info, nodes_original = IFB(labels, centroids, X, node_attr)
   
   #visualize_routes(EV_dict)

   columns = create_columns_from_EV_dict(EV_dict)
   
   [
        result, not_fractional, model, obj_val, status, CG_iteration, RG_iteration, RG_time, CG_time, 
        CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, constraints, columns
   ] = unpack_result(
            column_generation(
            None, forbidden_set={}, tsp_memo={}, L=None, feasibility_memo={}, global_tsp_memo={}, 
            initial=False, parent_constraints=set(), new_columns_to_add=columns
            )
        )
   end = time.perf_counter()
   print_solution(model)
   print(f"Execution time for IFB heuristic node {NODES}: {end - start}")