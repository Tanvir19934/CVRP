import numpy as np
from matplotlib import pyplot as plt
#import gurobipy
import networkx as nx
from gurobipy import Model, GRB, quicksum
#import re
import pandas as pd
import random
import math
import copy
import time
tic_overall = time.perf_counter()
#initial_values
rnd = np.random
rnd.seed(10)
pass_vehicles = 2200
capacity = 2400

#setting up the network                                             ############################EDIT HERE##################################
G = nx.DiGraph(directed=True)
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_edge(1,2,travel_time = 4.5,    vehicles=pass_vehicles,capacity=capacity)
G.add_edge(2,4,travel_time = 6,    vehicles=pass_vehicles,capacity=capacity)
G.add_edge(1,3,travel_time = 5,    vehicles=pass_vehicles,capacity=capacity)
G.add_edge(3,4,travel_time = 5,       vehicles=pass_vehicles,capacity=capacity)
G.add_edge(2,3,travel_time = 0.8,    vehicles=pass_vehicles,capacity=capacity)

G.add_edge(1,5,travel_time = 4.5,    vehicles=pass_vehicles,capacity=capacity) 
G.add_edge(5,4,travel_time = 8,    vehicles=pass_vehicles,capacity=capacity) 
G.add_edge(1,4,travel_time = 17,    vehicles=pass_vehicles,capacity=capacity) 
G.add_edge(2,6,travel_time = 6,    vehicles=pass_vehicles,capacity=capacity) 
G.add_edge(6,4,travel_time = 5,    vehicles=pass_vehicles,capacity=capacity)
nx.draw_networkx(G)
#plt.show()

#extracting individual road attributes
road_travel_time = {}
road_pass_vehicles = {}
road_capacity = {}
OD = ['OD1','OD2']                              ############################EDIT HERE##################################

for node1, node2, data in G.edges(data=True):
    #print(data['travel_time'])
    road_travel_time[int(str(node1)+str(node2))] = data['travel_time']
    road_pass_vehicles[int(str(node1)+str(node2))] = data['vehicles']
    road_capacity[int(str(node1)+str(node2))] = data['capacity']
#print(road_capacity)

#function that calculates unique path for a OD pair (gives a list of list separated by comma)
def find_path(G,source,target):
    path1 = []
    for path in nx.all_simple_paths(G, source, target):
        path1.append(path)
    return path1
path_OD1 = find_path(G, source=1, target=4)                     ############################EDIT HERE##################################
path_OD2 = find_path(G, source=2, target=4)                     ############################EDIT HERE##################################
#print(path_OD1)
#print(path_OD2)

#function that calculates unique paths for a OD pair (gives list without separation)
def find_unique_path(path):
    path_list = []
    for i in range(0,len(path)):
        chr = '0'
        for j in range(0,len(path[i])):
            chr = chr + str(path[i][j])
        chr = chr[1:]
        chr = int(chr)
        path_list.append(chr)
    return path_list
unique_path_OD1 = find_unique_path(path_OD1)                    ############################EDIT HERE##################################
unique_path_OD2 = find_unique_path(path_OD2)                    ############################EDIT HERE##################################
all_unique_routes = list(set(unique_path_OD1  + unique_path_OD2))
#print(all_unique_routes)

#function that calculates unique links for a OD pair
def find_link(path):
    link_list = []
    for i in range(0,len(path)):
        for j in range(0,len(path[i])):
            if j < len(path[i])-1:
                link_list.append(int(str(path[i][j])+str(path[i][j+1])))
    link_list = set(link_list)
    link_list = list(link_list)
    return link_list
links_OD1 = find_link(path_OD1)
links_OD2 = find_link(path_OD2)
all_unique_links = list(set(links_OD1 + links_OD2))
#print(all_unique_links)




all_OD_links = {'OD1':links_OD1,'OD2':links_OD2}                ############################EDIT HERE##################################
all_OD_paths = {'OD1':unique_path_OD1,'OD2':unique_path_OD2}    ############################EDIT HERE##################################
#print(all_OD_links)
#print(all_OD_paths)

def link_inside_route(all_unique_routes,all_unique_links):
    link_in_route = {}
    which_links_in_route = {}
    for element in all_unique_links:
        link_in_route[element] = []
        for item in all_unique_routes:
            link_list = []
            for j in range(0,len(str(item))):
                if j < len(str(item))-1:
                    link_list.append(int(str(item)[j]+str(item)[j+1]))
            which_links_in_route[item]=link_list
            if element in link_list:
                link_in_route[element].append(item)           
    return link_in_route,which_links_in_route

link_in_route,which_links_in_route = link_inside_route(all_unique_routes,all_unique_links)



#calculates the link cost for all OD pairs
#output is in the form of {'x34_OD1': 5.52, 'x34_OD2': 5.52, etc}
def link_time_function(l_trucks):
    #all link time using BPR function
    link_time_dict={}
    for item in link_trucks:
        link_time_dict[item] = road_travel_time[item]*(1+alpha*((3*l_trucks[item]+road_pass_vehicles[item])/road_capacity[item])**beta)
    return link_time_dict

""" t = link_time(1)
print(t)
print(all_OD_paths)
print(all_unique_routes)
print(unique_path_OD1,1)
print(unique_path_OD2) """
#calculates the cost for each route in each OD pair
#output is {'route134OD1_cost': 11.280, 'route24OD2_cost': 6.587...etc.}
def route_time_function(link_time_dict):
    #{'OD1': [124, 1234, 134], 'OD2': [24, 234]}
    route_time_dict = {}
    temp_dict = {}
    for elements in which_links_in_route:
        s = 0
        for item in link_time_dict:
            if item in which_links_in_route[elements]:
                s = s + link_time_dict[item]
        temp_dict[elements] = s
    for item in all_OD_paths:
        for j in all_OD_paths[item]:
            if j in temp_dict.keys():
                route_time_dict[j]= {'OD': item, 'route': j, 'time': temp_dict[j]}
    
    return route_time_dict
#t = route_time(1)
#print(t)                 
#link trucks in system optimum
#reading GAMS output data from excel file 
df_so_trucks = pd.read_excel(r'/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Cost sharing/GAMS/sys_opt.xlsx', sheet_name='Xlt')
df_so_trucks = df_so_trucks.loc[:,0:1]
df_so_trucks = df_so_trucks.rename(columns={0: "link", 1: "trucks"})
#print(df_so_trucks)

df_so_link_time = pd.read_excel(r'/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Cost sharing/GAMS/sys_opt.xlsx', sheet_name='Cl')
df_so_link_time = df_so_link_time.loc[:,0:1]
df_so_link_time = df_so_link_time.rename(columns={0: "link", 1: "time"})
#print(df_so_link_time)

df_eq_link_time = pd.read_excel(r'/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Cost sharing/GAMS/all.xlsx', sheet_name='Cl')
df_eq_link_time = df_eq_link_time.loc[:,0:1]
df_eq_link_time = df_eq_link_time.rename(columns={0: "link", 1: "time"})
#print(df_eq_link_time)

eq_link_time = dict(zip(df_eq_link_time.link, df_eq_link_time.time))
so_link_time = dict(zip(df_so_link_time.link, df_so_link_time.time))
so_trucks = dict(zip(df_so_trucks.link, df_so_trucks.trucks))


#print(eq_link_time)
#print(so_trucks)

eq_route_time = {}
so_route_time = {}

for item in which_links_in_route:
    eq_route_time[item] = 0
    so_route_time[item] = 0
    for element in which_links_in_route[item]:
        try:
            eq_route_time[item] += eq_link_time[element]
            so_route_time[item] += so_link_time[element]
        except:
            eq_route_time[item] =0
            so_route_time[item] =0

#print(eq_route_time)

#print(so_route_time)

#print(all_OD_paths)

avg_eq = {}
for item in all_OD_paths:
    avg_eq[item] = 0
    for element in all_OD_paths[item]:
        avg_eq[item] = avg_eq[item]+eq_route_time[element]
    avg_eq[item] = avg_eq[item]/len(all_OD_paths[item])
#print(avg_eq)


payment_factor = 0.85
reward_factor = 2.4
MC_factor = 1
account = 50
final_account = 0 
MIPGap = 0.45           #0.45 gives optimal solution
MIPFocus = 1
reopt_point = 50 #if len(assigned_customer)%reopt_point==0:
reopt_count = 0
#BPR parameters
alpha = 0.15
beta = 4
#generating customer attributes
UE_time = {'OD1': avg_eq['OD1'], 'OD2': avg_eq['OD2']}


#generating customer attributes
mu = 100
sigma = 0
s = np.random.normal(mu, sigma, 4)
s  = [math.ceil(item) for item in s]
if sum(s) > mu*4:
    diff = sum(s) - mu*4
    whole = int(diff/4)
    remainder = diff%4
    s = [item-whole for item in s]
    s[0] = s[0] - remainder
if sum(s) < mu*4:
    diff = mu*4 - sum(s)
    whole = int(diff/4)
    remainder = diff%4
    s = [item+whole for item in s]
    s[0] = s[0] + remainder

VOT = [1.8,2.1]   #have to be in increasing order
customer_att = {}  #{'1_OD2': {'OD': 'OD2', 'VOT': 2, 'UE_cost': 10.6}

OD_VOT = {}
i = 0
for item in OD:
    for elements in VOT:
        OD_VOT[(item,elements)] = s[i]
        i = i+1

j = 1

#OD_VOT = {('OD1', VOT[0]): 500, ('OD1',VOT[1]): 500, ('OD2', VOT[0]): 500, ('OD2', VOT[1]): 500}

for item in OD_VOT:
    for i in range(0,OD_VOT[item]):
        customer_att['{0}_{1}'.format(j,item[0])] = {'OD':item[0],'VOT':item[1],'UE_cost':UE_time[item[0]]*item[1]}
        j = j+1
print(OD_VOT)
#print(customer_att)
#random shuffling
items=list(customer_att.items()) 
random.shuffle(items)
customer_att = {key:value for key,value in items}
#print(customer_atttt==customer_att)

#initializing number of trucks in each road segment as 0
link_trucks = {}
for item in all_unique_links:
        link_trucks[item]= 0
#calculate initial route time
#print(link_trucks,10)            #{'x34_OD1': {'OD': 'OD1', 'link': 34, 'trucks': 0}, 'x12_OD1': {'OD': 'OD1', 'link': 12, 'trucks': 0}
#link_times = link_time_function(link_trucks) #{'x24_OD1': {'OD': 'OD1', 'link': 24, 'time': 5.806}, 'x24_OD2': {'OD': 'OD2', 'link': 24, 'time': 5.806}}
#print(link_times,1000)
#route_times = route_time_function(link_times)
#print(route_times)


def VOT_class(potential_routes_sorted):
    #create VOT classes for the routes
    temp_VOT = VOT
    if len(potential_routes_sorted) < len(VOT):
        x = len(VOT)/len(potential_routes_sorted)
        y = math.floor(x)
        z = len(VOT)%len(potential_routes_sorted)
        for item in potential_routes_sorted:
            potential_routes_sorted[item]['assign'] = []
            for i in range(1,y+1):
                potential_routes_sorted[item]['assign'].append(temp_VOT[-i])
            temp_VOT = temp_VOT[:-y]
        if z!=0:
            potential_routes_sorted = dict(sorted(potential_routes.items(), key=lambda x: x[1]['time'],reverse=True))
            for i,item in zip(range(0,z),potential_routes_sorted):
                potential_routes_sorted[item]['assign'].append(VOT[i])
            potential_routes_sorted = dict(sorted(potential_routes.items(), key=lambda x: x[1]['time'],reverse=False))
        return potential_routes_sorted 

    if len(potential_routes_sorted) == len(VOT):
        for i,item in zip(range(1,len(VOT)+1),potential_routes_sorted):
            potential_routes_sorted[item]['assign'] = [VOT[-i]]
            if i==len(VOT)+1:
                break
        return potential_routes_sorted 

    if len(potential_routes_sorted) > len(VOT):
        x = len(potential_routes_sorted)/len(VOT)
        y = math.floor(x)
        z = len(potential_routes_sorted)%len(VOT)
        for item in potential_routes_sorted:
            potential_routes_sorted[item]['assign'] = []
        for item in potential_routes_sorted:
            for i in range(1,y+1):
                potential_routes_sorted[item]['assign'].append(temp_VOT[-1])
            temp_VOT = temp_VOT[:-1]
            if len(temp_VOT)==0:
                break

        if z!=0:
            potential_routes_sorted = dict(sorted(potential_routes.items(), key=lambda x: x[1]['time'],reverse=True))
            for i,item in zip(range(0,z),potential_routes_sorted):
                potential_routes_sorted[item]['assign'].append(VOT[i])
            potential_routes_sorted = dict(sorted(potential_routes.items(), key=lambda x: x[1]['time'],reverse=False))
        return potential_routes_sorted 


def convert_route_to_link(route,OD):
    d = {}
    link_used = []
    route = str(route)
    #route = route.replace('route', '')    #x34_OD1
    for i in range(0,len(route)-1):
        d[int(route[i]+route[i+1])] = 1
        #link_used.append(route[i]+route[i+1])
        link_used.append(int(route[i]+route[i+1]))
    return d, link_used
def convert_route_to_link2(route):
    route = str(route)
    link_used = []
    for i in range(0,len(route)-1):
        link_used.append(int(route[i]+route[i+1]))
    return link_used   

OD_route_links = {}
OD_route = {'OD1':unique_path_OD1, 'OD2':unique_path_OD2}
for item in OD_route:
    for i in OD_route[item]:
        links = convert_route_to_link2(i)
        OD_route_links[item] = {i:links}    

def payment_function(assignment, link_trucks, payment_factor,assigned_customer, customer_att, cost_increase,
                      account_route, payment_reward ,unassigned_customer,assg_cust_first_time,MC_factor ):
    dummy_link_trucks = {}
    changed_link_trucks = {}
    #account_route_dummy = {}
    dummy_link_trucks = copy.deepcopy(link_trucks)
    #account_route_dummy = copy.deepcopy(account_route)
    temp_assigned_customer = copy.deepcopy(assigned_customer) #assigned_customer[customer_ID] = {'route': 'NA', 'links_used': 'NA', 'cost': 'NA','VOT': assignment[customer_ID][first_route]['VOT']}

    #initial assignment of the most efficient route
    customer_ID = next(iter(assignment))   #'4_OD1' (customer ID)
    first_route = next(iter(assignment[customer_ID]))     #'route124'
    
    changed_link_trucks, links_used = convert_route_to_link(first_route ,assignment[customer_ID][first_route]['OD'])    

    if temp_assigned_customer:   #if temp_assigned_customer is not empty (needed for the first customer)
        #calculate the cost increase
        for item in changed_link_trucks:
            dummy_link_trucks[item] = changed_link_trucks[item] + dummy_link_trucks[item]
        for element in temp_assigned_customer:
            for item in dummy_link_trucks:
                if item in temp_assigned_customer[element]['links_used']:
                    new_link_times = link_time_function(dummy_link_trucks) #{'x24_OD1': {'OD': 'OD1', 'link': 24, 'time': 5.806}, 'x24_OD2': {'OD': 'OD2', 'link': 24, 'time': 5.806}}
                    new_route_times = route_time_function(new_link_times)
                    update_route_time = temp_assigned_customer[element]['route']
                    temp_assigned_customer[element]['cost'] = new_route_times[update_route_time]['time']*temp_assigned_customer[element]['VOT']+ payment_reward[element]
                    temp_assigned_customer[element]['time'] = new_route_times[update_route_time]['time']
                        
        for k in temp_assigned_customer:
            cost_increase[k] = temp_assigned_customer[k]['cost']-assigned_customer[k]['cost']
        marginal_cost = sum(cost_increase.values())
        customer_operation_cost =  assignment[customer_ID][first_route]['cost']
        payment_cost = payment_factor*(customer_att[customer_ID]['UE_cost'] - customer_operation_cost) 
        customer_total_cost = customer_operation_cost + payment_cost + marginal_cost*(1-MC_factor)
        if customer_total_cost < customer_att[customer_ID]['UE_cost']:
            assg_cust_first_time[customer_ID] = {'route': first_route, 'links_used': links_used, 
            'cost': customer_total_cost,'VOT': assignment[customer_ID][first_route]['VOT'], 
            'time':assignment[customer_ID][first_route]['time'],'OD': assignment[customer_ID][first_route]['OD']}
            assigned_customer = copy.deepcopy(temp_assigned_customer)   #####?????#####
            assigned_customer[customer_ID] = {'route': first_route, 'links_used': links_used, 
            'cost': customer_total_cost,'VOT': assignment[customer_ID][first_route]['VOT'], 
            'time':assignment[customer_ID][first_route]['time'],'OD': assignment[customer_ID][first_route]['OD']}
            account_route[first_route] = payment_cost + account_route[first_route] -marginal_cost*MC_factor
            payment_reward[customer_ID] = payment_cost
            link_trucks = copy.deepcopy(dummy_link_trucks)
            return assigned_customer,  link_trucks, cost_increase,account_route,payment_reward,unassigned_customer,assg_cust_first_time 
        else:
            unassigned_customer[customer_ID] = {'route': 0, 'links_used': 0, 'cost': 0,
            'VOT': assignment[customer_ID][first_route]['VOT'], 'OD': assignment[customer_ID][first_route]['OD']}
            return assigned_customer,  link_trucks, cost_increase,account_route,payment_reward,unassigned_customer,assg_cust_first_time  
            #route_time['route{}'.format(all_OD_paths[elements][i])] = {'OD':elements,'route':all_OD_paths[elements][i],'time':s} 
    else:
        if (assignment[customer_ID][first_route]['cost'] < customer_att[customer_ID]['UE_cost']):
            customer_operation_cost =  assignment[customer_ID][first_route]['cost']
            payment_cost = payment_factor*(customer_att[customer_ID]['UE_cost'] - customer_operation_cost)
            customer_total_cost = customer_operation_cost + payment_cost
            assg_cust_first_time[customer_ID] = {'route': first_route, 'links_used': links_used, 
            'cost': customer_total_cost,'VOT': assignment[customer_ID][first_route]['VOT'], 
            'time':assignment[customer_ID][first_route]['time'],'OD': assignment[customer_ID][first_route]['OD']}            
            assigned_customer[customer_ID] = {'route': first_route, 'links_used': links_used, 
            'cost': customer_total_cost,'VOT': assignment[customer_ID][first_route]['VOT'], 
            'time':assignment[customer_ID][first_route]['time'],'OD': assignment[customer_ID][first_route]['OD']}
           
            account_route[first_route] = payment_cost + account_route[first_route]
            payment_reward[customer_ID] = payment_cost
            for item in changed_link_trucks:
                dummy_link_trucks[item] = changed_link_trucks[item] + dummy_link_trucks[item]
            link_trucks = copy.deepcopy(dummy_link_trucks)
            return assigned_customer,  link_trucks, cost_increase,account_route,payment_reward,unassigned_customer,assg_cust_first_time 
        else:
            unassigned_customer[customer_ID] = {'route': 0, 'links_used': 0, 'cost': 0,
            'VOT': assignment[customer_ID][first_route]['VOT'], 'OD': assignment[customer_ID][first_route]['OD']}
            return assigned_customer,  link_trucks, cost_increase,account_route,payment_reward,unassigned_customer,assg_cust_first_time  

def reward_function(assignment, link_trucks, reward_factor, assigned_customer, customer_att, cost_increase, 
                    account_route, payment_reward, unassigned_customer,assg_cust_first_time,MC_factor,reward_count ):
    dummy_link_trucks = {}
    changed_link_trucks = {}
    #account_route_dummy = {}
    dummy_link_trucks = copy.deepcopy(link_trucks)
    #account_route_dummy = copy.deepcopy(account_route)
    
    temp_assigned_customer = copy.deepcopy(assigned_customer) #assigned_customer[customer_ID] = {'route': 'NA', 'links_used': 'NA', 'cost': 'NA','VOT': assignment[customer_ID][first_route]['VOT']}
    
    #initial assignment of the least efficient route
    least_eff = {}
    customer_ID = next(iter(assignment))   #'4_OD1' (customer ID)
    for item in all_unique_routes:
        if item in list(assignment[customer_ID].keys()):
           least_eff[item]=assignment[customer_ID][item]['time']
    least_eff_route = {k: v for k, v in sorted(least_eff.items(), key=lambda item: item[1], reverse = True)}    
    first_route = next(iter(least_eff_route))     #'route124'
    
    #initial assignment of the most efficient route
    #customer_ID = next(iter(assignment))   #'4_OD1' (customer ID)
    #first_route = next(iter(assignment[customer_ID]))     #'route124'   
    
    changed_link_trucks, links_used = convert_route_to_link(first_route ,assignment[customer_ID][first_route]['OD'])    
    #changed_link_trucks = {'x12_OD1': 1, 'x24_OD1': 1}, 'links_used' = ['x12_OD1', 'x24_OD1']
    # #{'3_OD1': {'route': 'route124', 'links_used': ['x12_OD1', 'x24_OD1'], 'cost': 22.1}}
    #print(assigned_customer,100000)
    #temp_assigned_customer[customer_ID] = {'route': 'NA', 'links_used': 'NA', 'cost': 'NA','VOT': assignment[customer_ID][first_route]['VOT']}
    #print(new_link_time,100)

    if temp_assigned_customer:   #if temp_assigned_customer is not empty (needed for the first customer)
        #calculate the cost increase
        for item in changed_link_trucks:
            dummy_link_trucks[item] = changed_link_trucks[item] + dummy_link_trucks[item]
        for element in temp_assigned_customer:
            for item in changed_link_trucks:
                if item in temp_assigned_customer[element]['links_used']:
                    new_link_times = link_time_function(dummy_link_trucks) #{'x24_OD1': {'OD': 'OD1', 'link': 24, 'time': 5.806}, 'x24_OD2': {'OD': 'OD2', 'link': 24, 'time': 5.806}}
                    new_route_times = route_time_function(new_link_times)
                    update_route_time = temp_assigned_customer[element]['route']
                    temp_assigned_customer[element]['cost'] = new_route_times[update_route_time]['time']*temp_assigned_customer[element]['VOT'] + payment_reward[element]                  
                    #else:
                     #   temp_assigned_customer[element]['cost'] = new_route_times[update_route_time]['time']*temp_assigned_customer[element]['VOT']
        
        for k in temp_assigned_customer:
            cost_increase[k] = temp_assigned_customer[k]['cost']-assigned_customer[k]['cost']
        marginal_cost = sum(cost_increase.values())
        customer_operation_cost =  assignment[customer_ID][first_route]['cost']
        reward_cost = reward_factor*(customer_operation_cost-customer_att[customer_ID]['UE_cost']) 
        customer_total_cost = customer_operation_cost - reward_cost + marginal_cost*(1-MC_factor)
        #if (customer_total_cost < customer_att[customer_ID]['UE_cost']):
        if (customer_total_cost < customer_att[customer_ID]['UE_cost']) and (account_route[first_route]>reward_cost):
            assg_cust_first_time[customer_ID] = {'route': first_route, 'links_used': links_used, 
            'cost': customer_total_cost,'VOT': assignment[customer_ID][first_route]['VOT'], 
            'time':assignment[customer_ID][first_route]['time'],'OD': assignment[customer_ID][first_route]['OD']}            
            assigned_customer = copy.deepcopy(temp_assigned_customer)   #####?????#####
            assigned_customer[customer_ID] = {'route': first_route, 'links_used': links_used, 
            'cost': customer_total_cost,'VOT': assignment[customer_ID][first_route]['VOT'], 
            'time':assignment[customer_ID][first_route]['time'],'OD': assignment[customer_ID][first_route]['OD']}
            account_route[first_route] =  account_route[first_route] - reward_cost -marginal_cost*MC_factor
            payment_reward[customer_ID] = -reward_cost
            link_trucks = copy.deepcopy(dummy_link_trucks)
            return assigned_customer,  link_trucks, cost_increase,account_route,payment_reward,unassigned_customer,assg_cust_first_time ,reward_count   
        else:
            unassigned_customer[customer_ID] = {'route': 0, 'links_used': 0, 'cost': 0,
            'VOT': assignment[customer_ID][first_route]['VOT'], 'OD': assignment[customer_ID][first_route]['OD']}
            return assigned_customer,  link_trucks, cost_increase,account_route,payment_reward,unassigned_customer,assg_cust_first_time,reward_count  
    else:
        if (assignment[customer_ID][first_route]['cost'] > customer_att[customer_ID]['UE_cost']):
            customer_operation_cost =  assignment[customer_ID][first_route]['cost']
            reward_cost = reward_factor*(customer_operation_cost - customer_att[customer_ID]['UE_cost'])
            customer_total_cost = customer_operation_cost - reward_cost
            assg_cust_first_time[customer_ID] = {'route': first_route, 'links_used': links_used, 
            'cost': customer_total_cost,'VOT': assignment[customer_ID][first_route]['VOT'], 
            'time':assignment[customer_ID][first_route]['time'],'OD': assignment[customer_ID][first_route]['OD']}            
            assigned_customer[customer_ID] = {'route': first_route, 'links_used': links_used, 
            'cost': customer_total_cost,'VOT': assignment[customer_ID][first_route]['VOT'], 
            'time':assignment[customer_ID][first_route]['time'],'OD': assignment[customer_ID][first_route]['OD']}
            account_route[first_route] = account_route[first_route] - reward_cost
            payment_reward[customer_ID] = -reward_cost
            for item in changed_link_trucks:
                dummy_link_trucks[item] = changed_link_trucks[item] + dummy_link_trucks[item]
            link_trucks = copy.deepcopy(dummy_link_trucks)
            return assigned_customer,  link_trucks, cost_increase,account_route,payment_reward,unassigned_customer,assg_cust_first_time,reward_count  
        else:
            unassigned_customer[customer_ID] = {'route': 0, 'links_used': 0, 'cost': 0,
            'VOT': assignment[customer_ID][first_route]['VOT'], 'OD': assignment[customer_ID][first_route]['OD']}
            return assigned_customer,  link_trucks, cost_increase,account_route,payment_reward,unassigned_customer,assg_cust_first_time,reward_count 


assigned_customer = {}
iteration = {}
assg_cust_first_time = {}
cost_increase = {}
payment_reward = {}
unassigned_customer = {}
count_total = 0
count_unassigned = 0
count_assigned = 0
opt_count = 0 
reward_count = 0
"""#account_route = {}
 for item in all_unique_routes:
    account_route['route{}'.format(item)] = account
print(account_route) """

account_route = {item: account for  item in all_unique_routes}
#print(account_route)
#print(customer_att) #{'1_OD2': {'OD': 'OD2', 'VOT': 2, 'UE_cost': 10.6}


#re-optimization in gurobi
def re_opt(route_times, assigned_customer, link_trucks, OD_route_links, link_in_route, road_travel_time, alpha, 
beta, capacity, pass_vehicles,which_links_in_route,all_unique_links, all_unique_routes, opt_count, assg_cust_first_time,iteration,reopt_count):
    tic = time.perf_counter()
    #Sets
    customer_VOT_set = {item:assigned_customer[item]['VOT'] for item in assigned_customer} 
    pot_route = {item:all_OD_paths[assigned_customer[item]['OD']] for item in assigned_customer}
    prev_customer_cost = {item:assigned_customer[item]['cost'] for item in assigned_customer}
    #print(prev_customer_cost) #{'3_OD1': 21.33, '5_OD1': 21.30, '1_OD2': 12.42}
    #print(assigned_customer) #{'6_OD2': {'route': '234', 'links_used': ['x23_OD2', 'x34_OD2'], 'cost': 14.42, 'VOT': 2.1},
    #print(customer_VOT_set) #{'2_OD1': 2.1, '1_OD1': 1.8}
    #print(pot_route) #{'2_OD1': [124, 1234, 134], '10_OD2': [24, 234]}
    #print(link_in_route) #{34: [134, 234, 1234], 12: [1234, 124], 13: [134], 23: [234, 1234], 24: [24, 124]}
    #print(which_links_in_route) # {134: [13, 34], 234: [23, 34], 1234: [12, 23, 34], 24: [24], 124: [12, 24]}#print(road_travel_time) #{12: 4.5, 13: 5, 24: 6, 23: 0.8, 34: 5} 
    
    mdl = Model('OTR')
    X = {}
    z1 = {}
    z2 = {}
    for item in pot_route:
        for element in pot_route[item]:
            X[item,element] = mdl.addVar(vtype=GRB.BINARY, name="X[{0},{1}]".format(item,element))
        
    mdl.addConstrs((quicksum(X[item,j] for j in pot_route[item])==1) for item in pot_route)
        
    #Alternative formulation of the vars and constraints above
    #X = mdl.addVars(((item, element) for item in pot_route for element in pot_route[item]), vtype = GRB.BINARY, name = "X")
    #mdl.addConstrs(X.sum(itm,'*')==1 for itm in pot_route)  #worked      

    #for item,element in X[item,element]:
    #    z1[] == quicksum((X[i,r]  for i in list(pot_route.keys()) for r in pot_route[i])
    #   +pass_vehicles)/capacity 
        
    ZC_rep = {}
    ZC = {}
    LC = {}
        
    for item in all_unique_links:
            
        ZC_rep[item] = mdl.addVar(lb=0, name="ZC_rep[{0}]".format(item))
        route_set = []
        for element in which_links_in_route:
            if item in which_links_in_route[element]:
                route_set.append(element)
        mdl.addConstr(ZC_rep[item]==(pass_vehicles+3*(quicksum(X[i,j] for i in pot_route for j in route_set if j in pot_route[i])))/capacity, name="Z_rep[{0}]".format(item))
    for item in all_unique_links:
        LC[item] = mdl.addVar(lb=-GRB.INFINITY, name="LC[{0}]".format(item))
        ZC[item] = mdl.addVar(lb=0, name="ZC[{0}]".format(item))
        mdl.addGenConstrPow(ZC_rep[item],ZC[item],beta,"rep","FuncPieces=100")
    for item in all_unique_links:
        mdl.addConstr(LC[item]==road_travel_time[item]*(1+alpha*ZC[item]), name="Link_cost[{0}]".format(item))
    RC = {}
    RC_dummy = {}
    for item in all_unique_routes:
        RC[item] = mdl.addVar(lb=0, name="RC[{0}]".format(item))
        RC_dummy[item] = mdl.addVar(lb=0, name="RC_dummy[{0}]".format(item))
        mdl.addConstr(RC[item]== quicksum(LC[i] for i in which_links_in_route[item]), name="RC_cost[{0}]".format(item))

    for item in assigned_customer:
       mdl.addConstr(((quicksum(X[item,element]*RC[element] for element in all_unique_routes if element in pot_route[item]))*customer_VOT_set[item]<= prev_customer_cost[item])
        ,name='prev_cost_{0}'.format(item))   #or the constraint below
    #mdl.addConstrs(((quicksum(X[item,j] for j in pot_route[item]))*customer_VOT_set[item]<= prev_customer_cost[item]) for item in pot_route)

    mdl.setObjective(quicksum(customer_VOT_set[item]*X[item,element]*RC[element]
    for item in customer_VOT_set for element in all_unique_routes if element in pot_route[item]),GRB.MINIMIZE)
    mdl.params.NonConvex = 2
    mdl.setParam('MIPGap', MIPGap)
    mdl.setParam('MIPFocus', MIPFocus)


    mdl.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/my_lp.lp")

    mdl.modelSense = GRB.MINIMIZE
    mdl.optimize()
    #extracting the name and values of the solution variables
    flag = 0
    if mdl.SolCount>0:
        mvars = [var for var in mdl.getVars() if "X" in var.VarName]
        names = mdl.getAttr('VarName', mvars)
        values = mdl.getAttr('X', mvars)
        result = dict(zip(names, values))
        #print(result)
        for item in result:
            if result[item]!=0:
                s = item
                s = s.replace("X[", "")
                s = s.replace("]", "")
                s = s.split(",")
                for i in assigned_customer:
                    if s[0]==assigned_customer[i]:
                        assigned_customer[i]['route']=int(s[1])
                        assigned_customer[i]['link_used']=convert_route_to_link2(int(s[1]))
        #link_trucks = {}
        
        for elements in all_unique_links:
            s = 0
            for item in assigned_customer:
                if elements in assigned_customer[item]['links_used']:
                    s = s + 1
                    link_trucks[elements] = s
        link_times = link_time_function(link_trucks) 
        route_times = route_time_function(link_times)
        for item in assigned_customer:
            assigned_customer[item]['time']= route_times[assigned_customer[item]['route']]['time']
            assigned_customer[item]['cost'] = payment_reward[item]+(assigned_customer[item]['VOT'])*(assigned_customer[item]['time'])
    else:
        flag = flag + 1

    opt_count = opt_count + 1
    reopt_count = 0
    print(f"iteration = {opt_count}")
    toc = time.perf_counter()
    print("Elapsed time {}".format(toc-tic))
    iteration[opt_count]=toc-tic
    #print(link_trucks) #{'x34_OD1': {'OD': 'OD1', 'link': 34, 'trucks': 0}, 'x12_OD1': {'OD': 'OD1', 'link': 12, 'trucks': 0}
    return assigned_customer, link_trucks, OD_route_links, link_in_route, road_travel_time, alpha, beta, capacity, pass_vehicles,which_links_in_route,all_unique_links, all_unique_routes, opt_count, assg_cust_first_time, iteration, reopt_count
    #return opt_count, link_trucks
#tic = time.perf_counter()
#assignment of route at each arrival of request
for element in customer_att:
    count_total = count_total + 1
    assignment = {}
    potential_routes = {}
    link_times = {}
    route_times = {}
    customer_OD = customer_att[element]['OD']
    customer_VOT = customer_att[element]['VOT']
    
    link_times = link_time_function(link_trucks) 
    route_times = route_time_function(link_times)   

    for item in route_times:
        if customer_OD==route_times[item]['OD']:
            potential_routes[item] = copy.deepcopy(route_times[item])
    potential_routes_sorted = dict(sorted(potential_routes.items(), key=lambda x: x[1]['time'],reverse=False))
    potential_routes_sorted_VOT_class = VOT_class(potential_routes_sorted)
#print(potential_routes_sorted)
#print(potential_routes_sorted_VOT_class)

    temp_assignment = {}
    assignment = {}   #{'1_OD1': {'route1234': {'time': 11.169, 'cost': 11.16, 'OD': 'OD1', 'VOT': 1}, 'route134': {'time': 11.22, 'cost': 11.22, 'OD': 'OD1', 'VOT': 1}}}
    for item in potential_routes_sorted_VOT_class:
        if customer_VOT in potential_routes_sorted_VOT_class[item]['assign']:
            temp_assignment[item] = {'time': potential_routes_sorted_VOT_class[item]['time'], 
            'cost': potential_routes_sorted_VOT_class[item]['time']*customer_VOT,'OD':customer_OD, 'VOT': customer_VOT}
            assignment[element]=  temp_assignment
    #print(assignment,1000)

    for item in assignment:
        key = list(assignment[item].keys())
    count_higher = 0
    count_lower = 0

    for item in assignment:

        #check if all potential route cost have lower or higher (or neither) cost than UE. 
        # If so, assign the fastest route and take payment or rewards if conditions are met.
        #else assign the route closest to UE if conditions are met.

        for k in key:

            if assignment[item][k]['cost'] < customer_att[item]['UE_cost']:
                count_lower = count_lower + 1

            if assignment[item][k]['cost'] > customer_att[item]['UE_cost']:
                count_higher = count_higher + 1        
        
        if count_lower!=0:
            assigned_customer, link_trucks, cost_increase,account_route,payment_reward,unassigned_customer,assg_cust_first_time  = payment_function(assignment, link_trucks,  
                                                                                       payment_factor, assigned_customer, customer_att, cost_increase,
                                                                                        account_route, payment_reward,unassigned_customer,assg_cust_first_time,MC_factor  )
        if count_higher == len(key):
            assigned_customer,  link_trucks, cost_increase,account_route,payment_reward,unassigned_customer,assg_cust_first_time,reward_count  = reward_function(assignment, link_trucks,  
                                                                                       reward_factor, assigned_customer, customer_att, cost_increase,
                                                                                        account_route, payment_reward,unassigned_customer,assg_cust_first_time,MC_factor,reward_count )
    reopt_count = reopt_count + 1

    
    #if (count_total%re_opt_point==0) | (count_total == len(customer_att)):
    #reoptimize = re_opt(assigned_customer, link_trucks, OD_route_links, link_in_route, road_travel_time,
    #alpha, beta, capacity, pass_vehicles,which_links_in_route,all_unique_links, all_unique_routes, opt_count, assg_cust_first_time)

    #toc = time.perf_counter()
    if reopt_count == reopt_point:
        assigned_customer, link_trucks, OD_route_links, link_in_route, road_travel_time, alpha, beta, capacity, pass_vehicles,which_links_in_route,all_unique_links, all_unique_routes, opt_count, assg_cust_first_time,iteration, reopt_count = re_opt(route_times, assigned_customer, link_trucks, OD_route_links, link_in_route, road_travel_time, alpha, beta, capacity, pass_vehicles,which_links_in_route,all_unique_links, all_unique_routes, opt_count, assg_cust_first_time, iteration,reopt_count)
#OD_route_links = {'OD1': {134: [13, 34]}, 'OD2': {234: [23, 34]}}
#assigned_customer={'6_OD2': {'route': 'route234', 'links_used': ['x23_OD2', 'x34_OD2'], 'cost': 14.42, 'VOT': 2.1},


























#calculating number of assigned and unassigned requests, final account balance
for item in unassigned_customer:
    count_unassigned = count_unassigned + 1
for item in assigned_customer:
    count_assigned = count_assigned + 1
for item in account_route:
    final_account = final_account + account_route[item]
    account_route[item] = account_route[item]- account
initial_account = len(account_route)*account

#marginal cost
marg_cost_per_cust = {}
for item in assigned_customer:
     marg_cost_per_cust[item] = assigned_customer[item]['cost']-assg_cust_first_time[item]['cost']
marg_cost = sum(marg_cost_per_cust.values())


for item in payment_reward:
    if payment_reward[item]<0:
        print(payment_reward[item])



#print(assigned_customer,1000000)
#print(assg_cust_first_time)
#print(account)
#print(link_trucks)
#print(cost_increase)
#print(account_route)
#print(payment_reward)
#print(unassigned_customer)
#print(customer_att)
#print(route_times)




########################calculate cost increase due to background traffic################################

assignment_bg = copy.deepcopy(unassigned_customer) 

def bg_traffic(assignment_bg, UE_time, VOT):
    VOT_OD = {}
    count = 0
    for i in UE_time:
        for j in VOT:
            count = count + 1
            VOT_OD[count] = {'OD':i, 'VOT':j, 'count':0}
    #print(VOT_OD)
    
    for item in assignment_bg:
        for i in VOT_OD:
            if (str(assignment_bg[item]['OD']) == str(VOT_OD[i]['OD'])) & (assignment_bg[item]['VOT'] == VOT_OD[i]['VOT']):
                VOT_OD[i]['count'] = VOT_OD[i]['count'] + 1
    #print(VOT_OD) #{1: {'OD': 'OD1', 'VOT': 1.8, 'count': 77}, 2: {'OD': 'OD1', 'VOT': 2.1, 'count': 75}
    return(VOT_OD)

VOT_OD = bg_traffic(assignment_bg, UE_time, VOT)
#reading GAMS output data from excel file 
df = pd.read_excel(r'/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Cost sharing/GAMS/all.xlsx', sheet_name='alpha')
df = df.loc[:,0:3]
df = df.rename(columns={0: "OD", 1: "VOT", 2:'route',3:"fraction"})
df = df.loc[~(df['fraction'] == 0)]
df.reset_index(inplace=True)
#print(df)
for i in range(1,len(VOT)+1):
    df.replace('VOT{}'.format(i),VOT[i-1],inplace=True)
#print(df)

#calculating increased number of customer in different routes
bg_dict = {}
changed_link_trucks = {}
#print(df)
bg_link_trucks = copy.deepcopy(link_trucks)
for i in df.index:
    for item in VOT_OD:
        if (df.loc[i,'OD']==VOT_OD[item]['OD']) and (df.loc[i,'VOT']==VOT_OD[item]['VOT']):
            bg_dict[i+1] = {'OD':df.loc[i,'OD'],'VOT':df.loc[i,'VOT'], 'route':df.loc[i,'route'],
            'count':round(df.loc[i,'fraction']*VOT_OD[item]['count'])}

#print(bg_dict) #{1: {'OD': 'OD1', 'VOT': 1.8, 'route': 'route124', 'count': 12}, 2: {'OD': 'OD1', 'VOT': 1.8, 'route': 'route1234', 'count': 0},

#calculating increase in route times
for item in bg_dict:
    if bg_dict[item]['count']!=0:
        route = str(bg_dict[item]['route'])
        for i in range(0,len(route)-1):
            changed_link_trucks[int(route[i]+route[i+1])] = 0
for item in bg_dict:
    if bg_dict[item]['count']!=0:
        route = str(bg_dict[item]['route'])
        for i in range(0,len(route)-1):
            changed_link_trucks[int(route[i]+route[i+1])] += bg_dict[item]['count']     

if changed_link_trucks:     
    for item in changed_link_trucks:
        bg_link_trucks[item] = changed_link_trucks[item] + bg_link_trucks[item]

bg_link_times = link_time_function(bg_link_trucks) 
bg_route_times = route_time_function(bg_link_times)
increased_route_time = copy.deepcopy(bg_route_times)
#link_times = link_time_function(link_trucks)
#route_times = route_time_function(link_times)

for y in bg_route_times:
    increased_route_time[y]['time'] = bg_route_times[y]['time'] - route_times[y]['time'] 
total_payment = sum(payment_reward.values())
#calculating cost 
bg_assigned_customer = copy.deepcopy(assigned_customer)

for item in bg_assigned_customer:
    for element in increased_route_time:
        if bg_assigned_customer[item]['route']==element:
            bg_assigned_customer[item]['new_cost'] = increased_route_time[element]['time']*bg_assigned_customer[item]['VOT']+ bg_assigned_customer[item]['cost']
previous_cost = 0
for item in assigned_customer:
    previous_cost = assigned_customer[item]['cost'] + previous_cost
new_cost = 0
for item in bg_assigned_customer:
    new_cost = bg_assigned_customer[item]['new_cost'] + new_cost

########################calculate cost increase due to background traffic################################

#calculate benefit compared to UE
mechanism_cost = 0
total_UE_cost = 0
unassg_cost = 0
for item in assigned_customer:
    mechanism_cost += assigned_customer[item]['cost']
for item in customer_att:
    total_UE_cost += customer_att[item]['UE_cost']
for item in bg_dict:
    unassg_cost += UE_time[bg_dict[item]['OD']]*bg_dict[item]['VOT']*bg_dict[item]['count']

mechanism_cost = mechanism_cost - MC_factor*marg_cost + unassg_cost

#calculate cumulative time for mechanism and UE
cumulative_time_mechanism = 0
cumulative_time_UE = 0
for item in assigned_customer:
    for element in bg_route_times:
        if assigned_customer[item]['route']==bg_route_times[element]['route']:
            cumulative_time_mechanism = cumulative_time_mechanism + bg_route_times[element]['time']
for item in unassigned_customer:
        cumulative_time_mechanism = cumulative_time_mechanism + UE_time[customer_att[item]['OD']]
for item in customer_att:
        cumulative_time_UE = cumulative_time_UE + UE_time[customer_att[item]['OD']]

reward_count = 0
for item in payment_reward:
    if payment_reward[item]<0:
        #print(payment_reward[item])
        reward_count+=1
        
toc_overall = time.perf_counter()

#print(assigned_customer)
#print(bg_route_times)
#print(route_times)
#print(previous_cost)
#print(new_cost)
print('total requests = {}'.format(count_total))
print('total assigned requests = {}'.format(count_assigned))
print('total unassigned requests = {}'.format(count_unassigned))
print(f"reward given = {reward_count}")
print('initial account balance = {}'.format(initial_account))
print('final account balance = {}'.format(final_account))
print('balance gap considering MC increase = {}'.format(final_account-initial_account))
print('balance gap considering MC increase and background traffic = {}'.format(final_account-initial_account-(new_cost-previous_cost)))
print("marginal cost increase = {}".format(marg_cost))
print("total payment = {}".format(total_payment))
print("mechanism cost = {}".format(mechanism_cost))
print("total UE cost = {}".format(total_UE_cost))
print(f"cumulative mechanism time = {cumulative_time_mechanism}")
print(f"cumulative UE time = {cumulative_time_UE}")
print(f"total execution time = {toc_overall-tic_overall}")

#print("balance remaining after bg and marginal = {}".format((final_account-initial_account-(new_cost-previous_cost))-marg_cost))
print(link_times)
print(link_trucks)



