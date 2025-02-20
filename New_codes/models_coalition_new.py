from gurobipy import GRB, Model, quicksum
from typing import List
from config_new import *
import heapq
from collections import defaultdict
import re
from utils_new import *
from config_new import a, q, N, theta, w_dv, w_ev
import random


# Define a label structure
class Label:
    def __init__(self, node, resource_vector, parent=None):
        self.node = node  # Current node
        self.resource_vector = resource_vector  # Resources consumed up to the current node
        self.parent = parent  # Parent label (previous node)
    def __lt__(self, other):
        # Define less than operator for priority queue (heapq), e.g., based on some resource (distance)
        return self.resource_vector[0] < other.resource_vector[0]
    def __repr__(self):
        return f"Label(node={self.node}, resource_vector={self.resource_vector}, parent={self.parent})"



class SubProblem:
    def __init__(self,  adj, forbidden_set):
        #self.subproblem_model = model
        self.adj= adj
        self.forbidden_set = forbidden_set
        #self.memo={}
    #@staticmethod
    def feasibility_check(self, curr_node, extending_node, curr_time = 0, curr_load = 0, curr_battery = 0, curr_distance = 0):
        key = (curr_node, extending_node, curr_time, curr_load, curr_battery, curr_distance)
        
        # Check if the result is already in the memoization cache
        #if key in self.memo:
        #    return self.memo[key]

        #new_distance, new_load, new_battery, new_time

        if curr_node == 's':
            curr_node=0
        if extending_node == 't':
            extending_node=0
        new_load = curr_load + q[extending_node]
        if new_load>Q_EV:
            return None, None, None, None
        else:
            new_battery = curr_battery + (a[(curr_node,extending_node)]/EV_velocity)*(gamma+gamma_l*curr_load)
            if new_battery + (a[(0,extending_node)]/EV_velocity)*(gamma+gamma_l*new_load) > 1 - battery_threshold:
                return None, None, None, None
            else:
                new_time = curr_time + (st[curr_node]+(a[(curr_node,extending_node)]/EV_velocity))
                if new_time + (st[extending_node]+(a[(0,extending_node)]/EV_velocity)) > T_max_EV:
                    return None, None, None, None
        new_distance = curr_distance + a[(curr_node,extending_node)]
        #self.memo[key] = (new_time, new_load, new_battery, new_distance)

        return new_time, new_load, new_battery, new_distance 

    def calculate_reduced_cost(self, label, dual_values_delta, dual_values_subsidy, dual_values_IR):
        pointer = label
        partial_route = [label.node]
        while pointer.parent:
            partial_route.append(pointer.parent.node)
            pointer = pointer.parent
        partial_route = partial_route[::-1]
        if partial_route[-1]!='t':
            partial_route.append('t')
        partial_route[0]=0
        partial_route[-1]=0

        if partial_route==[0,10,1,0]:
            pass

        reduced_cost = 0
        for i in range(0,len(partial_route)-1):
            reduced_cost += w_ev*a[(partial_route[i],partial_route[i+1])]
        
        reduced_cost+= (theta-dual_values_subsidy)*ev_travel_cost(partial_route)
        delta_sum = [dual_values_delta[i] for i in partial_route if i!=0]
        IR_sum = [dual_values_IR[i]*(2*a[(i,0)]*GV_cost*q[i]) for i in partial_route if i!=0]
        reduced_cost += -sum(delta_sum) + sum(IR_sum) #(note the + sign for IR_sum)
        
        return reduced_cost
    
    def dy_prog(self, dual_values_delta, dual_values_subsidy, dual_values_IR):
        # Initialize the sets of labels
        U = []  # Priority queue for undominated labels
        L = defaultdict(list)  # Dictionary to store the sets of labels at each node
        N.extend(['s','t'])
        # Step 1: Initialize with the starting node
        start_node = 's'
        initial_resource_vector = (0, 0, 0, 0)  # (distance, load, battery, time)
        initial_label = Label(start_node, initial_resource_vector, None)
        heapq.heappush(U, initial_label)
        #cutoff = len(N)*5000000000000
        
        # Step 2: Main loop for label setting
        while U:
            #if len(L['t'])>=cutoff:
            #    break
            # 2a. Remove first label (label with the least resource cost in heap)
            current_label = heapq.heappop(U)
            current_node = current_label.node
  
            # 2c. Check for dominance and add label to the set of labels if not dominated
            is_dominated = False
            for label in L[current_node]:

                if current_label.resource_vector[0]>label.resource_vector[0] and \
                 current_label.resource_vector[1]<label.resource_vector[1] and \
                current_label.resource_vector[2]>label.resource_vector[2] and \
                current_label.resource_vector[3]>label.resource_vector[3]:
                    is_dominated = True
                    break
                
            if not is_dominated and current_label not in L[current_node]:
                    in_path_already = False
                    in_path_already = partial_path(current_label,current_node)
                    if in_path_already:
                        continue
                    L[current_node].append(current_label)
                    # 2c2. Extend the label along all arcs leaving the current node
                    neigh = list(set(N)-set([current_node]))
                    if current_node=='s':
                        neigh.remove('t')
                    if current_node!='s':
                        neigh.remove('s')
                    if current_node=='t':
                        continue
                    
                    for new_node in neigh:
                        if new_node=='t' or new_node=='s':
                            new_node_converted = 0
                        else: new_node_converted=new_node
                        if current_node=='t' or current_node=='s':
                            current_node_converted=0
                        else: current_node_converted = current_node
                        if new_node_converted not in self.adj[current_node_converted] or (current_node_converted,new_node_converted) in self.forbidden_set:
                            continue

                        if (new_node!='s' and current_node!='t'):
                            new_time, new_load, new_battery, new_distance   = self.feasibility_check(current_node, new_node, current_label.resource_vector[-1], current_label.resource_vector[1], current_label.resource_vector[2], current_label.resource_vector[0])

                            reduced_cost = False
                        if new_distance:
                            resource_vector = (new_distance, new_load, new_battery, new_time)
                            #new_resource_vector = tuple(map(sum, zip(current_label.resource_vector, resources)))
                            new_label = Label(new_node, resource_vector, current_label)
                            reduced_cost = self.calculate_reduced_cost(new_label, dual_values_delta, dual_values_subsidy, dual_values_IR)
                            new_label.resource_vector = (reduced_cost, new_load, new_battery, new_time) #update the resource vector with reduced cost
                            if reduced_cost<-tol:
                                # 2c3. Add all feasible extensions to U (if no constraint violation)
                                heapq.heappush(U, new_label)
                                if new_node=='t':
                                    L[new_node].append(new_label)

        # Step 3: Select the best label in L_t (sink node)
        sink_node = 't'
        #best_label = min(L[sink_node], key=lambda x: x.resource_vector[0]) if L[sink_node] else None
        # Output the path corresponding to the best label



        new_routes = []
        for item in L[sink_node]:
            candidate_route = reconstruct_path(item)
            if len(candidate_route)!=3:
                new_routes.append(reconstruct_path(item))
        for item in new_routes:
            item[0]=0
            item[-1]=0

        N.remove('s')
        N.remove('t')

        return new_routes ########################not optimizing the tours to avoid adding nodes from forbidden set###########

class MasterProblem:

    def __init__(self, adj, forbidden=[], allowed=[]):
        self.adj = adj
        self.forbidden = forbidden
        self.allowed = allowed
        self.r_set = [[0, node, 0] for node in V if node != 0]
        #self.model = Model('master_problem')
        self.y_r = {}
        self.p = {}


    def relaxedLP(self, extended_set, new_constraints = None) -> None:

        #override some config parameters
        q[0] = 0

        for item in self.forbidden:
            if item[0]==0 or item[1]==0:
                if tuple((0,item[0],0)) in self.r_set:
                    self.r_set.remove(tuple((0,item[0],0)))
                if tuple((0,item[1],0)) in self.r_set:
                    self.r_set.remove(tuple((0,item[1],0)))

        if extended_set:
            self.r_set.update(extended_set)
        
        self.r_set = set(tuple(route) for route in self.r_set)

        if self.allowed:
            for item in self.allowed:
                if item[0]!=0 and item[1]!=0:
                    s=list(item)
                    s.append(0)
                    s.insert(0,0)
                    self.r_set.add(tuple(s))
                elif item[0]==0 and item[1]!=0:
                    s=list(item)
                    s.append(0)
                    self.r_set.add(tuple(s))
                elif item[0]!=0 and item[1]==0:
                    s=list(item)
                    s.insert(0,0)
                    self.r_set.add(tuple(s))

        c_r = {item:0 for item in self.r_set}
        a_r = {item:0 for item in self.r_set}

        for item in self.r_set:
            l = len(item)
            for i in range(l-1):
                a_r[item] += a[(item[i],item[i+1])]
            c_r[item] = ev_travel_cost(item)
        

        delta = {}
        # Iterate over all routes and nodes
        for route in self.r_set:
            for i in range(1,len(V)+1):  # Nodes from 0 to 10
                # Set delta_i,r = 1 if node i is in route r, otherwise 0
                delta[(i, route)] = 1 if i in route else 0

        self.model = Model('master_problem')
        
        
        #DECISION VARIABLES
        for item in self.r_set:
            self.y_r[tuple(item)] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"y_r_[{item}]", lb=0, ub=1)
        for i in N:
            self.p[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name = f"p_{i}", lb=0)
        self.model.update()

        #CONSTRAINTS
        self.model.addConstrs((quicksum(delta[(i, route)] * self.y_r[route] for route in self.r_set) == 1 for i in N), name=f"delta_")
        self.model.addConstr((quicksum(c_r[route]*self.y_r[route] for route in self.r_set if len(route)>3) >= quicksum(self.p[i] for i in N)), name="subsidy")
        self.model.addConstrs((self.p[i] <= (a[(i,0)]*GV_cost*q[i]+a[(i,0)]*GV_cost)*(quicksum(delta[(i, route)] * self.y_r[route] for route in self.r_set if len(route)!=3)) for i in N), name=f"IR_")
        self.model.update()

        if new_constraints:
            for route, cost in new_constraints:
                self.model.addConstr((quicksum(self.p[i] for i in route if i!=0) <= cost), name=f"stability_{route}")
        self.model.update()

 
        
        
        #SET OBJECTIVE
        self.model.setObjective((quicksum(a_r[route]*self.y_r[route] for route in self.r_set if len(route)==3))*w_dv + (quicksum(a_r[route]*self.y_r[route] for route in self.r_set if len(route)!=3))*w_ev +  theta*(quicksum(c_r[route]*self.y_r[route] for route in self.r_set if len(route)>3) - quicksum(self.p[i] for i in N)))
        self.model.update()

        #self.model.setParam('Threads', 8)  # Use 8 threads for solving
        self.model.modelSense = GRB.MINIMIZE
        self.model.Params.OutputFlag = 0
        self.model.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/New_codes/master_prob.lp")
        self.model.optimize()

        try:
            self.model.computeIIS()
            self.model.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/New_codes/master_prob_iis.lp")
        except: pass
  
        if self.model.status!=2:
            return None, None, self.model, self.model.status

        def get_vars(item,opt_route):
            vars = [var for var in opt_route.getVars() if f"{item}" in var.VarName]
            names = opt_route.getAttr('VarName', vars)
            values = opt_route.getAttr('X', vars)
            return dict(zip(names, values))

        y_r_result = get_vars('y_r',self.model)
        p_result = get_vars('p',self.model)
        y_r_result_final = {}
        for item in y_r_result:
            if y_r_result[item]>0:
                y_r_result_final[item] = y_r_result[item]
                print(f"rlxd_{item}={y_r_result[item]}")
        if self.get_RMP_cost() < 0:
            pass
                
        return p_result, y_r_result_final, self.model, self.model.status

    def getDuals(self) -> List[int]:
        dual_values_subsidy=0
        dual_values_delta, dual_values_IR = defaultdict(float),defaultdict(float)

        for constr in self.model.getConstrs():
            if constr.ConstrName.startswith("delta"):
                # Extract the route and node index from the constraint name
                i = int(re.search(r'\[(\d+)\]', constr.ConstrName).group(1))
                dual_values_delta[i] = constr.Pi
            elif constr.ConstrName.startswith("subsidy"):
                dual_values_subsidy = constr.Pi
            elif constr.ConstrName.startswith("IR"):
                i = int(re.search(r'\[(\d+)\]', constr.ConstrName).group(1))
                dual_values_IR[i] = constr.Pi
        
        return dual_values_delta, dual_values_subsidy, dual_values_IR


    def get_RMP_solution(self) -> List[int]:
       
       vars = [var for var in self.model.getVars() if "y_r" in var.VarName]
       names = self.model.getAttr('VarName', vars)
       values = self.model.getAttr('X', vars)
       y_r_result_original = dict(zip(names, values))
       y_r_result = {}
       for item in y_r_result_original:
           if y_r_result_original[item]>0:
               y_r_result[item] = y_r_result_original[item]

       return y_r_result

    def get_RMP_cost(self) -> int:
        obj = self.model.getObjective()
        return obj.getValue()
    

class RowGeneratingSubProblem:
    def __init__(self,  adj, forbidden_set):
        self.adj= adj
        self.forbidden_set = forbidden_set

    def dy_prog(self):
        # Initialize the sets of labels
        U = []  # Priority queue for undominated labels
        L = defaultdict(list)  # Dictionary to store the sets of labels at each node
        #N.extend(['s','t'])
        # Step 1: Initialize with the starting node
        start_node = random.choice(N)
        initial_resource_vector = q[start_node]
        initial_label = Label(start_node, initial_resource_vector, None)
        U.append(initial_label)
        
        #cutoff = len(N)*5000000000000
        
        # Step 2: Main loop for label setting
        while U:
            #if len(L['t'])>=cutoff:
            #    break
            # 2a. Remove first label (label with the least resource cost in heap)
            current_label = U.pop()
            current_node = current_label.node

            in_path_already = False
            in_path_already = partial_path(current_label,current_node)
            if in_path_already:
                continue
            if current_node!=0:
                L[current_node].append(current_label)
            # 2c2. Extend the label along all arcs leaving the current node
            neigh = list(set(N)-set([current_node]))

            for new_node in neigh:
                if new_node not in self.adj[current_node] or current_label.resource_vector + q[new_node] > Q_EV or (current_node,new_node) in self.forbidden_set:
                    continue
                else:
                    U.append(Label(new_node, current_label.resource_vector + q[new_node], current_label))
        return L
                
    def generate_routes(self,L,p):
        new_routes = set()   
        for item in L:
            for elem in L[item]: 
                if elem.parent:
                    candidate_route = reconstruct_path(elem)
                    if len(candidate_route)==1:
                        candidate_route = [0, candidate_route[0], 0]
                    else:
                        candidate_route = [0] + candidate_route + [0]
                    payments = [p[f"p_{i}"] for i in candidate_route if i!=0]
                    tsp_cost, candidate_route = tsp_tour(candidate_route)
                    if candidate_route and tsp_cost-sum(payments)<-tol:
                        new_routes.add((tuple(candidate_route),round(tsp_cost, 3)))
        if new_routes:
            return new_routes

    