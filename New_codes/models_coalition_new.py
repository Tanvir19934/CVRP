from gurobipy import GRB, Model, quicksum
from typing import List
import heapq
from collections import defaultdict
import re
from utils_new import *
from config_new import battery_threshold, N, V, Q_EV, q, a, w_dv, w_ev, theta, tol, num_EV, st, gamma, gamma_l, T_max_EV, EV_velocity, GV_cost, unlimited_EV, row_dp_cutoff
import numpy as np
rnd = np.random
rnd.seed(42)
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
        self.adj= adj
        self.forbidden_set = forbidden_set

    def feasibility_check(self, curr_node, extending_node, curr_time = 0, curr_load = 0, curr_battery = 0, curr_distance = 0):
        
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

        return new_time, new_load, new_battery, new_distance 

    def calculate_reduced_cost(self, route, dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle):

        reduced_cost = 0
        delta_sum = [dual_values_delta[i] for i in route if i!=0]

        if len(route)==3:
            for i in range(0,len(route)-1):
                reduced_cost += w_dv*a[(route[i],route[i+1])]
            reduced_cost+=-sum(delta_sum)
            return reduced_cost

        for i in range(0,len(route)-1):
            reduced_cost += w_ev*a[(route[i],route[i+1])]

        reduced_cost+= (theta-dual_values_subsidy)*ev_travel_cost(route)
        IR_sum = [dual_values_IR[i]*(a[(i,0)]*GV_cost*q[i]+a[(i,0)]*GV_cost) for i in route if i!=0]
        #if [0, 7, 3, 1, 0]==route or [0, 7, 3, 0]==route:
        #    pass 
        reduced_cost += -sum(delta_sum) - sum(IR_sum) + dual_values_vehicle #(note the + sign for IR_sum)
        
        return reduced_cost
    
    def dy_prog(self, dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle, feasibility_memo={}):
        # Initialize the sets of labels
        U = []  # Priority queue for undominated labels
        L = defaultdict(list)  # Dictionary to store the sets of labels at each node
        N.extend(['s','t'])
        start_node = 's'
        initial_resource_vector = (0, 0, 0, 0)  # (reduced_cost, load, battery, time)
        initial_label = Label(start_node, initial_resource_vector, None)
        heapq.heappush(U, initial_label)
        print("Executing CG DP...")
        
        while U:
            current_label = heapq.heappop(U)
            current_node = current_label.node
            # Check for dominance and add label to the set of labels if not dominated
            is_dominated = False
            for label in L[current_node]:

                if current_label.resource_vector[0]>label.resource_vector[0] and \
                 current_label.resource_vector[1]>label.resource_vector[1] and \
                current_label.resource_vector[2]>label.resource_vector[2]:
                    is_dominated = True
                    break
                
            if not is_dominated:
                    in_path_already = False
                    in_path_already = partial_path(current_label,current_node)
                    if in_path_already:
                        continue
                    L[current_node].append(current_label)

                    #Extend the label along all arcs leaving the current node
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
                        if (current_node_converted,new_node_converted) in self.forbidden_set:
                            continue
                        if new_node=='t':
                            new_path = reconstruct_path(current_label) + [0]
                        else:
                            new_path = reconstruct_path(current_label) + [new_node] + [0]
                        
                        #if new_path==[0,7,3,1] or new_path==[0,7,3]:
                        #    pass
                        
                        if tuple(new_path) in feasibility_memo:
                            new_time, new_load, new_battery, new_cost = feasibility_memo[tuple(new_path)]
                        else:
                            new_time, new_load, new_battery, new_cost   = self.feasibility_check(current_node, new_node, current_label.resource_vector[-1], current_label.resource_vector[1], current_label.resource_vector[2], current_label.resource_vector[0])
                            feasibility_memo[tuple(new_path)] = (new_time, new_load, new_battery, new_cost)

                        if new_cost:
                            resource_vector = (new_cost, new_load, new_battery, new_time)
                            new_label = Label(new_node, resource_vector, current_label)
                            reduced_cost = self.calculate_reduced_cost(new_path, dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle)
                            new_label.resource_vector = (reduced_cost, new_load, new_battery, new_time) #update the resource vector with reduced cost

                            if reduced_cost<tol:
                                #Add all feasible extensions to U (if no constraint violation)
                                heapq.heappush(U, new_label)
                                if new_node=='t':
                                    L[new_node].append(new_label)
                        #
                        #if len(L['t'])>100:
                        #    break

        sink_node = 't'
        new_routes = {}
        for item in L[sink_node]:
            new_routes[tuple(reconstruct_path(item))] = item.resource_vector[0]
            #new_routes[tuple([0, 12, 7, 11, 0])] = -100
            #new_routes[tuple(reconstruct_path(item)[::-1])] = -1

        N.remove('s')
        N.remove('t')

        return new_routes, feasibility_memo

class MasterProblem:

    def __init__(self, adj, forbidden=[], allowed=[]):
        self.adj = adj
        self.forbidden = forbidden
        self.allowed = allowed
        self.y_r = {}
        self.p = {}
        self.r_set = [[0, node, 0] for node in V if node != 0]
        self.r_set = [tuple(path) for path in self.r_set]  # Convert lists to tuples

    def relaxedLP(self, extended_set, new_constraints = None, initial_lp=False) -> None:

        #override some config parameters
        q[0] = 0

        for item in self.forbidden:
            if item[0]==0 or item[1]==0:
                if tuple((0,item[0],0)) in self.r_set:
                    self.r_set.remove(tuple((0,item[0],0)))
                if tuple((0,item[1],0)) in self.r_set:
                    self.r_set.remove(tuple((0,item[1],0)))
        
        #def contains_forbidden_arc(path, forbidden):
        #    for i in range(len(path) - 1):
        #        if (path[i], path[i+1]) in forbidden:
        #            return True
        #    return False

        if extended_set:
            self.r_set.update(extended_set)

        # Filter out tuples that contain forbidden arcs
        #filtered_r_set = {path for path in self.r_set if not contains_forbidden_arc(path, self.forbidden)}
        # Update self.r_set
        #self.r_set = filtered_r_set

        if initial_lp:
            self.r_set = [[0, node, 0] for node in V if node != 0]
        
        self.r_set = set(tuple(route) for route in self.r_set)

        c_r = {item:0 for item in self.r_set}
        a_r = {item:0 for item in self.r_set}

        for item in self.r_set:
            l = len(item)
            for i in range(l-1):
                a_r[item] += a[(item[i],item[i+1])]
            c_r[item] = ev_travel_cost(item)
        
        delta = {}
        for route in self.r_set:
            for i in range(1,len(V)+1):
                delta[(i, route)] = 1 if i in route else 0

        self.model = Model('master_problem')
        
        #DECISION VARIABLES
        for item in self.r_set:
            self.y_r[tuple(item)] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"y_r_[{item}]", lb=0, ub=1)
        for i in N:
            self.p[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name = f"p_{i}", lb=0)
        self.model.update()

        #CONSTRAINTS
        self.model.addConstrs((quicksum(delta[(i, route)] * self.y_r[route] for route in self.r_set) >= 1 for i in N), name=f"delta_")
        self.model.addConstr((quicksum(c_r[route]*self.y_r[route] for route in self.r_set if len(route)>3) - quicksum(self.p[i] for i in N)) >= 0, name="subsidy")
        if unlimited_EV:
            self.model.addConstr((quicksum(self.y_r[route] for route in self.r_set if len(route)>3) <= num_EV*10000), name="vehicle")
        else:
            self.model.addConstr((quicksum(self.y_r[route] for route in self.r_set if len(route)>3) <= num_EV), name="vehicle")
        self.model.addConstrs(((a[(i,0)]*GV_cost*q[i]+a[(i,0)]*GV_cost)*(quicksum(delta[(i, route)] * self.y_r[route] for route in self.r_set if len(route)!=3)) - self.p[i] >= 0 for i in N), name=f"IR_")
        self.model.update()

        if new_constraints:
            for route, cost in new_constraints:
                self.model.addConstr((quicksum(self.p[i] for i in route if i!=0) <= cost), name=f"stability_{route}")
        self.model.update()


        
        #SET OBJECTIVE
        self.model.setObjective((quicksum(a_r[route]*self.y_r[route] for route in self.r_set if len(route)==3))*w_dv + (quicksum(a_r[route]*self.y_r[route] for route in self.r_set if len(route)!=3))*w_ev +  theta*(quicksum(c_r[route]*self.y_r[route] for route in self.r_set if len(route)>3) - quicksum(self.p[i] for i in N)))
        self.model.update()

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
                
        return p_result, y_r_result_final, self.model, self.model.status

    def getDuals(self) -> List[int]:
        dual_values_subsidy, dual_values_vehicle = 0, 0
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
            elif constr.ConstrName.startswith("vehicle"):
                dual_values_vehicle = constr.Pi
        
        return dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle

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
    
    def dy_prog(self, N, q, Q_EV):
        q = {i: q[i] for i in q if i != 0}  
        customers = list(q.keys())  
        weights = list(q.values())  
        N = len(customers)  

        dp = [set() for _ in range(Q_EV + 1)]
        dp[0].add(tuple())

        print("Executing RG DP...")  

        # Fill DP table
        for i in range(N):  
            customer_id = customers[i]
            weight = weights[i]
            for w in range(Q_EV, -1, -1):
                if dp[w]:
                    new_weight = w + weight
                    if new_weight <= Q_EV:
                        # Temporary set to store new combinations
                        new_combinations = set()
                        for combination in dp[w]:
                            new_combinations.add(combination + (customer_id,))
                        # Update dp[new_weight] after iteration
                        dp[new_weight].update(new_combinations)

        valid_combinations = []
        for w in range(1, Q_EV + 1): 
            valid_combinations.extend(dp[w])

        final_combinations = []    
        
        for item in valid_combinations:
            if len(item)==1:
                item = [0, item[0], 0]
            else:
                item = [0] + list(item) + [0]
            final_combinations.append(tuple(item))
            

        return final_combinations


    def generate_constr(self,tsp_memo,p,L):
        p['p_0']=0
        new_routes = set()
        print("Generating rows...")
        for item in L:
            candidate_route = list(item)
            sorted_candidate_route = tuple([0]+ sorted(candidate_route[1:-1])+ [0])
            if sorted_candidate_route in tsp_memo:
                tsp_cost = tsp_memo[sorted_candidate_route][-1]
            else: 
                candidate_route, tsp_cost = tsp_tour(sorted_candidate_route)
                tsp_memo[sorted_candidate_route] = (candidate_route,tsp_cost)
            payments = [p[f"p_{i}"] for i in candidate_route if i!=0]
            if tsp_cost<sum(payments)-tol:
                new_routes.add((tuple(candidate_route),tsp_cost))
            if len(new_routes)>row_dp_cutoff:
                return new_routes, tsp_memo
        return new_routes, tsp_memo