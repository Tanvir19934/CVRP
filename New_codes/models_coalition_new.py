from gurobipy import GRB, Model, quicksum
from typing import List
import heapq
from collections import defaultdict
import re
import copy
import time
from utils_new import ev_travel_cost, reconstruct_path, tsp_tour
from config_new import (
    col_dp_cutoff, battery_threshold, N, V, Q_EV, q, a, w_dv, w_ev, theta, tol, num_EV, gamma, 
    gamma_l, EV_velocity, GV_cost, unlimited_EV, dom_heuristic, rand_seed, best_obj, GV_cost, EV_cost
)
import random
random.seed(rand_seed)

class Label:
    def __init__(self, node, resource_vector, parent=None):
        self.node = node  # Current node
        self.resource_vector = resource_vector  # Resources consumed up to the current node
        self.parent = parent  # Parent label (previous node)
    def __lt__(self, other):
        # Define less than operator for priority queue (heapq), e.g., based on some resource (reduced cost)
        return self.resource_vector[0] < other.resource_vector[0]
    def __repr__(self):
        return f"Label(node={self.node}, resource_vector={self.resource_vector}, parent={self.parent})"

class SubProblem:
    def __init__(self, forbidden_set):
        self.forbidden_set = forbidden_set

    def feasibility_check(self, curr_node, extending_node, curr_load, curr_battery):
        q[0]=0
        if curr_node == 's':
            curr_node=0
        if extending_node == 't':
            extending_node=0
        new_load = curr_load + q[extending_node]
        if new_load>Q_EV:
            return None, None
        else:
            new_battery = curr_battery + (a[(curr_node,extending_node)]/EV_velocity)*(gamma+gamma_l*curr_load)

            if extending_node != 0 and  new_battery + (a[(0,extending_node)]/EV_velocity)*(gamma+gamma_l*new_load) > 1 - battery_threshold:
                return None, None
            elif extending_node==0 and new_battery > 1 - battery_threshold:
                return None, None

        return new_load, new_battery

    def calculate_reduced_cost(self, route, dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle, DV=False, curr=None, ext=None):

        reduced_cost = 0

        if DV:
            delta_sum = [dual_values_delta[i] for i in route if i!=0]
            for i in range(0,len(route)-1):
                reduced_cost += w_dv*a[(route[i],route[i+1])]
            reduced_cost+=-sum(delta_sum)
            return reduced_cost
        
        elif not DV:
            if curr.node == 's':
                curr_node = 0
            else: curr_node = curr.node
            if ext.node == 't':
                ext_node = 0
            else: ext_node = ext.node
            delta = dual_values_delta[ext_node]
            reduced_cost = curr.resource_vector[0] +  w_ev*a[(curr_node,ext_node)]
            reduced_cost += (theta-dual_values_subsidy)* (260*EV_cost*(a[(curr_node,ext_node)]/EV_velocity)*(gamma+gamma_l*(curr.resource_vector[1]))) 
            IR = dual_values_IR[ext_node]* (a[(ext_node,0)]*GV_cost*q[ext_node]+a[(ext_node,0)]*GV_cost)
            reduced_cost += -delta - IR # the dual value for vehicle is used at initial_resource_vector initializtion in dy_prog function
            return reduced_cost

    def calculate_reduced_cost_old(self, route, dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle, DV=False, curr=None, ext=None):

        reduced_cost = 0
        delta_sum = [dual_values_delta[i] for i in route if i!=0]
        if DV:
            for i in range(0,len(route)-1):
                reduced_cost += w_dv*a[(route[i],route[i+1])]
            reduced_cost+=-sum(delta_sum)
            return reduced_cost

        for i in range(0,len(route)-1):
            reduced_cost += w_ev*a[(route[i],route[i+1])]
        reduced_cost+= (theta-dual_values_subsidy)*ev_travel_cost(route)
        IR_sum = [dual_values_IR[i]* (a[(i,0)]*GV_cost*q[i]+a[(i,0)]*GV_cost) for i in route if i!=0]
        reduced_cost += -sum(delta_sum) - sum(IR_sum) - dual_values_vehicle #(note the + sign for IR_sum)

        return reduced_cost

    def label_domination_check(self, existing_label, current_label):
        """
        Check if existing_label dominates current_label.

        Resource vector format:
            [reduced_cost, current_load, battery_consumed, visited_set]

        Rules:
        - Reduced cost: lower is better
        - Current load: lower is better (means more remaining capacity)
        - Battery consumed: lower is better (means more remaining battery)
        - Visited set: smaller set (subset) is better
        - At least one resource strictly better
        """

        e_rc, e_load, e_batt, e_set = existing_label.resource_vector
        c_rc, c_load, c_batt, c_set = current_label.resource_vector

        # No worse in all dimensions (<= case)
        le = (
            e_rc   <= c_rc and
            e_load <= c_load and
            e_batt <= c_batt and
            e_set.issubset(c_set)
            #c_set.issubset(e_set)
        )

        # Strictly better in at least one dimension (< case)
        lt = (
            e_rc   < c_rc or
            e_load < c_load or
            e_batt < c_batt or
            (e_set < c_set)   # proper subset
            #(c_set < e_set)
        )

        return le and lt


    def label_domination_check_old(self, existing_label, current_label):

        # Assume resource_vector = [res0, res1, res2, visited_set]

        num_dims = 3
        existing_res = existing_label.resource_vector
        current_res   = current_label.resource_vector

        # 1) Check numeric domination
        numeric_le  = all(existing_res[i]  <= current_res[i]
                        for i in range(num_dims))
        numeric_lt  = any(existing_res[i]  <  current_res[i]
                        for i in range(num_dims))


        # 3) Combine them
        if numeric_le and True and numeric_lt:
            return True
        else:
            return False
    
    @staticmethod
    def ng_label_dominates(existing_label, candidate_label, tol=1e-9):
        """
        Returns True iff existing_label dominates candidate_label at the SAME node
        under NG-relaxation. Resource vector:
        [reduced_cost, current_load, battery_consumed, S_ng]
        where lower is better for all three numeric components.
        Set comparison uses S_existing ⊆ S_candidate (and strict ⊂ allowed for lt).
        """
        if existing_label.node != candidate_label.node:
            return False

        tol = 0
        e_rc, e_load, e_batt, S_e = existing_label.resource_vector
        c_rc, c_load, c_batt, S_c = candidate_label.resource_vector

        # Non-worse in all dimensions (<= with tolerance)
        le = (
            e_rc   <= c_rc   + tol and
            e_load <= c_load + tol and
            e_batt <= c_batt + tol 
            and S_e.issubset(S_c)
        )

        # Strictly better in at least one dimension
        lt = (
            e_rc   < c_rc   - tol or
            e_load < c_load - tol or
            e_batt < c_batt - tol 
            or (S_e < S_c)  # proper subset
        )

        return le and lt


    @staticmethod
    def ng_block(current_S, next_node):
        # Block revisiting only if next_node is in current NG memory
        return next_node in current_S
    
    @staticmethod
    def ng_update(S, current_node, next_node, NG):
        """
        Projection/update rule:
        S' = (S ∩ NG(next_node)) ∪ ({current_node} if current_node ∈ NG(next_node) else ∅) ∪ {next_node}
        Use frozenset so labels hash/compare nicely.
        """
        S_proj = S & NG[next_node]
        if current_node in NG[next_node]:
            S_proj = S_proj | {current_node}
        S_new = S_proj | {next_node}
        return frozenset(S_new)


    def dy_prog(self, dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle,
                feasibility_memo={}, IFB=False, NG=None):
        """
        Same signature, add NG dict (node -> set of neighbors). If NG is None, fall back to elementary.
        """
        U = []                     # PQ of labels
        L = defaultdict(list)      # labels stored per node
        N.extend(['s','t'])
        start_node = 's'

        # ---- initialize NG memory ----
        init_S = frozenset()  # start with empty memory (depot not tracked)
        initial_resource_vector = (-dual_values_vehicle, 0, 0, init_S)
        initial_label = Label(start_node, initial_resource_vector, None)

        heapq.heappush(U, initial_label)
        print("\nExecuting CG DP (NG) ...\n")
        neg_count = 0
        start = time.perf_counter()

        while U:
            current_label = heapq.heappop(U)
            current_node  = current_label.node

            # ----- dominance check at current node (using NG) -----
            is_dominated = False
            for label in L[current_node]:
                # Use NG-aware dominance
                if NG is None:
                    # fallback to your original checker
                    if self.label_domination_check(label, current_label):
                        is_dominated = True
                        break
                else:
                    if self.ng_label_dominates(label, current_label):
                        is_dominated = True
                        break

            if not is_dominated:
                # Store (so later arrivals can be compared against this)
                heapq.heappush(L[current_node], current_label)

                # ----- stop at sink -----
                if current_node == 't':
                    if current_label.resource_vector[0] < 0:
                        neg_count += 1
                    if IFB and neg_count >= col_dp_cutoff:
                        break
                    continue

                # ----- neighbors to extend -----
                neigh = list(set(N) - {current_node})
                if current_node == 's':
                    if 't' in neigh:
                        neigh.remove('t')
                else:
                    if 's' in neigh:
                        neigh.remove('s')

                # reconstruct for reduced cost + memo keys (unchanged)
                current_path = reconstruct_path(current_label)
                current_path_load = sum(q[i] for i in current_path if i != 0)
                rc, load, batt, S = current_label.resource_vector

                for new_node in neigh:
                    # Forbidden arcs remain
                    c_conv = 0 if current_node in ('s','t') else current_node
                    n_conv = 0 if new_node    in ('s','t') else new_node
                    if (c_conv, n_conv) in self.forbidden_set:
                        continue

                    # ---- NG: block only if new_node is in current NG memory ----
                    if NG is not None and self.ng_block(S, new_node):
                        continue

                    # Build new path for feasibility memo + reduced cost calc (unchanged)
                    if new_node == 't':
                        new_path = current_path + [0]
                    else:
                        new_path = current_path + [new_node]

                    if tuple(new_path) in feasibility_memo:
                        new_load, new_battery = feasibility_memo[tuple(new_path)]
                    else:
                        new_load, new_battery = self.feasibility_check(
                            current_node, new_node, load, batt
                        )
                        if current_path_load != Q_EV and new_load is not None:
                            feasibility_memo[tuple(new_path)] = (new_load, new_battery)

                    if new_load is None:
                        continue

                    # ---- NG: update memory ----
                    if NG is not None:
                        S_new = self.ng_update(S, current_node, new_node, NG)
                    else:
                        # Elementary fallback: carry full visited set
                        S_new = S | {new_node}  # (your original did union on visited)

                    # Build new label
                    new_label = Label(new_node, (0.0, new_load, new_battery, S_new), current_label)

                    # Reduced cost calculation (unchanged)
                    reduced_cost = self.calculate_reduced_cost(
                        new_path, dual_values_delta, dual_values_subsidy, dual_values_IR,
                        dual_values_vehicle, False, current_label, new_label
                    )
                    new_label.resource_vector = (reduced_cost, new_load, new_battery, S_new)
                    heapq.heappush(U, new_label)
                    if reduced_cost < -tol and new_node == 't':
                        neg_count += 1
                    

            if (IFB and neg_count >= col_dp_cutoff) or neg_count >= 10000:
                break

        # Gather negative columns at sink (unchanged)
        sink_node = 't'
        new_routes = {}
        for item in L[sink_node]:
            route = reconstruct_path(item)  # full path via predecessors
            if len(route) != 3 and item.resource_vector[0] < 0 and abs(item.resource_vector[0]) > tol:
                # (Optional) filter here to keep only elementary routes if you like
                new_routes[tuple(route)] = item.resource_vector[0]

        N.remove('s'); N.remove('t')

        end = time.perf_counter()
        print(f"CG DP time: {end-start:.2f} seconds")
        return new_routes, feasibility_memo


class MasterProblem:

    def __init__(self, forbidden=[]):
        self.forbidden = forbidden
        self.y_r = {}
        self.p = {}
        self.r_set = set(tuple([0, node, 0]) for node in V if node != 0)

    def relaxedLP(self, branching_arc, extended_set, new_constraints = None, initial_lp=False) -> None:

        #override some config parameters
        q[0] = 0

        if extended_set:
            self.r_set.update(extended_set)

        c_r = {}
        a_r = {item:0 for item in self.r_set}

        d = copy.deepcopy(a)
        for item in self.r_set:
            if len(item)==3:
                if (0,item[1]) in self.forbidden:
                    d[0,item[1]] = best_obj*3
                elif (item[1],0) in self.forbidden:
                    d[item[1],0] = best_obj*3
                    
        for item in self.r_set:
            l = len(item)
            for i in range(l-1):
                a_r[item] += d[(item[i],item[i+1])]
            if l>3:
                c_r[item] = ev_travel_cost(item)
        
        delta = {}
        for route in self.r_set:
            for i in range(1,len(V)+1):
                if i in route:
                    delta[(i, route)] = 1
                else:
                    delta[(i, route)] = 0

        self.model = Model('master_problem')
        
        #DECISION VARIABLES
        for item in self.r_set:
            self.y_r[tuple(item)] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"y_r_[{item}]", lb=0)
        for i in N:
            self.p[i] = self.model.addVar(vtype=GRB.CONTINUOUS, name = f"p_{i}", lb=0)
        # self.p = self.model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="p")
        self.model.update()

        #CONSTRAINTS
        self.model.addConstrs((quicksum(delta[(i, route)] * self.y_r[route] for route in self.r_set) == 1 for i in N), name=f"delta_")
        self.model.addConstr((quicksum(c_r[route]*self.y_r[route] for route in self.r_set if len(route)>3) - quicksum(self.p[i] for i in N)) >= 0, name="subsidy")
        if unlimited_EV:
            self.model.addConstr((quicksum(self.y_r[route] for route in self.r_set if len(route)>3) <= num_EV*10000), name="vehicle")
        else: 
            self.model.addConstr((quicksum(self.y_r[route] for route in self.r_set if len(route)>3) <= num_EV), name="vehicle")
        self.model.addConstrs(((a[(i,0)]*GV_cost*q[i]+a[(i,0)]*GV_cost)*(quicksum(delta[(i, route)] * self.y_r[route] for route in self.r_set if len(route)>3)) - self.p[i] >= 0 for i in N), name=f"IR_")
        self.model.update()

        if new_constraints:
            for route, cost in new_constraints:
                self.model.addConstr((quicksum(self.p[i] for i in route if i!=0) <= cost), name=f"stability_{route}")
                self.model.update()

        #SET OBJECTIVE
        self.model.setObjective((quicksum(a_r[route]*self.y_r[route] for route in self.r_set if len(route)==3))*w_dv + (quicksum(a_r[route]*self.y_r[route] for route in self.r_set if len(route)>3))*w_ev +  theta*(quicksum(c_r[route]*self.y_r[route] for route in self.r_set if len(route)>3) - quicksum(self.p[i] for i in N)))
        self.model.update()

        self.model.modelSense = GRB.MINIMIZE
        self.model.Params.OutputFlag = 0
        self.model.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/New_codes/master_prob.lp")
        self.model.optimize()

        try:
            self.model.computeIIS()
            self.model.write("/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/CVRP/Codes/New_codes/master_prob_iis.ilp")
        except: pass
  
        if self.model.status!=GRB.OPTIMAL:
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

        try:
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
        except:
            return None, None, None, None

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
