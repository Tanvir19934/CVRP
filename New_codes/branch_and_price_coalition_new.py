from gurobipy import GRB
import heapq
import os
from pricing_coalition_new import column_generation
from config_new import V, Q_EV, q, NODES,k
from collections import defaultdict
import copy
import time
import random
from utils_new import print_solution, save_to_excel, tsp_tour
import pandas as pd
import logging
from itertools import combinations


random.seed(42)

class Node:  
    
    def __init__(self, depth, name, adj, forbidden, allowed, parent):
        #self.model = model
        self.depth = depth
        self.solution = None
        self.obj_val = None
        self.name = name
        self.adj = adj
        self.parent = parent
        self.forbidden = forbidden
        self.allowed = allowed
        self.not_fractional = False
        print("depth =",self.depth)  #sanity check

    def __lt__(self, other):
        return self.obj_val > other.obj_val  # For heap implementation. The heapq.heapify() will heapify the list based on this criteria.

def generate_routes(L):
    tsp_memo = {}
    for item in L:
        for idx, elem in enumerate(L[item]): 
            if elem.parent:
                candidate_route = []
                current = elem
                while current:
                    candidate_route.insert(0, current.node)
                    current = current.parent
                #candidate_route = reconstruct_path(elem)
                if len(candidate_route)==1:
                    candidate_route = [0, candidate_route[0], 0]
                else:
                    candidate_route = [0] + candidate_route + [0]
                candidate_route,tsp_cost = tsp_tour(tuple(candidate_route))
                tsp_memo[tuple(candidate_route)] = tsp_cost
    return tsp_memo

def generate_tsp_cache(N, k):
    print(f"\nCreating initial TSP memo... \ntotal {N}C{k} combinations\n")
    tsp_memo = {}
    customers = list(range(1, N + 1))  # Assuming customers are labeled 1 to N
    all_combinations = []
    
    for i in range(1, k + 1):
        for comb in combinations(customers, i):
            all_combinations.append((0,) + comb + (0,))
    for item in all_combinations:
        candidate_route, tsp_cost = tsp_tour(item)
        tsp_memo[item] = (tuple(candidate_route),tsp_cost)

    print(f"Finished creating initial TSP memo")
    time.sleep(1)
    return tsp_memo

def branching() -> None:

    adj = {}
    q[0] = 0
    forbidden_set = set()
    for node in V:
        adj[node] = []
        for n in V:
            if n != node:
                if q[node] + q[n] <= Q_EV:
                    adj[node].append(n)  # Add valid arcs to adj
                else:
                    forbidden_set.add((node, n))  # Add violating arcs to forbidden_set
                    forbidden_set.add((n,node))  # Add violating arcs to forbidden_set
    
    #row_generating_subproblem = RowGeneratingSubProblem(adj, forbidden_set)
    #start_1 = time.perf_counter()
    #L = row_generating_subproblem.dy_prog()
    #end_1 = time.perf_counter()
    #RG_DP_time = end_1-start_1
    log_dir = "New_codes/LogFiles"
    os.makedirs(log_dir, exist_ok=True) 
    filename = f"{log_dir}/bpc_{NODES}.log" 
    logging.basicConfig(filename=filename, level=logging.INFO, filemode='w', format='%(asctime)s - %(message)s')
    start_2 = time.perf_counter()
    tsp_memo = generate_tsp_cache(NODES, k)
    end_2 = time.perf_counter()
    tsp_cache_time = end_2-start_2
    logging.info(f"Time taken to generate tsp cache: {tsp_cache_time:0.2f}")
    
    global Total_CG_iteration, Total_RG_iteration, Total_RG_time, Total_CG_time
    global Total_RG_DP_time, Total_CG_DP_time, Total_execution_time, Total_LP_time
    Total_CG_iteration,  Total_RG_iteration,  Total_RG_time,  Total_CG_time,  Total_RG_DP_time,  Total_CG_DP_time, Total_LP_time = 0, 0, 0, 0, 0, 0, 0
    Total_num_lp = 0
    feasibility_memo={}
    
    
    # Create the root node by solving the initial rmp
    y_r_result, not_fractional, master_prob_model, root_obj_val, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, num_lp, L, root_constraints = column_generation(adj, forbidden_set= {}, allowed_set = [], tsp_memo=tsp_memo, L = None, feasibility_memo=feasibility_memo, initial = True, root_constraints=set())
    track_time_iterations(CG_iteration, RG_iteration, RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time)
    Total_num_lp+=num_lp
    if not_fractional:
        print("Optimal solution found at the root node: did not need branching")
        obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes = print_solution(master_prob_model)
        print(f"Total CG iterations: {Total_CG_iteration}")
        print(f"Total RG iterations: {Total_RG_iteration}")
        print(f"Total nodes explored: {num_nodes_explored}")
        print(f"Total RG time: {Total_RG_time}")
        print(f"Total CG time: {Total_CG_time}")
        print(f"Total RG DP time: {Total_RG_DP_time}")
        print(f"Total CG DP time: {Total_CG_DP_time}")
        print(f"Total LP time: {Total_LP_time:.2f}")
        print(f"tsp cache time: {tsp_cache_time}")
        print("LP gap: ", ((obj-root_obj_val)/obj)*100)
        print(f"Total number of LPs solved: {Total_num_lp}")
        return obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes, root_obj_val, 1, tsp_cache_time, Total_num_lp
    
    left_not_fractional = right_not_fractional = False
    root_node = Node(0, "root_node", adj, set(), set(), None)
    root_node.solution = y_r_result
    root_node.not_fractional = not_fractional
    root_node.obj_val = root_obj_val
    root_node.model = master_prob_model
    
    #initialize best node and the stack
    best_node = None
    best_obj = GRB.INFINITY
    stack = [root_node]
    tol = 0.0001
    heapq.heapify(stack)  #we are going to traverse in best first manner
    frac_count = 0

    outer_iter = 0
    num_nodes_explored = 0 
    
    
    #loop through the stack
    while stack:
        outer_iter+=1
        print(f"outer iteration count: {outer_iter}")
        print(f"Integer solution found so far = {frac_count}. Size of stack = {len(stack)}")
        
        node = heapq.heappop(stack)
        if node.not_fractional==True:
            frac_count+=1
            print(node.obj_val)
            if node.obj_val < best_obj:
                best_obj = node.obj_val
                best_node = node
        else:  
            if node.obj_val > best_obj:
                continue
            else:
                flow_vars = defaultdict(float)
                model_vars ={var.VarName: var.X for var in node.model.getVars()}
                var_names = list(model_vars.keys())
                fractional_var = {}
                for var in var_names:
                    if not (abs(model_vars[var] - 0) <= tol or abs(model_vars[var] - 1) <= tol) and var.split('[')[0]=='y_r_':
                        fractional_var[var] = model_vars[var]
                for item in fractional_var:
                    element = eval(item.split('[')[-1][0:-1])
                    for i in range(len(element)-1):
                        flow_vars[(element[i],element[i+1])]= model_vars[item]
                
                branching_arcs= {arc: flow_vars[arc] for arc in flow_vars if abs(flow_vars[arc] - 1) <= tol or abs(flow_vars[arc] - 0) <= tol or (flow_vars[arc] > tol and flow_vars[arc] < 1 - tol)}

                if not branching_arcs:
                    continue
                
                while True:
                    branching_arc = random.choice(list(branching_arcs.keys()))
                    if (branching_arc[0]==0 ) or (branching_arc[1]==0):
                        continue
                    else:
                        break
                

                # Create left branch node
                left_node = Node(node.depth + 1, f'{branching_arc}={0}', [], copy.deepcopy(node.forbidden) , copy.deepcopy(node.allowed), node)
                left_node.adj = copy.deepcopy(left_node.parent.adj)
                try:
                    left_node.adj[branching_arc[0]].remove(branching_arc[1])
                except: pass
                # enforcing branching_arc = 0
                left_node.forbidden.add(branching_arc)
                print(f"branching {branching_arc}={0}")
                if branching_arc==(5,9):
                    pass
                y_r_result, left_not_fractional, master_prob_model, obj_val, left_status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, num_lp, L, _ \
                      = column_generation(left_node.adj, left_node.forbidden, [], tsp_memo, L, feasibility_memo=feasibility_memo,initial = False, root_constraints=root_constraints.copy())
                track_time_iterations(CG_iteration, RG_iteration, RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time)
                num_nodes_explored+=1
                Total_num_lp+=num_lp

                if left_status==2:
                    if left_not_fractional:
                        left_node.not_fractional = True
                    left_node.obj_val = obj_val
                    left_node.model = master_prob_model
                    left_node.solution = y_r_result
                    heapq.heappush(stack, left_node)
                # Create right branch node
                right_node = Node(node.depth + 1, f'{branching_arc}={1}', [], copy.deepcopy(node.forbidden), copy.deepcopy(node.allowed), node)
                right_node.adj = copy.deepcopy(right_node.parent.adj)
                right_node.allowed.add(branching_arc)
                # enforcing branching_arc = 1
                right_node.adj[branching_arc[0]] = [branching_arc[1]]
                try:
                    right_node.adj[branching_arc[1]].remove(branching_arc[0])
                except: pass
                for item in V:
                    if item!=branching_arc[1] and item!=branching_arc[0]:
                        right_node.forbidden.add((branching_arc[0],item))
                        #right_node.forbidden.add((item,branching_arc[1])) ############added, review later################
                print(f"branching {branching_arc}={1}")
                y_r_result, right_not_fractional, master_prob_model, obj_val, right_status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, num_lp, L, _ = column_generation(right_node.adj,right_node.forbidden,[],tsp_memo,L,feasibility_memo=feasibility_memo,initial = False, root_constraints=root_constraints.copy())
                track_time_iterations(CG_iteration, RG_iteration, RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time)
                num_nodes_explored+=1
                Total_num_lp+=num_lp


                if right_status==2:
                    if right_not_fractional:
                        right_node.not_fractional = True  
                    right_node.obj_val = obj_val
                    right_node.model = master_prob_model
                    right_node.solution = y_r_result
                    heapq.heappush(stack, right_node)


    print(frac_count)
    if best_node:
        print("Optimal solution found:")
        obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes = print_solution(best_node.model)
    else:
        print("No optimal solution found.")
    
    print(f"Total CG iterations: {Total_CG_iteration}")
    print(f"Total RG iterations: {Total_RG_iteration}")
    print(f"Total nodes explored: {num_nodes_explored}")
    print(f"Total RG time: {Total_RG_time:.2f}")
    print(f"Total CG time: {Total_CG_time:.2f}")
    print(f"Total RG DP time: {Total_RG_DP_time:.2f}")
    print(f"Total CG DP time: {Total_CG_DP_time:.2f}")
    print(f"Total LP time: {Total_LP_time:.2f}")
    print(f"tsp cache time: {tsp_cache_time}")
    print(f"LP gap: {((obj-root_obj_val)/obj)*100}")
    print(f"Total number of LPs solved: {Total_num_lp}")
    
    return obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes, root_obj_val, num_nodes_explored, tsp_cache_time, Total_num_lp

def track_time_iterations(CG_iteration, RG_iteration, RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time):
    global Total_CG_iteration, Total_RG_iteration, Total_RG_time, Total_CG_time, Total_RG_DP_time, Total_CG_DP_time, Total_LP_time
    Total_CG_iteration+=CG_iteration
    Total_RG_iteration+=RG_iteration
    Total_RG_time+=RG_time
    Total_CG_time+=CG_time
    Total_CG_DP_time+=CG_DP_time
    Total_RG_DP_time+=RG_DP_time
    Total_LP_time+=LP_time


def main():
        start = time.perf_counter()
        obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes, root_obj_val, num_nodes_explored, tsp_cache_time, Total_num_lp = branching()
        end = time.perf_counter()
        print(f"Execution time for nodes={NODES}: {end - start}")
        Execution_time = end - start
        data = {
            "Nodes": [NODES],
            "Obj": [obj],
            "Total Miles": [total_miles],
            "EV miles": [EV_miles],
            "DV Miles": [total_miles-EV_miles],
            "Total payments": [Total_payments],
            "Subsidy": [Subsidy],
            "Execution time (sec.)": [Execution_time],
            "LP Gap": [((obj-root_obj_val)/obj)*100],
            "Number of nodes explored": [num_nodes_explored],
            "Total CG iterations": [Total_CG_iteration],
            "Total RG iterations": [Total_RG_iteration],
            "Total CG DP time": [Total_CG_DP_time],
            "Total RG DP time": [Total_RG_DP_time],
            "TSP cache time": [tsp_cache_time],
            "Total LP relaxation time": [Total_LP_time],
            "Total execution time": [Execution_time],
            "Total number of LPs solved": [Total_num_lp]
        }
        df = pd.DataFrame(data)
        file_name = "Results/results.xlsx" 
        save_to_excel(file_name, "Sheet1", df)
        data = {
            "Nodes": [NODES],
            "Payments": [payments],
            "Solution routes": [solution_routes]
        }
        df = pd.DataFrame(data)
        file_name = "Results/results.xlsx"
        save_to_excel(file_name, "Sheet2", df)


if __name__ == "__main__":
    main()