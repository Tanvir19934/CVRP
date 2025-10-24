from pricing_coalition_new import column_generation
from config_new import (
    V, Q_EV, q, NODES,k, plot_enabled, use_column_heuristic,
    always_generate_rows, N, rand_seed
    )
import time
import random
from utils_new import (
    print_solution, save_to_excel, print_metadata, unpack_result,

    
    generate_tsp_cache, code_status, validate_solution
    )
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import GRB

random.seed(rand_seed)  
class Node:  
    
    def __init__(self, depth, name, forbidden, parent, constraints=set()):
        self.depth = depth
        self.solution = None
        self.obj_val = None
        self.name = name
        self.parent = parent
        self.forbidden = forbidden
        self.not_fractional = False
        self.constraints = constraints
        print("\ndepth =",self.depth) 
        print("\n")

    def __lt__(self, other):
        return self.obj_val < other.obj_val  #For heap implementation. The heapq.heapify() will heapify the list based on this criteria.

def branching() -> None:

    q[0] = 0
    num_nodes_explored=1
    tol = 1e-4 
    forbidden_set = set()
    for node in V:
        for n in V:
            if n != node:
                if q[node] + q[n] > Q_EV:
                    forbidden_set.add((node, n))  # Add violating arcs to forbidden_set
                    forbidden_set.add((n,node))  # Add violating arcs to forbidden_set

    start_2 = time.perf_counter()
    tsp_memo, global_tsp_memo = generate_tsp_cache(NODES, k)
    end_2 = time.perf_counter()
    tsp_cache_time = end_2-start_2
    
    global Total_CG_iteration, Total_RG_iteration, Total_RG_time, Total_CG_time, iterations, lp_gaps
    global Total_RG_DP_time, Total_CG_DP_time, Total_execution_time, Total_LP_time
    Total_CG_iteration,  Total_RG_iteration,  Total_RG_time,  Total_CG_time,  Total_RG_DP_time,  Total_CG_DP_time, Total_LP_time = 0, 0, 0, 0, 0, 0, 0
    Total_num_lp = 0
    feasibility_memo={}
    iterations = []
    lp_gaps = []
    
    
    # Create the root node by solving the initial rmp
    start_3 = time.perf_counter()
    [
        root_y_r_result, root_not_fractional, root_master_prob_model, root_obj_val, status, CG_iteration, RG_iteration, RG_time, CG_time, 
        CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, root_constraints, columns
    ] = unpack_result(
            column_generation(
            None, forbidden_set={}, tsp_memo=tsp_memo, L=None, feasibility_memo=feasibility_memo, global_tsp_memo=global_tsp_memo, 
            initial=True, parent_constraints=set()
            )
        )
    end_3 = time.perf_counter()
    root_node_time = end_3 - start_3
    print(f"Time to solve root node: {root_node_time}")
    print(f"root_obj_val: {root_obj_val}\n\n")

    [
        result, not_fractional, model, obj_val, status, CG_iteration, RG_iteration, RG_time, CG_time, 
        CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, constraints, columns
    ] = unpack_result(
            column_generation(
            None, forbidden_set={}, tsp_memo=tsp_memo, L=None, feasibility_memo=feasibility_memo, global_tsp_memo=global_tsp_memo, 
            initial=False, parent_constraints=set(), new_columns_to_add=columns
            )
        )
    track_time_iterations(CG_iteration, RG_iteration, RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time)
    Total_num_lp += num_lp



    if status == GRB.OPTIMAL:
        print("Optimal solution found:")
        obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes = print_solution(model)
    else:
        print("No optimal solution found.")
    
    print_metadata(Total_CG_iteration, Total_RG_iteration, num_nodes_explored,
              Total_RG_time, Total_CG_time, Total_RG_DP_time, Total_CG_DP_time,
              Total_LP_time, tsp_cache_time, obj, root_obj_val, Total_num_lp, root_node_time)
    
    return (
        obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes, 
        root_obj_val, num_nodes_explored, tsp_cache_time, Total_num_lp, tsp_memo, root_node_time
    )

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
        [
            obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes, 
            root_obj_val, num_nodes_explored, tsp_cache_time, Total_num_lp, tsp_memo, root_node_time
        ] = branching()
        end = time.perf_counter()
        
        print(f"Execution time for nodes={NODES}: {end - start}")
        code = code_status(use_column_heuristic, always_generate_rows)
        validate_solution(payments, tsp_memo, N, solution_routes)
        
        data = {
            "Nodes": [NODES],
            "Obj": [obj],
            "Total Miles": [total_miles],
            "EV miles": [EV_miles],
            "DV Miles": [total_miles-EV_miles],
            "Total payments": [Total_payments],
            "Subsidy": [Subsidy],
            "LP Gap": [((obj-root_obj_val)/obj)*100],
            "Number of nodes explored": [num_nodes_explored],
            "Total CG iterations": [Total_CG_iteration],
            "Total RG iterations": [Total_RG_iteration],
            "Total CG DP time": [Total_CG_DP_time],
            "Total RG DP time": [Total_RG_DP_time],
            #"TSP cache time": [tsp_cache_time],
            "Total LP relaxation time": [Total_LP_time],
            "Root node time": [root_node_time],
            "Root gap": [((obj-root_obj_val)/obj)*100],
            "Total execution time": [end-start],
            "Total number of LPs solved": [Total_num_lp],
            "code": [code]
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
    if plot_enabled:
        plt.show()
        input("Press Enter to exit...")