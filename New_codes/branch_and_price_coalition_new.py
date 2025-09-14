from pricing_coalition_new import column_generation
from config_new import (
    V, Q_EV, q, NODES,k, plot_enabled, use_column_heuristic,
    always_generate_rows, N, rand_seed, best_obj, SEARCH_MODE
    )
from collections import defaultdict
import copy
import time
import random
from utils_new import (
    print_solution, save_to_excel, print_metadata,
    generate_tsp_cache, update_plot, make_stack, code_status, validate_solution
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
    root_y_r_result, root_not_fractional, root_master_prob_model, root_obj_val, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, root_constraints = column_generation(None, forbidden_set={}, tsp_memo=tsp_memo, L = None, feasibility_memo=feasibility_memo, global_tsp_memo=global_tsp_memo, initial = True, parent_constraints = set())
    print(f"root_obj_val: {root_obj_val}\n\n")

    track_time_iterations(CG_iteration, RG_iteration, RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time)
    Total_num_lp+=num_lp
    
    if root_not_fractional:
        print("Optimal solution found at the root node: did not need branching")
        obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes = print_solution(root_master_prob_model)
        print_metadata(Total_CG_iteration, Total_RG_iteration, num_nodes_explored,
              Total_RG_time, Total_CG_time, Total_RG_DP_time, Total_CG_DP_time,
              Total_LP_time, tsp_cache_time, obj, root_obj_val, Total_num_lp)

        return obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes, root_obj_val, num_nodes_explored, tsp_cache_time, Total_num_lp, tsp_memo
    
    root_node = Node(0, "root_node", set(), None, set())
    left_not_fractional = right_not_fractional = False
    root_node.solution = root_y_r_result
    root_node.not_fractional = root_not_fractional
    root_node.obj_val = root_obj_val
    root_node.model = root_master_prob_model
    root_node.constraints = root_constraints

    stack, push, pop = make_stack(SEARCH_MODE)

    def _process_child(status, obj_val, not_fractional, model, solution, node):
        """
        Local helper: handle a child node after column generation.
        Uses enclosing scope: best_objective, best_node, int_count, tol, push.
        """
        nonlocal best_objective, best_node, int_count

        if status != GRB.OPTIMAL:
            return  # infeasible or not solved to optimal → prune silently

        # Always store what we learned about the child
        node.obj_val  = obj_val
        node.model    = model
        node.solution = solution

        if not_fractional:  # integral ⇒ fathom (update UB, don't push)
            if obj_val < best_objective:
                best_objective = obj_val
                best_node = node
                int_count += 1
                print(f"\033[1m[INCUMBENT]\033[0m UB ← {best_objective:.6g}")
            return

        # Fractional: prune by bound or keep
        if obj_val < best_objective:
            push(node)  # promising → explore later
        else:
            print(f"\033[1m[PRUNE LB]\033[0m LB {obj_val:.6g} ≥ UB {best_objective:.6g}")

    best_node = None
    best_objective = best_obj
    push(root_node)
    int_count = 0
    outer_iter = 0
    num_nodes_explored = 0
    
    while len(stack) > 0:
        outer_iter += 1
        node = pop()
        print(f"outer iteration count: {outer_iter}")
        print(f"Integer solution found so far = {int_count}. Size of stack = {len(stack)}")
        print(f"\nCurrent best = {best_objective}")
        print(f"\nCurrent best's LP gap = {((best_objective - root_obj_val) / best_objective) * 100}%")

        if plot_enabled and outer_iter % 10 == 0:
            # Update the plot every 10 iterations
            update_plot(outer_iter, ((best_objective - root_obj_val) / best_objective) * 100, iterations, lp_gaps)
            plt.show()


        if node.obj_val  > best_objective:
            print("\n\033[1mPruning the node as its obj is worse than the best integer solution found so far\033[0m\n")
            continue
        else:
            # 1) Collect all y_r_ variables and their values
            model_vars = {v.VarName: v.X for v in node.model.getVars()}
            fractional_ys = {
                name: x for name, x in model_vars.items()
                if name.startswith('y_r_') and 0.00000001 < x < 1 - 0.00000001
            }

            # 2) Sum y-values over every arc in each fractional route
            flow_vars = defaultdict(float)
            for var_name, y_val in fractional_ys.items():
                # extract the route tuple from the var name, e.g. y_r_[(1,2,3)]
                route = eval(var_name[var_name.find('[') + 1 : var_name.rfind(']')])
                for u, v in zip(route, route[1:]):
                    flow_vars[(u, v)] += y_val

            # 3) Now pick only those arcs whose total flow is strictly fractional
            branching_arcs = {
                arc: flow for arc, flow in flow_vars.items()
                if tol < flow < 1 - tol
            }

            # branching_arcs now contains only (i,j) with 0 < x_{i,j} < 1
            # you can then choose the one closest to 0.5, e.g.:
            branching_arc, best_val = min(
                branching_arcs.items(),
                key=lambda iv: abs(iv[1] - 0.5),
                default=(None, None)
            )
            
            if not branching_arcs:
                continue 

            # Create left branch node
            left_node = Node(node.depth + 1, f'{branching_arc}={0}', copy.deepcopy(node.forbidden), node, copy.deepcopy(node.constraints))
            left_node.forbidden.add(branching_arc)
            print(f"branching {branching_arc}={0}")

            [
                left_result, left_not_fractional, left_model, left_obj_val, left_status, CG_iteration, RG_iteration, RG_time, CG_time, 
                CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, left_constraints 
            ] = column_generation(
                (branching_arc,0), left_node.forbidden, tsp_memo, feasibility_memo=feasibility_memo, global_tsp_memo=global_tsp_memo, 
                initial = False, parent_constraints=copy.deepcopy(node.constraints)
                )
            track_time_iterations(CG_iteration, RG_iteration, RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time)
            num_nodes_explored+=1
            Total_num_lp+=num_lp
            
            if not always_generate_rows:
                left_node.constraints = left_constraints

            _process_child(left_status, left_obj_val, left_not_fractional, left_model, left_result, left_node)

            # Create right branch node
            right_node = Node(node.depth + 1, f'{branching_arc}={1}', copy.deepcopy(node.forbidden), node, copy.deepcopy(node.constraints))
            for item in V:
                if branching_arc[0]!=0 and branching_arc[1]!=0:             # for (x,y) type of arcs, forbid all (x,y) arcs for all x,y not 0
                    if item!=branching_arc[1]:
                        right_node.forbidden.add((branching_arc[0],item))
                    if  item!=branching_arc[0]:
                        right_node.forbidden.add((item,branching_arc[1])) 
                elif branching_arc[0]==0:                                   # for (0,x) type of arcs, forbid all (y,x) arcs for all y not 0
                    if item!=0:
                        right_node.forbidden.add((item,branching_arc[1])) 
                elif branching_arc[1]==0:                                   # for (x,0) type of arcs, forbid all (x,y) arcs for all y not 0
                    if item!=0:
                        right_node.forbidden.add((branching_arc[0],item))


            print(f"branching {branching_arc}={1}")
            [
                right_result, right_not_fractional, right_model, right_obj_val, right_status, CG_iteration, RG_iteration, RG_time, CG_time, 
                CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, right_constraints
            ] = column_generation(
                (branching_arc,1), right_node.forbidden,tsp_memo, feasibility_memo=feasibility_memo, global_tsp_memo=global_tsp_memo, 
                initial = False, parent_constraints=copy.deepcopy(node.constraints)
                )
            track_time_iterations(CG_iteration, RG_iteration, RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time)
            num_nodes_explored+=1
            Total_num_lp+=num_lp

            if not always_generate_rows:
                right_node.constraints = right_constraints
            
            _process_child(right_status, right_obj_val, right_not_fractional, right_model, right_result, right_node)

    if best_node:
        print("Optimal solution found:")
        obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes = print_solution(best_node.model)
    else:
        print("No optimal solution found.")
    
    print_metadata(Total_CG_iteration, Total_RG_iteration, num_nodes_explored,
              Total_RG_time, Total_CG_time, Total_RG_DP_time, Total_CG_DP_time,
              Total_LP_time, tsp_cache_time, obj, root_obj_val, Total_num_lp)
    
    return obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes, root_obj_val, num_nodes_explored, tsp_cache_time, Total_num_lp, tsp_memo

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
        obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes, root_obj_val, num_nodes_explored, tsp_cache_time, Total_num_lp, tsp_memo = branching()
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