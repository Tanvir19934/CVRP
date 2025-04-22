import heapq
from pricing_coalition_new import column_generation
from config_new import V, Q_EV, q, NODES,k, plot_enabled, use_column_heuristic, always_generate_rows, N, rand_seed, best_obj
from collections import defaultdict
import copy
import time
import random
from utils_new import print_solution, save_to_excel, tsp_tour, gv_tsp_cost, ev_travel_cost
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from gurobipy import GRB

random.seed(rand_seed)  # Set a seed v
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
        print("\ndepth =",self.depth)  #sanity check
        print("\n")

    def __lt__(self, other):
        return self.obj_val < other.obj_val  #For heap implementation. The heapq.heapify() will heapify the list based on this criteria.

def generate_tsp_cache(N, k):
    print(f"\nCreating initial TSP memo... \ntotal {N}C{k} combinations\n")
    tsp_memo = {}
    global_tsp_memo = {}
    customers = list(range(1, N + 1))  #Assuming customers are labeled 1 to N
    all_combinations = []
      
    for i in range(1, k + 1):
        for comb in combinations(customers, i):
            all_combinations.append((0,) + comb + (0,))
    for item in all_combinations:
        if sum(q[i] for i in item) <= Q_EV:
            candidate_route, tsp_cost = tsp_tour(item)
            tsp_memo[item] = (tuple(candidate_route),tsp_cost)
            global_tsp_memo[item] = (tuple(candidate_route),tsp_cost)
        else:
            candidate_route, tsp_cost = tsp_tour(item)
            global_tsp_memo[item] = (tuple(candidate_route),tsp_cost)

    print(f"\nFinished creating initial TSP memo\n")
    time.sleep(1)
    return tsp_memo, global_tsp_memo

def update_plot(outer_iter, lp_gap):
    global iterations, lp_gaps
    """Updates the LP gap plot dynamically."""
    if not plot_enabled:
        return  # Do nothing if plotting is disabled

    # Append new data
    iterations.append(outer_iter)
    lp_gaps.append(lp_gap)

    # Static figure and axis setup (only first time)
    if not hasattr(update_plot, "fig"):
        plt.ion()  # Enable interactive mode
        update_plot.fig, update_plot.ax = plt.subplots()
        update_plot.line, = update_plot.ax.plot([], [], marker='o', linestyle='-')

    # Update the plot
    update_plot.line.set_xdata(iterations)
    update_plot.line.set_ydata(lp_gaps)
    update_plot.ax.relim()  # Recompute limits
    update_plot.ax.autoscale_view()  # Adjust view
    update_plot.ax.set_xlabel("Outer Iteration")
    update_plot.ax.set_ylabel("LP Gap (%)")
    update_plot.ax.set_title("LP Gap vs. Outer Iteration")
    plt.draw()
    plt.pause(0.01)  # Pause for smooth updating



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
    root_y_r_result, root_not_fractional, root_master_prob_model, root_obj_val, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, L, root_constraints = column_generation(None, forbidden_set={}, tsp_memo=tsp_memo, L = None, feasibility_memo=feasibility_memo, global_tsp_memo=global_tsp_memo, initial = True, parent_constraints = set())
    print(f"root_obj_val: {root_obj_val}\n\n")


    track_time_iterations(CG_iteration, RG_iteration, RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time)
    Total_num_lp+=num_lp
    if root_not_fractional:
        print("Optimal solution found at the root node: did not need branching")
        obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes = print_solution(root_master_prob_model)
        print(f"Total CG iterations: {Total_CG_iteration}")
        print(f"Total RG iterations: {Total_RG_iteration}")
        print(f"Total nodes explored: {num_nodes_explored}")
        print(f"Total RG time: {Total_RG_time}")
        print(f"Total CG time: {Total_CG_time}")
        print(f"Total RG DP time: {Total_RG_DP_time}")
        print(f"Total CG DP time: {Total_CG_DP_time}")
        print(f"Total LP time: {Total_LP_time:.2f}")
        print(f"tsp cache time: {tsp_cache_time}")
        print("LP gap(%): ", ((obj-root_obj_val)/obj)*100)
        print(f"root_obj_val: {root_obj_val}")
        print(f"Total number of LPs solved: {Total_num_lp}")
        return obj, total_miles, EV_miles, Total_payments, Subsidy, payments, solution_routes, root_obj_val, num_nodes_explored, tsp_cache_time, Total_num_lp, tsp_memo
    root_node = Node(0, "root_node", set(), None, set())
    left_not_fractional = right_not_fractional = False
    root_node.solution = root_y_r_result
    root_node.not_fractional = root_not_fractional
    root_node.obj_val = root_obj_val
    root_node.model = root_master_prob_model
    root_node.constraints = root_constraints
    
    #initialize best node and the stack
    best_node = None


    upper_bound = best_obj
    best_objective = float('inf')

    
    stack = [root_node]
    heapq.heapify(stack)  #we are going to traverse in best first manner
    frac_count = 0

    outer_iter = 0
    num_nodes_explored = 0 
    

    while stack:
        outer_iter += 1
        if outer_iter==9:
            pass
        print(f"outer iteration count: {outer_iter}")
        print(f"Integer solution found so far = {frac_count}. Size of stack = {len(stack)}")
        print(f"\nCurrent best = {best_objective}")
        print(f"\nCurrent best's LP gap = {((best_objective - root_obj_val) / best_objective) * 100}%")

        if plot_enabled and outer_iter % 10 == 0:
            # Update the plot every 10 iterations
            update_plot(outer_iter, ((best_objective - root_obj_val) / best_objective) * 100)

        # 80% chance take the best, 20% chance explore
        if random.random() < 1:
            # Regular best node (min obj)
            node = heapq.heappop(stack)
        else:
            # Random exploration
            idx = random.randint(1, len(stack) - 1)  # Avoid index 0 (already best)
            node = stack[idx]
            stack[idx] = stack[-1]
            stack.pop()
            heapq.heapify(stack)


        if node.not_fractional==True:
            frac_count+=1
            if node.obj_val < best_objective:
                best_objective = node.obj_val
                best_node = node
        else:  
            if node.obj_val  > best_objective and best_node is not None:
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

                if branching_arc==(0,5):
                    pass


                #prev_arcs = []
                #node_copy = node
                #while node_copy.parent is not None:
                #    prev_arcs.append(eval(node_copy.name.split('=')[0]))
                #    node_copy = node_copy.parent 
            
                #while True:
                #    branching_arc = random.choice(list(branching_arcs.keys()))
                #    #if (branching_arc[0]==0) or (branching_arc[1]==0) or branching_arc in prev_arcs:
                #    if branching_arc in prev_arcs:
                #        continue
                #    else:
                #        break


                #flow_vars = defaultdict(float)
                #model_vars ={var.VarName: var.X for var in node.model.getVars()}
                #var_names = list(model_vars.keys())
                #fractional_var = {}
                #for var in var_names:
                #    if not (abs(model_vars[var] - 0) <= tol or abs(model_vars[var] - 1) <= tol) and var.split('[')[0]=='y_r_':
                #        fractional_var[var] = model_vars[var]
                #for item in fractional_var:
                #    element = eval(item.split('[')[-1][0:-1])
                #    for i in range(len(element)-1):
                #        flow_vars[(element[i],element[i+1])]= model_vars[item]           
                
                
                if not branching_arcs:
                    continue 

                #while True:
                #    branching_arc = random.choice(list(branching_arcs.keys()))
                #    if (branching_arc[0]==0 ) or (branching_arc[1]==0):
                #        continue
                #    else:
                #        break
                #branching_arc = random.choice(list(branching_arcs.keys()))

                # Create left branch node
                left_node = Node(node.depth + 1, f'{branching_arc}={0}', copy.deepcopy(node.forbidden), node, copy.deepcopy(node.constraints))
                left_node.forbidden.add(branching_arc)
                print(f"branching {branching_arc}={0}")

                left_result, left_not_fractional, left_model, left_obj_val, left_status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, L, left_constraints = column_generation((branching_arc,0), left_node.forbidden, tsp_memo, L, feasibility_memo=feasibility_memo, global_tsp_memo=global_tsp_memo, initial = False, parent_constraints=copy.deepcopy(node.constraints))
                track_time_iterations(CG_iteration, RG_iteration, RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time)
                num_nodes_explored+=1
                Total_num_lp+=num_lp
                if not always_generate_rows:
                    left_node.constraints = left_constraints
                if left_result is not None:
                    tpl = []
                    for item in left_result:
                        if left_result[item] != 0:
                            h=list(eval(item[5:-1]))
                            tpl.extend([(h[i],h[i+1])for i in range(len(h)-1)])
                    if branching_arc in tpl:
                        pass

                if left_status==GRB.OPTIMAL and left_obj_val < upper_bound:
                    if left_not_fractional:
                        left_node.not_fractional = True
                    left_node.obj_val = left_obj_val
                    left_node.model = left_model
                    left_node.solution = left_result
                    heapq.heappush(stack, left_node)
                # Create right branch node
                right_node = Node(node.depth + 1, f'{branching_arc}={1}', copy.deepcopy(node.forbidden), node, copy.deepcopy(node.constraints))
                for item in V:
                    if branching_arc[0]!=0 and branching_arc[1]!=0:
                        if item!=branching_arc[1]:
                            right_node.forbidden.add((branching_arc[0],item))
                        if  item!=branching_arc[0]:
                            right_node.forbidden.add((item,branching_arc[1])) 
                    elif branching_arc[0]==0:
                        if item!=0:
                            right_node.forbidden.add((item,branching_arc[1])) 
                    elif branching_arc[1]==0:
                        if item!=0:
                            right_node.forbidden.add((branching_arc[0],item))


                print(f"branching {branching_arc}={1}")
                right_result, right_not_fractional, right_model, right_obj_val, right_status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, L, right_constraints= column_generation((branching_arc,1), right_node.forbidden,tsp_memo,L,feasibility_memo=feasibility_memo, global_tsp_memo=global_tsp_memo, initial = False, parent_constraints=copy.deepcopy(node.constraints))
                track_time_iterations(CG_iteration, RG_iteration, RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time)
                num_nodes_explored+=1
                Total_num_lp+=num_lp

                if not always_generate_rows:
                    right_node.constraints = right_constraints
                
                tpl = []
                if right_result is not None:
                    for item in right_result:
                        if right_result[item] != 0:
                            h=list(eval(item[5:-1]))
                            tpl.extend([(h[i],h[i+1])for i in range(len(h)-1)])
                    if branching_arc not in tpl:
                        pass


                if right_status==GRB.OPTIMAL and right_obj_val < upper_bound:
                    if right_not_fractional:
                        right_node.not_fractional = True  
                    right_node.obj_val = right_obj_val
                    right_node.model = right_model
                    right_node.solution = right_result
                    heapq.heappush(stack, right_node)


    if plot_enabled:
        plt.ioff()
        plt.show()

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
    print(f"LP gap(%): {((obj-root_obj_val)/obj)*100}")
    print(f"root_obj_val: {root_obj_val}")
    print(f"Total number of LPs solved: {Total_num_lp}")
    
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
        Execution_time = end - start
        if use_column_heuristic==False and always_generate_rows==True:
            code=1
        elif use_column_heuristic==False and always_generate_rows==False:
            code=2
        elif use_column_heuristic==True and always_generate_rows==False:
            code=3
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
            "TSP cache time": [tsp_cache_time],
            "Total LP relaxation time": [Total_LP_time],
            "Total execution time": [Execution_time],
            "Total number of LPs solved": [Total_num_lp],
            "code": [code]
        }


        payments[0]=0
        flag = 0
        for item in tsp_memo:
            pay = sum(payments[i] for i in item)
            if pay > tsp_memo[item][1]+0.1:
                flag = 1
                print(item)
                print(sum([payments[i] for i in item]))
                print(tsp_memo[item][1])
        if flag==0:
            print("No payment violations")
        flag = 0
        for i in N:
            if payments[i] > 0.01+gv_tsp_cost((0,i,0)):
                flag = 1
        if flag==0:
            print("No IR violations")

        for i in solution_routes:
            ev_travel_cost(i)
        
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