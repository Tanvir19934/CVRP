from models_coalition_new import SubProblem, MasterProblem, RowGeneratingSubProblem
from utils_new import check_values
from config_new import NODES, tol, N, q, Q_EV
import time
import os
import numpy as np
rnd = np.random
rnd.seed(42)
def column_generation(adj, forbidden_set=[], allowed_set = [], tsp_memo={}, L=None, initial = False):

    not_fractional = False
    CG_iteration = 0
    RG_time, CG_time, RG_DP_time, CG_DP_time = 0, 0, 0, 0
    new_routes_to_add=set()
    new_constr = set()
    num_lp = 0
 
    master_prob = MasterProblem(adj, forbidden_set, allowed_set)
    sub_problem = SubProblem(adj, forbidden_set)
    row_generating_subproblem = RowGeneratingSubProblem(adj, forbidden_set)

    start_4 = time.perf_counter()
    
    while True:

        CG_iteration+=1
        flag = 0
        print(f"CG iteration count: {CG_iteration}")

        RG_iteration = 0
        
        ''' Now we are in the RGSP and making it RGSP feasible by iterating between RMP and RGSP until no constraints are found'''
        new_constraints = set()
        start_3 = time.perf_counter()

        while True:
            RG_iteration+=1
            print(f"RG iteration count: {RG_iteration}")
            if CG_iteration == 1:
                p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, new_constraints,True)
                num_lp+=1
            else:
                p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, new_constraints,False)
                num_lp+=1
            if not y_r_result:
                return None, None, None, None, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, tsp_memo, num_lp
            if RG_iteration == 1 and CG_iteration == 1:
                break
            start_5 = time.perf_counter()
            if L is None:
                L = row_generating_subproblem.dy_prog(N,q,Q_EV)
            new_constr, tsp_memo = row_generating_subproblem.generate_constr(tsp_memo,p_result,L)
            end_5 = time.perf_counter()
            RG_DP_time += end_5-start_5
            if not new_constr:
                break
            for array in new_constr:
                new_constraints.add(tuple(array))
        end_3 = time.perf_counter()


        ''' This is the CGSP, at this point our solution is RGSP feasible and CGSP feasible but not optimum, meaning there may be better routes to add '''
        dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle = master_prob.getDuals()
        start_2 = time.perf_counter()
        new = sub_problem.dy_prog(dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle)
        end_2 = time.perf_counter()
        new = {key: value for key, value in new.items() if value <= -tol}


        if not new:
            break

        for item in new:
            if tuple(item) not in new_routes_to_add:
                flag = 1
        if flag == 0:
            break

        # add the new routes with negative reduced costs to the set
        for array in new:
            new_routes_to_add.add(tuple(array))
        2

    if check_values(y_r_result):
        print("All non-zero values are 1")
        not_fractional = True
    
    end_4 = time.perf_counter()

    RG_time =    end_3-start_3
    CG_time =    end_4-start_4
    CG_DP_time = end_2-start_2

    return y_r_result, not_fractional, master_prob_model, master_prob_model.ObjVal, master_prob_model.status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, tsp_memo, num_lp