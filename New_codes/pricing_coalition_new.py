from models_coalition_new import SubProblem, MasterProblem, RowGeneratingSubProblem
from utils_new import check_values, tsp_tour
from config_new import NODES, tol, N, q, Q_EV, always_generate_rows, use_column_heuristic
import time
import os
import numpy as np
rnd = np.random
rnd.seed(42)


def column_generation(adj, forbidden_set=[], allowed_set = [], tsp_memo={}, L=None, feasibility_memo={}, initial = False, root_constraints=set()):

    not_fractional = False
    CG_iteration = 0
    RG_iteration = 0
    col_int_flag = 0
    RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time = 0, 0, 0, 0, 0
    new_routes_to_add=set()
    new_constr = set()
    if always_generate_rows:
        root_constraints = set()
    else:
        new_constraints = root_constraints
    #for item in [[0, 11, 1, 7, 3, 0], [0, 5, 10, 15, 0], [0, 20, 0], [0, 19, 16, 8, 0], [0, 18, 0], [0, 2, 0], [0, 13, 4, 0], [0, 6, 14, 0], [0, 17, 0], [0, 12, 9, 0]]:
    #    new_constraints.add((tuple(item), tsp_tour(item)[1]))


    num_lp = 0
 
    master_prob = MasterProblem(adj, forbidden_set, allowed_set)
    sub_problem = SubProblem(adj, forbidden_set)
    row_generating_subproblem = RowGeneratingSubProblem(adj, forbidden_set)

    start_4 = time.perf_counter()
    
    if always_generate_rows or initial:
        while True:
            CG_iteration+=1
            flag = 0
            print(f"CG iteration count: {CG_iteration}")
            
            ''' Now we are in the RGSP and making it RGSP feasible by iterating between RMP and RGSP until no constraints are found'''
            new_constraints = set()
            start_3 = time.perf_counter()

            while True:
                RG_iteration+=1
                print(f"RG iteration count: {RG_iteration}")
                start_lp = time.perf_counter()
                if CG_iteration == 1:
                    p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, new_constraints,True)
                else:
                    p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, new_constraints,False)
                end_lp = time.perf_counter()
                LP_time += end_lp-start_lp
                num_lp+=1
                if not y_r_result:
                    return None, None, None, None, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, num_lp, L, new_constraints
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
            RG_time +=  end_3-start_3


            ''' This is the CGSP, at this point our solution is RGSP feasible and CGSP feasible but not optimum, meaning there may be better routes to add '''
            dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle = master_prob.getDuals()
            start_2 = time.perf_counter()
            new, feasibility_memo = sub_problem.dy_prog(dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle, feasibility_memo)
            end_2 = time.perf_counter()
            new = {key: value for key, value in new.items() if value <= -tol}
            CG_DP_time+=end_2-start_2


            if not new:
                break

            for item in new:
                if tuple(item) not in new_routes_to_add:
                    flag = 1
                    break
            if flag == 0:
                break

            # add the new routes with negative reduced costs to the set
            for array in new:
                new_routes_to_add.add(tuple(array))
                if use_column_heuristic and tuple(array) not in new_constraints:
                    if array in tsp_memo:
                        new_column_constr = tsp_memo[array]
                    else:
                        new_column_constr = tsp_tour(array)
                        tsp_memo[array] = new_column_constr
                    new_constraints.add(new_column_constr)
        2
            

    else:
        while True:
            CG_iteration+=1
            flag = 0
            print(f"CG iteration count: {CG_iteration}")
            start_lp = time.perf_counter()
            if CG_iteration == 1:
                p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, new_constraints,True)
            else:
                p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, new_constraints,False)
            end_lp = time.perf_counter()
            LP_time += end_lp-start_lp
            num_lp+=1
            if not y_r_result:
                return None, None, None, None, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, num_lp, L, new_constraints

                #start_5 = time.perf_counter()
                #if L is None:
                #    L = row_generating_subproblem.dy_prog(N,q,Q_EV)
                #new_constr, tsp_memo = row_generating_subproblem.generate_constr(tsp_memo,p_result,L)
                #end_5 = time.perf_counter()
                #RG_DP_time += end_5-start_5
                #if not new_constr:
                #    break
                #for array in new_constr:
                #    new_constraints.add(tuple(array))
            #end_3 = time.perf_counter()

            ''' This is the CGSP, at this point our solution is RGSP feasible and CGSP feasible but not optimum, meaning there may be better routes to add '''
            dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle = master_prob.getDuals()
            start_2 = time.perf_counter()
            new, feasibility_memo = sub_problem.dy_prog(dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle, feasibility_memo)
            end_2 = time.perf_counter()
            new = {key: value for key, value in new.items() if value <= -tol}
            CG_DP_time+=end_2-start_2


            if not new:
                break


            for item in new:
                if tuple(item) not in new_routes_to_add:
                    flag = 1
                    break
            if flag == 0:
                break

            # add the new routes with negative reduced costs to the set
            for array in new:
                new_routes_to_add.add(tuple(array))
                if use_column_heuristic and tuple(array) not in new_constraints:
                    if array in tsp_memo:
                        new_column_constr = tsp_memo[array]
                    else:
                        new_column_constr = tsp_tour(array)
                        tsp_memo[array] = new_column_constr
                    new_constraints.add(new_column_constr)
            2
        
        start_3 = time.perf_counter()
        if check_values(y_r_result):
            col_int_flag = 1
            print("Integer solution has been hit, starting row generation")
            RG_iteration += 1
            #if not use_column_heuristic:
            #    new_constraints = set()
            
            #while True:
            print(f"RG iteration count: {RG_iteration}")
            
            start_5 = time.perf_counter()
            if L is None:
                L = row_generating_subproblem.dy_prog(N,q,Q_EV)
            new_constr, tsp_memo = row_generating_subproblem.generate_constr(tsp_memo,p_result,L)
            end_5 = time.perf_counter()
            RG_DP_time += end_5-start_5
            #if not new_constr:
            #    break
            for array in new_constr:
                new_constraints.add(tuple(array))
            if new_constr:
                p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, new_constraints, False)
                num_lp+=1
                if not y_r_result:
                    return None, None, None, None, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, num_lp, L, new_constraints

        end_3 = time.perf_counter()
        RG_time += end_3-start_3




    if check_values(y_r_result):
        print("All non-zero values are 1")
        not_fractional = True
    else:
        if col_int_flag==1:
            2
    
    end_4 = time.perf_counter()

    CG_time =    end_4-start_4
    #new_constraints=root_constraints

    return y_r_result, not_fractional, master_prob_model, master_prob_model.ObjVal, master_prob_model.status, \
        CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, num_lp, L, new_constraints