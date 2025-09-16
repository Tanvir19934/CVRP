from models_coalition_new import SubProblem, MasterProblem
from utils_new import check_values, tsp_tour, prize_collecting_tsp
from config_new import always_generate_rows, use_column_heuristic, rand_seed, tol
import time
import copy
import random

random.seed(rand_seed)

def column_generation(branching_arc, forbidden_set=[], tsp_memo={}, L=None, feasibility_memo={}, global_tsp_memo={}, initial = False, parent_constraints=set()):

    not_fractional = False
    CG_iteration = 0
    RG_iteration = 0
    col_int_flag = 0
    RG_time, CG_time, RG_DP_time, CG_DP_time, LP_time = 0, 0, 0, 0, 0
    new_columns_to_add=set()
    new_constraints = copy.deepcopy(parent_constraints) if (parent_constraints and not always_generate_rows) else set()
    num_lp = 0
    master_prob = MasterProblem(forbidden_set)
    sub_problem = SubProblem(forbidden_set)

    start_4 = time.perf_counter()
    
    if always_generate_rows or initial:
        while True:
            CG_iteration+=1
            flag = 0
            print(f"CG iteration count: {CG_iteration}")
            
            ''' Now we are in the RGSP and making it RGSP feasible by iterating between RMP and RGSP until no constraints are found'''
            start_3 = time.perf_counter()

            while True:
                RG_iteration+=1
                print(f"RG iteration count: {RG_iteration}")
                start_lp = time.perf_counter()
                if CG_iteration == 1 and RG_iteration == 1:
                    p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(branching_arc, new_columns_to_add, new_constraints,True)
                    break
                else:
                    p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(branching_arc, new_columns_to_add, new_constraints, False)
                    print(master_prob_model.ObjVal)
                end_lp = time.perf_counter()
                LP_time += end_lp-start_lp
                num_lp+=1
                if not y_r_result:
                    return None, None, None, None, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, new_constraints

                start_5 = time.perf_counter()

                new_route = prize_collecting_tsp(p_result)
                end_5 = time.perf_counter()

                RG_DP_time += end_5-start_5
                if not new_route:
                    break
                else:
                    for item in new_route:
                        new_constraints.add((tuple(item[0]), item[2]))
            end_3 = time.perf_counter()
            RG_time +=  end_3-start_3

            ''' This is the CGSP, at this point our solution is RGSP feasible and CGSP feasible but not optimum, meaning there may be better routes to add '''
            dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle = master_prob.getDuals()
            if dual_values_delta==None:
                return None, None, None, None, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, new_constraints
            start_2 = time.perf_counter()
            if CG_iteration == 1:
                new_columns, feasibility_memo = sub_problem.dy_prog(dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle, feasibility_memo, True)
            else:
                new_columns, feasibility_memo = sub_problem.dy_prog(dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle, feasibility_memo, False)
            end_2 = time.perf_counter()
            
            if not new_columns:
                break

            for item in new_columns:
                if tuple(item) not in new_columns_to_add:
                    flag = 1
                    break
                            
            CG_DP_time+=end_2-start_2

            # add the new routes with negative reduced costs to the set
            for array in new_columns:
                new_columns_to_add.add(tuple(array))
            
            """CGSP ends here"""

        if use_column_heuristic:
            for array in new_columns_to_add:
                sorted_array = tuple([0]+ sorted(array[1:-1])+ [0])
                if sorted_array in global_tsp_memo:
                    new_column_constr = global_tsp_memo[sorted_array]
                else:
                    new_column_constr = tsp_tour(array)
                    global_tsp_memo[sorted_array] = new_column_constr
                new_constraints.add(new_column_constr)

    else:
        while True:
            CG_iteration+=1
            flag = 0
            print(f"CG iteration count: {CG_iteration}")
            start_lp = time.perf_counter()
            if CG_iteration == 1:
                p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(branching_arc, new_columns_to_add, new_constraints,True)
            else:
                p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(branching_arc, new_columns_to_add, new_constraints,False)
            end_lp = time.perf_counter()
            LP_time += end_lp-start_lp
            num_lp+=1
            if not y_r_result:
                return None, None, None, None, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, new_constraints

            ''' This is the CGSP, at this point our solution is RGSP feasible and CGSP feasible but not optimum, meaning there may be better routes to add '''
            dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle = master_prob.getDuals()
            if dual_values_delta==None:
                return None, None, None, None, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, new_constraints
            start_2 = time.perf_counter()
            if CG_iteration == 1:
                new_columns, feasibility_memo = sub_problem.dy_prog(dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle, feasibility_memo, True)
            else:
                new_columns, feasibility_memo = sub_problem.dy_prog(dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle, feasibility_memo, False)            
            end_2 = time.perf_counter()
            if not new_columns:
                break

            for item in new_columns:
                if tuple(item) not in new_columns_to_add:
                    flag = 1
                    break
            if flag == 0:
                break
                            
            CG_DP_time+=end_2-start_2

            # add the new routes with negative reduced costs to the set
            for array in new_columns:
                new_columns_to_add.add(tuple(array))
            
            """CGSP ends here"""
        
        if use_column_heuristic:
            for array in new_columns_to_add:
                sorted_array = tuple([0]+ sorted(array[1:-1])+ [0])
                if sorted_array in global_tsp_memo:
                    new_column_constr = global_tsp_memo[sorted_array]
                else:
                    new_column_constr = tsp_tour(array)
                    global_tsp_memo[sorted_array] = new_column_constr
                new_constraints.add(new_column_constr)
        
        start_3 = time.perf_counter()
        if check_values(y_r_result):
            col_int_flag = 1
            print("Integer solution has been hit, starting row generation")
            
            while True:
                RG_iteration+=1
                print(f"RG iteration count: {RG_iteration}")
                
                start_5 = time.perf_counter()
                new_route = prize_collecting_tsp(p_result)
                end_5 = time.perf_counter()
                RG_DP_time += end_5-start_5
                if not new_route:
                    break
                else:
                    for item in new_route:
                        new_constraints.add((tuple(item[0]), item[2]))
                if True:
                    p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(branching_arc, new_columns_to_add, new_constraints, False)
                    num_lp+=1
                    if not y_r_result:
                        return None, None, None, None, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, new_constraints

        end_3 = time.perf_counter()
        RG_time += end_3-start_3

    if check_values(y_r_result):
        print("All non-zero values are 1")
        not_fractional = True
    else:
        if col_int_flag==1:
            pass
    
    end_4 = time.perf_counter()

    CG_time = end_4-start_4

    return y_r_result, not_fractional, master_prob_model, master_prob_model.ObjVal, master_prob_model.status, \
        CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time, RG_DP_time, LP_time, tsp_memo, feasibility_memo, global_tsp_memo, num_lp, new_constraints