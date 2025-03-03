from models_coalition_new import SubProblem, MasterProblem, RowGeneratingSubProblem
from utils_new import check_values
import logging
from config_new import NODES, tol
import time
import os

def column_generation(adj, forbidden_set=[], allowed_set = [], tsp_memo={}, initial = False):

    not_fractional = False
    CG_iteration = 0
    RG_time, CG_time, RG_DP_time, CG_DP_time = 0, 0, 0, 0
    new_routes_to_add=set()
    log_dir = "New_codes/LogFiles"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
    filename = f"{log_dir}/bpc_{NODES}.log" 
    logging.basicConfig(filename=filename, level=logging.INFO, filemode='w', format='%(asctime)s - %(message)s')
    
    
    master_prob = MasterProblem(adj, forbidden_set, allowed_set)
    sub_problem = SubProblem(adj, forbidden_set)
    row_generating_subproblem = RowGeneratingSubProblem(adj, forbidden_set)

    #start_1 = time.perf_counter()
    #L = row_generating_subproblem.dy_prog()
    #end_1 = time.perf_counter()
    #logging.info(f"Time taken to solve the Row Generation DP: {end_1-start_1:0.2f}")

    new_arr =[[0]]*4

    start_4 = time.perf_counter()
    
    while True:

        CG_iteration+=1
        flag = 0
        print(f"CG iteration count: {CG_iteration}")
        
        #'''Initial LP relaxation, guaranteed to be RGSP feasible, so skipping that and going straight to the CGSP at the first iteration'''
        #if initial:
        #    p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, None)
        #    initial = False
        RG_iteration = 0
        
        ''' Now we are in the RGSP and making it RGSP feasible by iterating between RMP and RGSP until no constraints are found'''
        new_constraints = set()
        start_3 = time.perf_counter()
        while True:
            RG_iteration+=1
            print(f"RG iteration count: {RG_iteration}")
            p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, new_constraints)
            if not y_r_result:
                return None, None, None, None, status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time
            start_5 = time.perf_counter()
            new_constr = row_generating_subproblem.generate_routes(tsp_memo, p_result)
            end_5 = time.perf_counter()
            logging.info(f"Time taken to generate new constraints: {end_5-start_5:0.2f}")
            RG_DP_time += end_5-start_5
            if not new_constr:
                break
            for array in new_constr:
                new_constraints.add(tuple(array))
        end_3 = time.perf_counter()
        logging.info(f"Time taken to solve the RGSP: {end_3-start_3:0.2f}")


        ''' This is the CGSP, at this point our solution is RGSP feasible and CGSP feasible but not optimum, meaning there may be better routes to add '''
        dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle = master_prob.getDuals()
        start_2 = time.perf_counter()
        new = sub_problem.dy_prog(dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle)
        end_2 = time.perf_counter()
        logging.info(f"Time taken to solve the CG DP: {end_2-start_2:0.2f}")
        # if no new routes are found, break the loop
        new = {key: value for key, value in new.items() if value <= -tol}


        if not new:
            break

        #for item in new:
        #    if tuple(item) not in new_routes_to_add:
        #        flag = 1
        #if flag == 0:
        #    break

        #print(len(new))
        new_arr.append(new)
        #print(len(new_routes_to_add))

        #if new_arr[-1] == new_arr[-2] == new_arr[-3] == new_arr[-4]:
        #    print("No new routes found, breaking the loop.")
        #    break
        
        # add the new routes with negative reduced costs to the set
        for array in new:
            new_routes_to_add.add(tuple(array))
        #print(len(new_routes_to_add))
        2


    if check_values(y_r_result):
        print("All non-zero values are 1")
        not_fractional = True
    
    end_4 = time.perf_counter()
    logging.info(f"Time taken to solve the CGSP: {end_4-start_4:0.2f}")

    RG_time =    round(end_3-start_3,2)
    CG_time =    round(end_4-start_4,2)
    CG_DP_time = round(end_2-start_2,2)

    return y_r_result, not_fractional, master_prob_model, master_prob_model.ObjVal, master_prob_model.status, CG_iteration, RG_iteration, RG_time, CG_time, CG_DP_time