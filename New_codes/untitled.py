from models_coalition_new import SubProblem, MasterProblem, RowGeneratingSubProblem
from utils_new import check_values

def column_generation(adj, forbidden_set=[], allowed_set = [], initial = False):

    not_fractional = False
    iteration = 0
    new_routes_to_add=set()
    
    
    master_prob = MasterProblem(adj, forbidden_set, allowed_set)
    sub_problem = SubProblem(adj, forbidden_set)

    
    #new_arr =[[0]]*4
    
    while True:

        iteration+=1
        print(f"pricing outer iteration count: {iteration}")
        new_constraints = set()
        row_generating_subproblem = RowGeneratingSubProblem(adj, forbidden_set)
        L = row_generating_subproblem.dy_prog()
        
        '''Initial LP relaxation, guaranteed to be RGSP feasible, so skipping that and going straight to the CGSP at the first iteration'''
        if iteration==1:
            p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, None)
        inner_iteration = 0
        
        ''' Now we are in the RGSP and making it RGSP feasible by iterating between RMP and RGSP until no constraints are found'''
        while iteration!=1:
            inner_iteration+=1
            print(f"inner iteration count: {inner_iteration}")
            p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, new_constraints)
            if not y_r_result:
                return None, None, None, None, status
            new_constr = row_generating_subproblem.generate_routes(L, p_result)
            if not new_constr:
                break
            for array in new_constr:
                new_constraints.add(tuple(array))


        ''' This is the CGSP, at this point our solution is RGSP feasible and CGSP feasible but not optimum, meaning there may be better routes to add '''
        dual_values_delta, dual_values_subsidy, dual_values_IR = master_prob.getDuals()
        new = sub_problem.dy_prog(dual_values_delta, dual_values_subsidy, dual_values_IR)
        # if no new routes are found, break the loop
        if not new:
            break
        #print(len(new))
        #new_arr.append(new)
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

    return y_r_result, not_fractional, master_prob_model, master_prob_model.ObjVal, master_prob_model.status