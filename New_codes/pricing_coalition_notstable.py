from models_coalition_new import SubProblem, MasterProblem


def column_generation(adj, forbidden_set=[], allowed_set = [], initial = False):

    #if initial:
    not_fractional = False
    iteration = 0
    new_routes_to_add=set()
    master_prob_model = None
    
    master_prob = MasterProblem(adj, forbidden_set, allowed_set)
    sub_problem = SubProblem(adj, forbidden_set)
    
    def check_values(d):
        for key, value in d.items():
            if value != 0 and value!=1:
                return False
        return True
    
    while True:
        iteration+=1
        #print(f"iteration count: {iteration}")
        p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add, None)

        if y_r_result is None:
            return None, None, None, None, status
        
        dual_values_delta, dual_values_subsidy, dual_values_IR = master_prob.getDuals()
        
        new = sub_problem.dy_prog(dual_values_delta, dual_values_subsidy, dual_values_IR)
        
        # if no new routes are found, break the loop
        if new:
            for array in new:
                new_routes_to_add.add(tuple(array))
            continue
        else:
            if iteration!=1 and check_values(y_r_result):
                print("All non-zero values are 1, no new routes found, breaking the loop.")
                not_fractional = True
            print("No new routes found, breaking the loop.")
            break  

        
    return y_r_result, not_fractional, master_prob_model, master_prob_model.ObjVal, master_prob_model.status