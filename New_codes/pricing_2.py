





from models_coalition_new import SubProblem, MasterProblem


def column_generation(adj, forbidden_set=[], allowed_set = [], initial = False):

    #if initial:
    not_fractional = False
    new_routes_record = [0,0,0]
    iteration = 1
    new_routes_to_add=set()
    

    master_prob = MasterProblem(adj, forbidden_set, allowed_set)
    p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add)
    
    if not y_r_result:
        return None, None, None, None, status
    
    def check_values(d):
        for key, value in d.items():
            if value != 0 and value!=1:
                return False
        return True
    
    #if check_values(y_r_result):
    #    not_fractional = True
    #    return y_r_result, not_fractional, master_prob_model, master_prob_model.ObjVal, master_prob_model
    
    dual_values_delta, dual_values_subsidy, dual_values_IR = master_prob.getDuals()
    sub_problem = SubProblem(adj, forbidden_set)

    while True:
        iteration+=1
        print(f"iteration count: {iteration}")
        new = sub_problem.dy_prog(dual_values_delta, dual_values_subsidy, dual_values_IR)
        if not new:
            break
        for array in new:
            new_routes_to_add.add(tuple(array))
        new_routes_record.append(new_routes_to_add)
        
        p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add)
        dual_values_delta, dual_values_subsidy, dual_values_IR = master_prob.getDuals()



        if check_values(y_r_result):
            print("All non-zero values are 1, breaking the loop.")
            not_fractional = True
            break

        if new_routes_record[-1]==new_routes_record[-2]==new_routes_record[-3]:
            break

        if not new_routes_to_add:
            break
    

    return y_r_result, not_fractional, master_prob_model, master_prob_model.ObjVal, master_prob_model.status
 
def main():
    y_r_result, not_fractional, master_prob_model, obj_val, status = column_generation()

if __name__ == "__main__":
    main()




 
#def main():
#    y_r_result, not_fractional, master_prob_model, obj_val, status = column_generation()
#if __name__ == "__main__":
#    main()