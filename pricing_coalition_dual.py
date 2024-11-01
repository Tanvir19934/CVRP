from models_coalition_dual import SubProblem, MasterProblem


def column_generation(adj, forbidden_set=[], allowed_set = [], initial = False):

    #if initial:
    not_fractional = False
    new_routes_record = [0,0,0]
    iteration = 1
    new_routes_to_add=set()
    

    master_prob = MasterProblem(adj, forbidden_set, allowed_set)
    y_r_result, master_prob_model, status = master_prob.relaxedLP(None)
    
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
    
    dual_values, dual_values_BB, dual_values_IR, dual_values_stability = master_prob.getDuals()
    sub_problem = SubProblem(adj, forbidden_set)

    while True:
        new = sub_problem.dy_prog(dual_values,dual_values_BB, dual_values_IR, dual_values_stability)
        if not new:
            break
        for array in new:
            new_routes_to_add.add(tuple(array))
        new_routes_record.append(new_routes_to_add)

        #for item in new_routes_to_add: p_result,e_S_result,e_BB_result,e_IR_result
        #    master_prob.r_set.add(tuple(item))
        #e_P, e_P_total = {}, {}
        #e_S, e_S_total = {}, {}
        #e_BB, e_BB_total = {}, {}
        #e_IR, e_IR_total = {}, {}
        #for route in new_routes_to_add:
        #    e_P[route], e_S[route], e_BB[route], e_IR[route] = lp(route, standalone_cost_degree_2, N)
        #    e_P_total[route] = sum(e_P[route].values())
        #    e_S_total[route] = sum(e_S[route].values())
        #    e_BB_total[route] = sum(e_BB[route].values())
        #    e_IR_total[route] = sum(e_IR[route].values())
        

        y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add)
        dual_values,dual_values_BB, dual_values_IR, dual_values_stability = master_prob.getDuals()


        iteration+=1



        if check_values(y_r_result):
            print("All non-zero values are 1, breaking the loop.")
            not_fractional = True
            break

        if new_routes_record[-1]==new_routes_record[-2]==new_routes_record[-3]:
            break

        if not new_routes_to_add:
            break
    print(f"iteration count: {iteration}")

    return y_r_result, not_fractional, master_prob_model, master_prob_model.ObjVal, master_prob_model.status
 
def main():
    y_r_result, not_fractional, master_prob_model, obj_val, status = column_generation()

if __name__ == "__main__":
    main()