from models import SubProblem, MasterProblem


def column_generation(adj,forbidden_set=[], initial = False):

    #if initial:
    not_fractional = False
    new_routes_record = [0,0,0,0,0]
    iteration = 1
    new_routes_to_add=set()

    master_prob = MasterProblem(adj, forbidden_set)
    y_r_result, master_prob_model, status = master_prob.relaxedLP(None)
    if not y_r_result:
        return None, None, None, None, status
    dual_values = master_prob.getDuals()
    print(dual_values)
    print(sum(dual_values.values()))
    sub_problem = SubProblem(adj, forbidden_set)

    while True:
        new = sub_problem.dy_prog(dual_values)
        if not new:
            break
        for array in new:
            new_routes_to_add.add(tuple(array))
        new_routes_record.append(new_routes_to_add)

        #for item in new_routes_to_add: 
        #    master_prob.r_set.add(tuple(item))

        y_r_result, master_prob_model, status = master_prob.relaxedLP(new_routes_to_add)
        dual_values = master_prob.getDuals()
        print(dual_values)
        print(sum(dual_values.values()))

        iteration+=1

        def check_values(d):
            for key, value in d.items():
                if value != 0 and value!=1:
                    return False
            return True

        if check_values(y_r_result):
            print("All non-zero values are 1, breaking the loop.")
            not_fractional = True
            break

        if new_routes_record[-1]==new_routes_record[-2]==new_routes_record[-3]==new_routes_record[-4]==new_routes_record[-5]==new_routes_record[-6]:
            break

        if not new_routes_to_add:
            break
    print(f"iteration count: {iteration}")

    return y_r_result, not_fractional, master_prob_model, master_prob_model.ObjVal, master_prob_model.status
 
def main():
    y_r_result, not_fractional, master_prob_model, obj_val, status = column_generation()

if __name__ == "__main__":
    main()