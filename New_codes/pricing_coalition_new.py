from models_coalition_new import SubProblem, MasterProblem
from utils_new import check_values, tsp_tour, prize_collecting_tsp, CGResult
from config_new import always_generate_rows, use_column_heuristic, rand_seed, run_dp
import time
import copy
import random

random.seed(rand_seed)

def run_CGSP(master_prob, sub_problem, new_columns_to_add, feasibility_memo,
             new_constraints, stats, status, forbidden_set):
    """Run Column Generation Subproblem once (dual extraction + dy_prog)."""
    dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle = master_prob.getDuals()
    if dual_values_delta is None:
        return None, feasibility_memo, stats["CG_DP_time"], status, new_columns_to_add, new_constraints

    start_2 = time.perf_counter()
    if run_dp:
        new_columns, feasibility_memo = sub_problem.dy_prog(
            dual_values_delta, dual_values_subsidy, dual_values_IR,
            dual_values_vehicle, feasibility_memo, stats["CG_iteration"] == 1
        )
    else:
        cg_pctsp_obj = prize_collecting_tsp(None, forbidden_set, dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle)
        new_columns = cg_pctsp_obj.cg_pctsp()


    stats["CG_DP_time"] += time.perf_counter() - start_2

    for array in new_columns:
        new_columns_to_add.add(tuple(array))

    return new_columns, feasibility_memo, stats["CG_DP_time"], status, new_columns_to_add, new_constraints


def column_generation(branching_arc, forbidden_set=[], tsp_memo={}, L=None,
                      feasibility_memo={}, global_tsp_memo={}, initial=False,
                      parent_constraints=set()):

    not_fractional = False
    stats = dict(CG_iteration=0, RG_iteration=0, RG_time=0, CG_time=0,
                 CG_DP_time=0, RG_DP_time=0, LP_time=0, num_lp=0)
    new_columns_to_add = set()
    new_constraints = copy.deepcopy(parent_constraints) if (parent_constraints and not always_generate_rows) else set()
    num_lp = 0
    master_prob = MasterProblem(forbidden_set)
    sub_problem = SubProblem(forbidden_set)

    start_4 = time.perf_counter()

    while True:
        stats["CG_iteration"] += 1
        print(f"CG iteration count: {stats['CG_iteration']}")

        start_lp = time.perf_counter()
        p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(
            branching_arc, new_columns_to_add, new_constraints, initial_lp=(stats["CG_iteration"] == 1)
        )
        stats["LP_time"] += time.perf_counter() - start_lp
        stats["num_lp"] += 1

        if not y_r_result:
            return CGResult(
                y_r_result=None, not_fractional=False,
                model=None, objval=None, status=status,
                tsp_memo=tsp_memo, feasibility_memo=feasibility_memo,
                global_tsp_memo=global_tsp_memo,
                new_constraints=new_constraints, **stats
            )

        new_columns, feasibility_memo, stats["CG_DP_time"], status, new_columns_to_add, new_constraints = run_CGSP(
            master_prob, sub_problem, new_columns_to_add, feasibility_memo, 
            new_constraints, stats, status, forbidden_set
            )

        if not new_columns:
            break

    if check_values(y_r_result):
        print("All non-zero values are 1")
        not_fractional = True

    stats["CG_time"] = time.perf_counter() - start_4

    return CGResult(
        y_r_result=y_r_result,
        not_fractional=not_fractional,
        model=master_prob_model,
        objval=master_prob_model.ObjVal,
        status=master_prob_model.status,
        tsp_memo=tsp_memo,
        feasibility_memo=feasibility_memo,
        global_tsp_memo=global_tsp_memo,
        new_constraints=new_constraints,
        **stats
    )