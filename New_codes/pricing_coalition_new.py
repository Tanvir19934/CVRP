from models_coalition_new import SubProblem, MasterProblem
from utils_new import check_values, tsp_tour, prize_collecting_tsp, CGResult
from config_new import always_generate_rows, use_column_heuristic, rand_seed
import time
import copy
import random

random.seed(rand_seed)

def run_CGSP(master_prob, sub_problem, new_columns_to_add, feasibility_memo,
             new_constraints, stats, status):
    """Run Column Generation Subproblem once (dual extraction + dy_prog)."""
    dual_values_delta, dual_values_subsidy, dual_values_IR, dual_values_vehicle = master_prob.getDuals()
    if dual_values_delta is None:
        return None, feasibility_memo, stats["CG_DP_time"], status, new_columns_to_add, new_constraints

    start_2 = time.perf_counter()
    new_columns, feasibility_memo = sub_problem.dy_prog(
        dual_values_delta, dual_values_subsidy, dual_values_IR,
        dual_values_vehicle, feasibility_memo, stats["CG_iteration"] == 1
    )
    stats["CG_DP_time"] += time.perf_counter() - start_2

    if not new_columns:
        return None, feasibility_memo, stats["CG_DP_time"], status, new_columns_to_add, new_constraints

    for array in new_columns:
        new_columns_to_add.add(tuple(array))

    return new_columns, feasibility_memo, stats["CG_DP_time"], status, new_columns_to_add, new_constraints


def apply_column_heuristic(new_columns_to_add, global_tsp_memo, new_constraints):
    """Apply column heuristic step to strengthen formulation."""
    if use_column_heuristic:
        for array in new_columns_to_add:
            sorted_array = tuple([0] + sorted(array[1:-1]) + [0])
            if sorted_array in global_tsp_memo:
                new_column_constr = global_tsp_memo[sorted_array]
            else:
                new_column_constr = tsp_tour(array)
                global_tsp_memo[sorted_array] = new_column_constr
            new_constraints.add(new_column_constr)
    return new_constraints, global_tsp_memo


def run_RGSP(master_prob, branching_arc, new_columns_to_add, new_constraints,
             stats, tsp_memo, feasibility_memo, global_tsp_memo):
    """Run RGSP loop until no new constraints are found, update stats dict."""
    start_3 = time.perf_counter()

    while True:
        stats["RG_iteration"] += 1
        print(f"RG iteration count: {stats['RG_iteration']}")

        start_lp = time.perf_counter()
        if stats["CG_iteration"] == 1 and stats["RG_iteration"] == 1:
            p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(
                branching_arc, new_columns_to_add, new_constraints, True
            )
            break
        else:
            p_result, y_r_result, master_prob_model, status = master_prob.relaxedLP(
                branching_arc, new_columns_to_add, new_constraints, False
            )
            print(master_prob_model.ObjVal)

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

        start_5 = time.perf_counter()
        rg_pctsp_obj = prize_collecting_tsp(p_result)
        new_route = rg_pctsp_obj.rg_pctsp()
        stats["RG_DP_time"] += time.perf_counter() - start_5

        if not new_route:
            break
        for item in new_route:
            new_constraints.add((tuple(item[0]), item[2]))

    stats["RG_time"] += time.perf_counter() - start_3

    return (
        p_result, y_r_result, master_prob_model, status,
        tsp_memo, feasibility_memo, global_tsp_memo,
        new_constraints, stats
    )


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

    if always_generate_rows or initial:
        while True:
            stats["CG_iteration"] += 1
            print(f"CG iteration count: {stats['CG_iteration']}")

            (p_result, y_r_result, master_prob_model, status,
             tsp_memo, feasibility_memo, global_tsp_memo,
             new_constraints, stats) = run_RGSP(
                master_prob, branching_arc, new_columns_to_add, new_constraints,
                stats, tsp_memo, feasibility_memo, global_tsp_memo
            )

            new_columns, feasibility_memo, stats["CG_DP_time"], status, new_columns_to_add, new_constraints = run_CGSP(
                master_prob, sub_problem, new_columns_to_add, feasibility_memo,
                new_constraints, stats, status
                )

            if not new_columns:  # stop if no new columns
                break

        new_constraints, global_tsp_memo = apply_column_heuristic(
            new_columns_to_add, global_tsp_memo, new_constraints
        )

    else:
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
                new_constraints, stats, status
                )

            if not new_columns:
                break

        new_constraints, global_tsp_memo = apply_column_heuristic(
            new_columns_to_add, global_tsp_memo, new_constraints
        )

        if check_values(y_r_result):
            print("Integer solution has been hit, starting row generation")
            (p_result, y_r_result, master_prob_model, status,
             tsp_memo, feasibility_memo, global_tsp_memo,
             new_constraints, stats) = run_RGSP(
                master_prob, branching_arc, new_columns_to_add, new_constraints,
                stats, tsp_memo, feasibility_memo, global_tsp_memo
            )

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