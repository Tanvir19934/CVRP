from gurobipy import GRB
import heapq
from typing import Dict
from pricing_coalition_dual import column_generation
from config import V, Q_EV, q, a
from collections import defaultdict
import copy
import time
from utils import ev_travel_cost
import random
import numpy as np
[[0, 5, 2, 0], [0, 1, 11, 0], [0, 3, 7, 9, 8, 0], [0, 4, 6, 0]]

#random.seed(42)

class Node:
    
    def __init__(self, depth, name, adj, forbidden, allowed, parent):
        #self.model = model
        self.depth = depth
        self.solution = None
        self.obj_val = None
        self.name = name
        self.adj = adj
        self.parent = parent
        self.forbidden = forbidden
        self.allowed = allowed
        self.not_fractional = False
        self.solution = None
        print("depth =",self.depth)  #sanity check

    def __lt__(self, other):
        return self.obj_val > other.obj_val  # For heap implementation. The heapq.heapify() will heapify the list based on this criteria.

def print_solution(final_model) -> None:
    '''This function prints the solution of the cutting stock problem'''
    model_vars ={var.VarName: var.X for var in final_model.getVars()}
    var_names = list(model_vars.keys())
    sol = []
    solution_routes = []
    for var in var_names:
        if final_model.getVarByName(var).x!=0 and var.split('[')[0]=='y_r':
            print(f"{var}: {final_model.getVarByName(var).x}")
            solution_routes.append(list(eval(var[4:-1])))
        sol.append(final_model.getVarByName(var).x)
    print(f"Objective value: {final_model.getObjective().getValue()}")

    P = {}
    e_S = {}
    e_BB= {}
    e_IR= {}
    for item in model_vars:
        cand = item.split('[')[0]
        if cand=='y_r':
            continue
        if model_vars[item]!=0:
            if cand[0:4]=='e_IR':
                e_IR[cand]=model_vars[item]
            elif cand[0:4]=='e_BB':
                e_BB[cand]=model_vars[item]
            elif cand[0:3]=='e_S':
                e_S[cand]=model_vars[item]
            elif cand[0]=='p':
                P[cand]=model_vars[item]

    print(f"Objective value: {final_model.obj_val-0.001*(sum(e_S.values())+sum(e_BB.values())+sum(e_IR.values()))}")
    print(f"total payments: {sum(P.values())}")
    print(f"total stability: {sum(e_S.values())}")
    print(f"total IR: {sum(e_IR.values())}")
    print(f"total BB: {sum(e_BB.values())}")
    print(f"total subsidy: {sum(e_S.values())+sum(e_BB.values())+sum(e_IR.values())}")
    print(P)
    print(e_S)

    tour_cost = 0
    for item in solution_routes:
        if len(item)>3:
            tour_cost += ev_travel_cost(item)[0]
    print(f"total EV cost = {tour_cost}")


     
def branching() -> None:

    global_stack = []
    
    #adj = {node: [n for n in V if n != node] for node in V}
    
    adj = {}
    q[0] = 0
    forbidden_set = set()
    #avg_dist = sum(list(a.values()))/len(a)
    for node in V:
        adj[node] = []
        for n in V:
            if n != node:
                if q[node] + q[n] <= Q_EV:
                    adj[node].append(n)  # Add valid arcs to adj
                else:
                    forbidden_set.add((node, n))  # Add violating arcs to forbidden_set
                    forbidden_set.add((n,node))  # Add violating arcs to forbidden_set
    # Create the root node by solving the initial rmp
    y_r_result, not_fractional, master_prob_model, obj_val, status = column_generation(adj,forbidden_set)

    if not_fractional:
        print("Optimal solution found at the root node: did not need branching")
        print_solution(master_prob_model)
        return
    
    left_not_fractional = right_not_fractional = False
    root_node = Node(0, "root_node", adj, set(), set(), None)
    root_node.solution = y_r_result
    root_node.not_fractional = not_fractional
    root_node.obj_val = obj_val
    root_node.model = master_prob_model
    
    #initialize best node and the stack
    best_node = None
    best_obj = GRB.INFINITY
    stack = [root_node]
    tol = 0.01
    heapq.heapify(stack)  #we are going to traverse in best first manner
    frac_count = 0
    
    #loop through the stack
    while stack:
        
        node = heapq.heappop(stack)
        if node.not_fractional==True:
            frac_count+=1
            print(f"fractional solution found so far = {frac_count}. length of stack = {len(stack)}")
            if node.obj_val < best_obj:
                best_obj = node.obj_val
                best_node = node
            continue
        else:  
            if node.obj_val > best_obj:
                continue
            else:
                flow_vars = defaultdict(float)
                model_vars ={var.VarName: var.X for var in node.model.getVars()}
                var_names = list(model_vars.keys())
                fractional_var = {}
                for var in var_names:
                    if not (abs(model_vars[var] - 0) <= tol or abs(model_vars[var] - 1) <= tol) and var.split('[')[0]=='y_r':
                        fractional_var[var] = model_vars[var]
                for item in fractional_var:
                    element = eval(item.split('[')[-1][0:-1])
                    for i in range(len(element)-1):
                        flow_vars[(element[i],element[i+1])]+=model_vars[item]
                
                branching_arc= {arc:flow_vars[arc] for arc in flow_vars if flow_vars[arc]!=1}
                #branching_arc= {arc:flow_vars[arc] for arc in flow_vars if arc[0]!=0 and arc[1]!=0}

                branching_arc = random.choice(list(branching_arc.keys()))
                #branching_arc = sorted(branching_arc, key=lambda k: abs(branching_arc[k] - 0.5))[0]

                if not branching_arc:
                    break

                # Create left branch node
                left_node = Node(node.depth + 1, f'{branching_arc}={0}', [], copy.deepcopy(node.forbidden) , copy.deepcopy(node.allowed), node)
                left_node.adj = copy.deepcopy(left_node.parent.adj)
                try:
                    left_node.adj[branching_arc[0]].remove(branching_arc[1])
                except: pass

                # enforcing branching_arc = 0
                left_node.forbidden.add(branching_arc)
                #left_node.forbidden.add(tuple(list(branching_arc)[::-1]))
                print(f"branching {branching_arc}={0}")
                if branching_arc==(0,1):
                    pass
                y_r_result, left_not_fractional, master_prob_model, obj_val, status = column_generation(left_node.adj, left_node.forbidden, left_node.allowed)
                if status!=3:
                    if left_not_fractional:
                        left_node.not_fractional = True
                    left_node.obj_val = obj_val
                    left_node.model = master_prob_model
                    left_node.solution = y_r_result
                    if left_node.name not in global_stack:
                        heapq.heappush(stack, left_node)
                        global_stack.append(left_node.name)
                    
                    
                    
                # Create right branch node
                right_node = Node(node.depth + 1, f'{branching_arc}={1}', [], copy.deepcopy(node.forbidden), copy.deepcopy(node.allowed), node)
                right_node.adj = copy.deepcopy(right_node.parent.adj)
                right_node.allowed.add(branching_arc)

                # enforcing branching_arc = 1
                right_node.adj[branching_arc[0]] = [branching_arc[1]]
                try:
                    right_node.adj[branching_arc[1]].remove(branching_arc[0])
                except: pass
                for item in V:
                    if item!=branching_arc[1] and item!=branching_arc[0]:
                        right_node.forbidden.add((branching_arc[0],item))
                print(f"branching {branching_arc}={1}")
                y_r_result, right_not_fractional, master_prob_model, obj_val, status = column_generation(right_node.adj,right_node.forbidden,right_node.allowed)
                if status!=3:
                    if right_not_fractional:
                        right_node.not_fractional = True  
                    right_node.obj_val = obj_val
                    right_node.model = master_prob_model
                    right_node.solution = y_r_result
                    if right_node.name not in global_stack:
                        heapq.heappush(stack, right_node)
                        global_stack.append(right_node.name)
        if len(stack)==0:
            print("stack is empty")
    print(frac_count)
    if best_node:
        print("Optimal solution found:")
        print_solution(best_node.model)
    else:
        print("No optimal solution found.")


def main():
    start =time.perf_counter()
    branching()
    end =time.perf_counter()
    print(f"execution time = {end-start}")

if __name__ == "__main__":
    main()