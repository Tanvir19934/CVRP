from gurobipy import GRB
import heapq
from typing import Dict, List, Tuple, Optional
from pricing_coalition_new import column_generation
from config_new import V, Q_EV, q, a
from collections import defaultdict
import copy
import time
import random
import numpy as np
from utils_new import print_solution

random.seed(20)

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

def branching() -> None:

    global_stack = []
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

    outer_iter = 0
    
    #loop through the stack
    while stack:
        outer_iter+=1
        print(f"outer iteration count: {outer_iter}")
        print(f"fractional solution found so far = {frac_count}. Size of stack = {len(stack)}")
        
        node = heapq.heappop(stack)
        if node.not_fractional==True:
            frac_count+=1
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
                    if not (abs(model_vars[var] - 0) <= tol or abs(model_vars[var] - 1) <= tol) and var.split('[')[0]=='y_r_':
                        fractional_var[var] = model_vars[var]
                for item in fractional_var:
                    element = eval(item.split('[')[-1][0:-1])
                    for i in range(len(element)-1):
                        flow_vars[(element[i],element[i+1])]= model_vars[item]
                
                branching_arcs= {arc:flow_vars[arc] for arc in flow_vars if flow_vars[arc]!=1}
                #branching_arc= {arc:flow_vars[arc] for arc in flow_vars if arc[0]!=0 and arc[1]!=0}]
                if not branching_arcs:
                    continue
                
                while True:
                    branching_arc = random.choice(list(branching_arcs.keys()))
                    if branching_arc[0]==0 and outer_iter==1:
                        continue
                    else:
                        break
                #branching_arc = sorted(branching_arc, key=lambda k: abs(branching_arc[k] - 0.5))[0]
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
                y_r_result, left_not_fractional, master_prob_model, obj_val, left_status = column_generation(left_node.adj, left_node.forbidden, left_node.allowed)
                if left_status!=3:
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
                y_r_result, right_not_fractional, master_prob_model, obj_val, right_status = column_generation(right_node.adj,right_node.forbidden,right_node.allowed)
                if right_status!=3:
                    if right_not_fractional:
                        right_node.not_fractional = True  
                    right_node.obj_val = obj_val
                    right_node.model = master_prob_model
                    right_node.solution = y_r_result
                    if right_node.name not in global_stack:
                        heapq.heappush(stack, right_node)
                        global_stack.append(right_node.name)
                    #if left_status==3 and right_status==3 and len(stack)==0 and len(branching_arcs)>1:
                    #    branching_arcs.pop(branching_arc)
                    #else:
                    #    break

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