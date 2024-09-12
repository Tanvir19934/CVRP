from gurobipy import GRB
import heapq
from typing import Dict
import math
from gurobipy import GRB
#from initial_RMP import rmp
import heapq
from pricing import column_generation
from config import a, A, N, V
from collections import defaultdict
import copy
class Node:
    
    def __init__(self, depth, name, adj, parent):
        #self.model = model
        self.depth = depth
        self.solution = None
        self.obj_val = None
        self.name = name
        self.adj = adj
        self.parent = parent
        print("depth =",self.depth)  #sanity check

    def __lt__(self, other):
        return self.obj_val > other.obj_val  # For heap implementation. The heapq.heapify() will heapify the list based on this criteria.
        
def branching() -> None:
    
    adj = {node: [n for n in V if n != node] for node in V}

    stack_list = set()
    # Create the root node by solving the initial rmp
    
    y_r_result, not_fractional, master_prob_model, obj_val, status = column_generation(adj,forbidden_set=[])
    if not_fractional:
        return
    
    root_node = Node(0, "root_node", adj, None)
    
    #initialize best node and the stack
    best_node = None
    best_obj = GRB.INFINITY
    stack = [root_node]
    tol = 0.01
    heapq.heapify(stack)  #we are going to traverse in best first manner
    
    #loop through the stack
    while stack:
        
        node = heapq.heappop(stack)
        node.obj_val = obj_val
        node.model = master_prob_model
        if not_fractional==True:
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
                    if not (abs(model_vars[var] - 0) <= tol or abs(model_vars[var] - 1) <= tol):
                        fractional_var[var] = model_vars[var]
                for item in fractional_var:
                    element = eval(item.split('[')[-1][0:-1])
                    for i in range(len(element)-1):
                        flow_vars[(element[i],element[i+1])]+=model_vars[item]

                branching_arcs = sorted(flow_vars, key=lambda k: abs(flow_vars[k] - 0.5))
                branching_arc = None
                for item in branching_arcs:
                    if f'{item}={1}' not in stack_list:
                        branching_arc=item
                        break
                if not branching_arc:
                    break

                # Create left branch node
                forbidden_set = []

                left_node = Node(node.depth + 1, f'{branching_arc}={0}', [] ,node)
                

                left_node.adj = copy.deepcopy(left_node.parent.adj)
                try:
                    left_node.adj[branching_arc[0]].remove(branching_arc[1])
                except:pass
                try:
                    left_node.adj[branching_arc[1]].remove(branching_arc[0])
                except:pass
                #modified_arcs = copy.deepcopy(a)
                # enforcing branching_arc = 0
                #del modified_arcs[branching_arc]
                #del modified_arcs[tuple(list(branching_arc)[::-1])]
                forbidden_set.append(branching_arc)
                forbidden_set.append(tuple(list(branching_arc)[::-1]))
                
                stack_list.add(f'{branching_arc}={0}')
                #modified_arcs[branching_arc] = penalty
                #modified_arcs[tuple(list(branching_arc)[::-1])] = penalty
                y_r_result, not_fractional, master_prob_model, obj_val, status = column_generation(left_node.adj, forbidden_set)
                if status!=3:
                    left_node.obj_val = obj_val
                    left_node.model = master_prob_model # master_prob.model
                    if f'{branching_arc}={0}' not in stack_list:
                        heapq.heappush(stack, left_node)
                    

                # Create right branch nodes
                forbidden_set = set()
                right_node = Node(node.depth + 1, f'{branching_arc}={1}', [] , node)
                right_node.adj = copy.deepcopy(right_node.parent.adj)
                modified_adj = copy.deepcopy(right_node.adj)
                # enforcing branching_arc = 1
                modified_adj_copy = copy.deepcopy(right_node.adj)
                modified_adj[branching_arc[0]]=[branching_arc[1]]
                modified_adj[branching_arc[1]]=[branching_arc[0]]
                for item in modified_adj_copy:
                    if item!=branching_arc[1] and item!=branching_arc[0]:
                        if branching_arc[1] in modified_adj[item]:
                            modified_adj[item].remove(branching_arc[1])
                            forbidden_set.add((item,branching_arc[1]))
                            forbidden_set.add((branching_arc[1],item))
                        if branching_arc[0] in modified_adj[item]:
                            modified_adj[item].remove(branching_arc[0])
                            forbidden_set.add((item,branching_arc[0]))
                            forbidden_set.add((branching_arc[0],item))
                right_node.adj = copy.deepcopy(modified_adj)
                y_r_result, not_fractional, master_prob_model, obj_val, status = column_generation(right_node.adj,forbidden_set)
                if status!=3:  
                    right_node.obj_val = obj_val
                    right_node.model = master_prob_model #master_prob.model
                    heapq.heappush(stack, right_node)

    if best_node:
        print("Optimal solution found:")
        model_vars ={var.VarName: var.X for var in best_node.model.getVars()}
        var_names = list(model_vars.keys())
        sol = []
        for var in var_names:
            if best_node.model.getVarByName(var).x!=0:
                print(f"{var}: {best_node.model.getVarByName(var).x}")
            sol.append(best_node.model.getVarByName(var).x)
        print(f"Objective value: {best_node.obj_val}")
        constraint_coeffs = retrieve_patterns(best_node.model)
        #for item in constraint_coeffs:
        #    print(constraint_coeffs[item])
    else:
        print("No optimal solution found.")
    
def retrieve_patterns(model) -> Dict:
    '''This function just returns the solution pattern for the cutting stock'''
    constraint_coeffs = {}
    for constr in model.getConstrs():
        coeffs = {}
        for var in model.getVars():
            coeff = model.getCoeff(constr, var)
            coeffs[var.VarName] = coeff
        constraint_coeffs[constr.ConstrName] = coeffs
    return constraint_coeffs

def main():
    branching() #here model is the initial rmp obtained by a simple heuristic from initial_rmp.py
    2

if __name__ == "__main__":
    main()