from gurobipy import GRB
import heapq
from typing import Dict
from pricing import column_generation
from config import V
from collections import defaultdict
import copy
import time
class Node:
    
    def __init__(self, depth, name, adj, forbidden, parent):
        #self.model = model
        self.depth = depth
        self.solution = None
        self.obj_val = None
        self.name = name
        self.adj = adj
        self.parent = parent
        self.forbidden = forbidden
        self.not_fractional = False
        self.solution = None
        print("depth =",self.depth)  #sanity check

    def __lt__(self, other):
        return self.obj_val > other.obj_val  # For heap implementation. The heapq.heapify() will heapify the list based on this criteria.
        
def branching() -> None:

    global_stack = []
    
    adj = {node: [n for n in V if n != node] for node in V}

    # Create the root node by solving the initial rmp
    y_r_result, not_fractional, master_prob_model, obj_val, status = column_generation(adj,forbidden_set=[])

    if not_fractional:
        print("Optimal solution found at the root node: did not need branching")
        return
    
    left_not_fractional = right_not_fractional = False
    root_node = Node(0, "root_node", adj, set(), None)
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
                    if not (abs(model_vars[var] - 0) <= tol or abs(model_vars[var] - 1) <= tol):
                        fractional_var[var] = model_vars[var]
                for item in fractional_var:
                    element = eval(item.split('[')[-1][0:-1])
                    for i in range(len(element)-1):
                        flow_vars[(element[i],element[i+1])]+=model_vars[item]
                
                branching_arc= {arc:flow_vars[arc] for arc in flow_vars}
                #branching_arc= {arc:flow_vars[arc] for arc in flow_vars if arc[0]!=0 and arc[1]!=0}


                branching_arc = sorted(branching_arc, key=lambda k: abs(branching_arc[k] - 0.5))[0]
                if branching_arc==(0,7):
                    pass
                if not branching_arc:
                    break

                # Create left branch node
                left_node = Node(node.depth + 1, f'{branching_arc}={0}', [], copy.deepcopy(node.forbidden) ,node)
                left_node.adj = copy.deepcopy(left_node.parent.adj)
                try:
                    left_node.adj[branching_arc[0]].remove(branching_arc[1])
                except: pass

                # enforcing branching_arc = 0
                left_node.forbidden.add(branching_arc)
                #left_node.forbidden.add(tuple(list(branching_arc)[::-1]))
                print(f"branching {branching_arc}={0}")
                y_r_result, left_not_fractional, master_prob_model, obj_val, status = column_generation(left_node.adj, left_node.forbidden)
                if status!=3:
                    if left_not_fractional:
                        left_node.not_fractional = True
                    left_node.obj_val = obj_val
                    left_node.model = master_prob_model
                    left_node.solution = y_r_result
                    heapq.heappush(stack, left_node)
                    global_stack.append(left_node.name)
                    
                    
                # Create right branch node
                right_node = Node(node.depth + 1, f'{branching_arc}={1}', [], copy.deepcopy(node.forbidden) ,node)
                right_node.adj = copy.deepcopy(right_node.parent.adj)

                # enforcing branching_arc = 1
                right_node.adj[branching_arc[0]] = [branching_arc[1]]
                try:
                    right_node.adj[branching_arc[1]].remove(branching_arc[0])
                except: pass
                for item in V:
                    if item!=branching_arc[1] and item!=branching_arc[0]:
                        right_node.forbidden.add((branching_arc[0],item))
                print(f"branching {branching_arc}={1}")
                y_r_result, right_not_fractional, master_prob_model, obj_val, status = column_generation(right_node.adj,right_node.forbidden)
                if status!=3:
                    if right_not_fractional:
                        right_node.not_fractional = True  
                    right_node.obj_val = obj_val
                    right_node.model = master_prob_model
                    right_node.solution = y_r_result
                    heapq.heappush(stack, right_node)
                    global_stack.append(right_node.name)
        if len(stack)==0:
            print("stack is empty")
    print(frac_count)
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
    start =time.perf_counter()
    branching()
    end =time.perf_counter()
    print(f"execution time = {end-start}")

if __name__ == "__main__":
    main()