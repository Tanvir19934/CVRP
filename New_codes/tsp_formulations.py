from config_new import (
    q, a, EV_velocity, gamma, gamma_l, EV_cost, tol,
    GV_cost, w_dv, w_ev, theta, battery_threshold, V, N, Q_EV,
)
from gurobipy import Model, GRB, quicksum
from itertools import permutations
import numpy as np

np.random.seed(42)

def gv_tsp_cost(route):
    cost_GV = a[(route[0],route[1])]*GV_cost
    l = 0
    q[0]=0
    try:
        route = eval(route)
    except:
        for i in range(1,len(route)-1):
            l+=q[route[i]]
            cost_GV += a[(route[i],route[i+1])]*GV_cost*l
        return cost_GV

def tsp_tour(route):
    
    if len(route) == 3 and route[0]==0:
        return route, a[(0,route[1])]* GV_cost * q[route[1]] + a[(route[1],0)] * GV_cost

    intermediate_nodes = route[1:-1]
    all_routes = [[0] + list(p) + [0] for p in permutations(intermediate_nodes)]
    routes_list = [tuple(all_routes) for all_routes in all_routes]
    route_cost = {}
    for item in routes_list:
        route_cost[tuple(item)] = gv_tsp_cost(item)

    model = Model("TSP")
    x = model.addVars(routes_list, vtype=GRB.BINARY, name="x")

    model.addConstr(quicksum(x[i] for i in routes_list) == 1)

    model.setObjective(quicksum(route_cost[i] * x[i] for i in routes_list), GRB.MINIMIZE)

    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.00000001
    model.optimize()
    
    # Extract the solution
    if model.status == GRB.OPTIMAL:
        for i in routes_list:
            if x[i].X > 0.5:
                tour = i
                break
        return tour, model.getObjective().getValue()
    

def compute_bigM(a_ij, c_e, v_e, gamma_o, gamma_l, max_load=10, eps=1e-3):
    C_max = max((c_e * a / v_e) * (gamma_o + gamma_l * max_load) for a in a_ij.values())
    return (1 - eps) + C_max
big_M = compute_bigM(a, c_e=EV_cost, v_e=EV_velocity, gamma_o=gamma, gamma_l=gamma_l, max_load=Q_EV, eps=battery_threshold)

class prize_collecting_tsp:
    def __init__(self, p_result=None, forbidden_set=None, dual_values_delta=None, dual_values_subsidy=None, dual_values_IR=None, dual_values_vehicle=None):
        self.p_result = p_result
        self.forbidden_set = forbidden_set
        self.dual_values_delta = dual_values_delta
        self.dual_values_subsidy = dual_values_subsidy
        self.dual_values_IR = dual_values_IR
        self.dual_values_vehicle = dual_values_vehicle
        self.big_M = big_M

    def pctsp(self):
        # Decision variables
        self.m = Model("PrizeCollectingTSP")
        self.x = self.m.addVars(V, V, vtype=GRB.BINARY, name="x")      # arc used
        self.y = self.m.addVars(V, vtype=GRB.BINARY, name="y")         # node visited
        self.f = self.m.addVars(V, V, vtype=GRB.CONTINUOUS, lb=0.0, ub=Q_EV, name="f")  # flow on arc
        # Flow balance: each visited node must have exactly one in/out arc
        for i in V:
            self.m.addConstr(quicksum(self.x[i, j] for j in V if j != i) == self.y[i])
            self.m.addConstr(quicksum(self.x[j, i] for j in V if j != i) == self.y[i])

        # Depot must be visited
        self.m.addConstr(self.y[0] == 1)

        # Truck starts empty at depot
        self.m.addConstr(quicksum(self.f[0, j] for j in V if j != 0) == 0, name="DepotStartEmpty")

        # Truck returns to depot carrying total pickups collected
        self.m.addConstr(quicksum(self.f[j, 0] for j in V if j != 0) ==
                    quicksum(q[i] * self.y[i] for i in N),
                    name="DepotReturnFull")

        # Flow capacity: if arc is not used, no flow
        for i in V:
            for j in V:
                if i != j:
                    self.m.addConstr(self.f[i, j] <= Q_EV * self.x[i, j])

        # Flow conservation for pickups
        for i in N:  # customers only
            self.m.addConstr(
                quicksum(self.f[i, j] for j in V if j != i)
                - quicksum(self.f[j, i] for j in V if j != i)
                == q[i] * self.y[i]
            )
        self.m.update()
        return self.m

    def cg_pctsp(self):
        print("\n Executing pctsp for CG... \n")
        self.m = self.pctsp()

        # v_ij represents the fraction of usable battery consumed on arc (i,j)
        self.v = self.m.addVars(V, V, vtype=GRB.CONTINUOUS, lb=0.0, ub = 1, name="v")

        # forbid certain arcs
        self.m.addConstrs((self.x[i, j] == 0 for (i, j) in self.forbidden_set), name="forbidden_arcs")
                 
        # battery initialization
        self.m.addConstrs(self.v[0, j] == ((a[0,j]/EV_velocity)) * gamma * self.x[0, j] for j in N)

        # Battery flow conservation
        for i in N:
            lhs = quicksum(self.v[i, j] for j in V if j != i) - quicksum(self.v[j, i] for j in V if j != i)
            rhs = quicksum((a[i,j]/EV_velocity) * (gamma * self.x[i, j] + gamma_l * self.f[i, j])
                        for j in V if j != i)
            self.m.addConstr(lhs == rhs, name=f"BattFlow[{i}]")

        # Link energy flow to arc usage: prevents "phantom" battery flow on unused arcs
        # (v_ij = 0 if x_ij = 0; ensures battery consumption only occurs along active routes)
        self.m.addConstrs(self.v[i,j] <= (1 - battery_threshold) * self.x[i,j]
                        for i in V for j in V if i != j)

        # return battery requirement at depot
        self.m.addConstrs(
            (
                self.v[i, j] + (a[j,0]/EV_velocity) * (gamma * self.x[i, j] + gamma_l * self.f[i, j])
                <= (1 - battery_threshold) * self.x[i, j]            
            )
            for i in V for j in V if i != j
        )

        # no [0, n, 0] type routes
        self.m.addConstrs(self.x[0, j] + self.x[j, 0] <= 1 for j in V if j != 0)

        # Objective
        self.m.setObjective(
            quicksum(w_ev*a[i,j]*self.x[i,j]  for i in V for j in V if i != j)   # base distance cost
            + (theta-self.dual_values_subsidy)* quicksum(260*EV_cost*(a[i,j]/EV_velocity)*(gamma*self.x[i,j]+gamma_l*(self.f[i,j])) for i in V for j in V if i != j)
            - quicksum(self.dual_values_delta[i]*self.y[i] for i in N)
            - self.dual_values_vehicle
            - quicksum(self.dual_values_IR[i]*self.y[i]*(a[i,0]*GV_cost*q[i]+a[i,0]*GV_cost) for i in N),
            GRB.MINIMIZE
        )
        
        # show/dont show log
        self.m.Params.OutputFlag = 1
        #self.m.Params.PoolSearchMode = 1     # find multiple solutions
        #self.m.Params.PoolSolutions = 100    # maximum number of solutions to keep
        self.update_optimize_check_feasibility(self.m, iis_path="model.ilp")
        results = self.extract_solution_pool_tours(self.m, V, self.x)
        return results

    def rg_pctsp(self):
        """
        Prize-Collecting TSP with load-dependent travel costs.
        Flow-based formulation (no big-M load variables).
        Collects all negative-valued solutions.
        """
        print("\n Executing pctsp (arc-based) for RG... \n")

        # Map prizes to nodes
        prizes = {i: self.p_result.get(f"p_{i}", 0.0) for i in N}
        prizes[0] = 0.0  # depot has no prize

        self.m = self.pctsp()

        # Objective = base distance cost + load*distance cost – collected prizes
        self.m.setObjective(
            quicksum(a[0, j] * self.x[0, j] * GV_cost for j in N)   # base distance cost
            + quicksum(a[i, j] * self.f[i, j] * GV_cost for i in V for j in V if i != j) # load * distance cost
            - quicksum(prizes[i] * self.y[i] for i in V),                                # collected prizes
            GRB.MINIMIZE
        )
        
        self.m.Params.OutputFlag = 1

        self.update_optimize_check_feasibility(self.m, iis_path="model.ilp")
        results = self.extract_solution_pool_tours(
            self.m, V, self.x,
            compute_details=True,
            cost_func=gv_tsp_cost,
            prizes=prizes
        )

        return results

    def cg_pctsp_node_based(self):
        """
        Prize-Collecting TSP with load-dependent travel costs.
        Node-based formulation (big-M load variables).
        Collects all negative-valued solutions.
        """
        print("\n Executing pctsp (node-based) for CG... \n")

        self.m = self.pctsp()
        self.b = self.m.addVars(V + ['t'], vtype=GRB.CONTINUOUS, ub = 1, lb = 0, name="b")         # battery level
        self.m.addConstr(self.b[0] == 1, name="DepotBatteryFull")                          # depot starts with full battery
        self.m.addConstrs(self.b[i] >= battery_threshold for i in V + ['t'])                       # min battery at customers
        self.m.addConstrs(
            self.b[j] <= self.b[i] - (a.get((i,j),a[i,0])/EV_velocity)*(gamma*self.x[i,j]+gamma_l*self.f.get((i,j),self.f[i,0])) + self.big_M * (1-self.x[i,j])
            for i in V for j in N + ['t'] if (i != j and (i!=0 and j!='t'))
            )  # battery depletion

        # forbid certain arcs
        self.m.addConstrs((self.x[i, j] == 0 for (i, j) in self.forbidden_set), name="forbidden_arcs")

        # no [0, n, 0] type routes
        self.m.addConstrs(self.x[0, j] + self.x[j, 0] <= 1 for j in V if j != 0)

        self.m.setObjective(
            quicksum(w_ev*a[i,j]*self.x[i,j]  for i in V for j in V if i != j)   # base distance cost
            + (theta-self.dual_values_subsidy)* quicksum(260*EV_cost*(a[i,j]/EV_velocity)*(gamma*self.x[i,j]+gamma_l*(self.f[i,j])) for i in V for j in V if i != j)
            - self.dual_values_vehicle
            - quicksum(self.dual_values_delta[i]*self.y[i] for i in N)
            - quicksum(self.dual_values_IR[i]*self.y[i]* (a[i,0]*GV_cost*q[i]+a[i,0]*GV_cost) for i in N)
            + - tol*0.001*(self.b['t']),     # to encourage the correct battery level at depot, otherwise Gurobi may set it to artificially small value to reduce cost
            GRB.MINIMIZE
        )

        self.m.setParam("OutputFlag", 1)

        self.update_optimize_check_feasibility(self.m, iis_path="model.ilp")
        results = self.extract_solution_pool_tours(self.m, V, self.x)
        return results

    @staticmethod
    def extract_solution_pool_tours(model, V, x, tol=1e-6,
                                    compute_details=False,
                                    cost_func=None,
                                    prizes=None):
        """
        Extract tours from Gurobi solution pool.

        Parameters
        ----------
        model : gurobipy.Model
            The solved model.
        V : iterable
            Set/list of nodes.
        x : dict or tuple-indexed Gurobi Var
            Edge decision variables (x[i, j]).
        tol : float, optional
            Tolerance for positive edge selection (default = 1e-6).
        compute_details : bool, optional
            If True, also compute (travel_cost, collected_prizes) for each tour.
        cost_func : callable, optional
            Function like gv_tsp_cost(tour), required if compute_details=True.
        prizes : dict, optional
            Node → prize mapping, required if compute_details=True.

        Returns
        -------
        results : list
            If compute_details=False: [(tour,), ...]
            If compute_details=True:  [(tour, obj_val, travel_cost, collected_prizes), ...]
        """
        results = []

        if model.SolCount == 0:
            return results

        for k in range(model.SolCount):
            model.setParam(GRB.Param.SolutionNumber, k)
            obj_val = model.PoolObjVal

            if obj_val < -tol and abs(obj_val) > 0.001:
                tour = [0]
                current = 0

                while True:
                    next_nodes = [j for j in V if j != current and x[current, j].Xn > 0.5]
                    if not next_nodes:
                        break
                    nxt = next_nodes[0]
                    tour.append(nxt)
                    if nxt == 0:
                        break
                    current = nxt

                if len(tour) > 3:
                    if compute_details:
                        travel_cost = cost_func(tour) if cost_func else None
                        collected_prizes_val = sum(prizes[i] for i in tour) if prizes else None
                        results.append((tour, obj_val, travel_cost, collected_prizes_val))
                    else:
                        results.append(tuple(tour))

        return results

    @staticmethod
    def extract_solution_pool_tours(model, V, x, tol=1e-6,
                                    compute_details=False,
                                    cost_func=None,
                                    prizes=None):
        """
        Extract tours from Gurobi solution pool.

        Parameters
        ----------
        model : gurobipy.Model
            The solved model.
        V : iterable
            Set/list of nodes.
        x : dict or tuple-indexed Gurobi Var
            Edge decision variables (x[i, j]).
        tol : float, optional
            Tolerance for positive edge selection (default = 1e-6).
        compute_details : bool, optional
            If True, also compute (travel_cost, collected_prizes) for each tour.
        cost_func : callable, optional
            Function like gv_tsp_cost(tour), required if compute_details=True.
        prizes : dict, optional
            Node → prize mapping, required if compute_details=True.

        Returns
        -------
        results : list
            If compute_details=False: [(tour,), ...]
            If compute_details=True:  [(tour, obj_val, travel_cost, collected_prizes), ...]
        """
        results = []

        if model.SolCount == 0:
            return results

        for k in range(model.SolCount):
            model.setParam(GRB.Param.SolutionNumber, k)
            obj_val = model.PoolObjVal

            if obj_val < -tol and abs(obj_val) > 0.001:
                tour = [0]
                current = 0

                while True:
                    next_nodes = [j for j in V if j != current and x[current, j].Xn > 0.5]
                    if not next_nodes:
                        break
                    nxt = next_nodes[0]
                    tour.append(nxt)
                    if nxt == 0:
                        break
                    current = nxt

                if len(tour) > 3:
                    if compute_details:
                        travel_cost = cost_func(tour) if cost_func else None
                        collected_prizes_val = sum(prizes[i] for i in tour) if prizes else None
                        results.append((tour, obj_val, travel_cost, collected_prizes_val))
                    else:
                        results.append(tuple(tour))

        return results

    @staticmethod
    def update_optimize_check_feasibility(model, iis_path="model.ilp"):
        model.update()
        model.optimize()
        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            model.write(iis_path)
            print(f"IIS written to {iis_path}")

