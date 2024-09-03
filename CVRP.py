import numpy as np
from matplotlib import pyplot as plt
import gurobipy
import networkx as nx
from gurobipy import Model, GRB, quicksum



rnd = np.random
rnd.seed(10)
n = 10 #number of clients
xc = rnd.rand(n+1)*200
yc = rnd.rand(n+1)*100
plt.plot(xc[0],yc[0], c = 'r', marker = 's') #depot
plt.scatter(xc[1:], yc[1:], c = 'b') #clients
for i, j in zip(xc, yc):
   plt.text(i, j+1, '({}, {})'.format(i, j))

N = [i for i in range(1,n+1)]
V = [0] + N
A = [(i,j) for i in V for j in V if i!=j]
c = {(i,j): np.hypot(xc[i]-xc[j], yc[i]-yc[j]) for (i,j) in A} #eucledian distance
Q = 10 #capacity of the vehicles
q = {i: rnd.randint(1,10) for i in N} #demand for customers
mdl = Model('CVRP')

x = mdl.addVars(A, vtype = GRB.BINARY)
u = mdl.addVars(N, vtype=GRB.CONTINUOUS)
print(type(x))
print(u)

mdl.modelSense = GRB.MINIMIZE
mdl.setObjective(quicksum(x[i,j]*c[i,j] for i,j in A))
mdl.addConstrs(quicksum(x[i,j] for j in V if j!=i)==1 for i in N)
print(mdl.addConstrs(quicksum(x[i,j] for j in V if j!=i)==1 for i in N),10000000000000000000000)
mdl.addConstrs(quicksum(x[i,j] for i in V if j!=i)==1 for j in N)
mdl.addConstrs((x[i,j]==1) >> (u[i]+q[j]==u[j])
for i,j in A if i!=0 and j!=0)
mdl.addConstrs(u[i]>=q[i] for i in N)
mdl.addConstrs(u[i]<=Q for i in N)
mdl.update()

mdl
mdl.Params.MIPGap = 0.05
mdl.Params.TimeLimit = 1000 #seconds
mdl.optimize()
active_arcs = [a for a in A if x[a].x>0.99]
s = []
t = []
for i in active_arcs:
  s.append(str(i[0]))
  t.append(str(i[1]))
#tuple(zip(s, t))
for i, j in active_arcs:
    plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='g', zorder=0)
plt.plot(xc[0], yc[0], c='r', marker='s')
plt.scatter(xc[1:], yc[1:], c='b')


G = nx.DiGraph(directed=True)
G.add_edges_from(tuple(zip(s, t)))    #active_arcs
options = {
    'node_color': 'green',
    'node_size': 800,
    'width': 2,
    'arrowstyle': '-|>',
    'arrowsize': 15,
}


# coordinates of xc and yc
locations = tuple(zip(xc, yc))

# generating position dictionary
pos = {str(i):location for i, location in enumerate(locations)}

# drawing graph, with positions included.  
nx.draw_networkx(G, pos=pos, arrows=True, **options)
plt.show()
