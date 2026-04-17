edges = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (2,4)]
vertices = 5
adjList = [[] for _ in range(vertices)]
parent = {'0':None}
cycle = []
visited = [False]*vertices
# add edges to the undirected graph
for (src, dest) in edges:
    # add an edge from source to destination   
    adjList[src].append(dest)
    adjList[dest].append(src)
for i,j in enumerate(adjList):
    adjList[i] = list(set(adjList[i]))

stack = []
start_node = 0
stack.append(0)

while len(stack)!=0:
    u = stack.pop()
    cycle.append(u) 
    if visited[u]==False:
        visited[u]=True
    else:
        print(cycle.pop())
        while True:
            c = cycle.pop()
            print(c)
            if c==u:
                break
    for item in adjList[u]:
        if parent[f'{u}']!=item:
            stack.append(item)
            parent[f'{item}']=u
