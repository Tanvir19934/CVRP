import networkx as nx

class graph_algorithm:

    def __init__(self, graph, start_node, goal_node):
        self.graph = graph
        self.start_node = start_node
        self.goal_node = goal_node
    
    def BFS(self):
        visited = set()
        queue = []
        if self.start_node not in visited:
            visited.add(self.start_node)
            queue.append(self.start_node)
            while queue:
                node = queue.pop(0)
                print(node)
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        if neighbor==self.goal_node:
                            print(f'{neighbor}')
                            print('goal found')
                            return
    def BFS_recursion(self,visited,queue):
        if not queue:
            return
        node = queue.pop(0)
        print(node)
        for neighbor in self.graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if neighbor==self.goal_node:
                    print(f'{neighbor}')
                    print('goal found')
                    return
        self.BFS_recursion(visited,queue)
    
    def DFS(self):
        visited = set()
        queue = []
        if self.start_node not in visited:
            queue.append(self.start_node)
            while queue:
                node = queue.pop(0)
                visited.add(node)
                print(node)
                for neighbor in list(reversed(self.graph[node])):
                    if neighbor not in visited and neighbor not in queue:
                        queue.insert(0,neighbor)
                        if neighbor==self.goal_node:
                            print(f'{neighbor}')
                            print('goal found')
                            return
    
    def DFS_recursion(self,visited,queue):
        if not queue:
            return
        node = queue.pop(0)
        visited.add(node)
        print(node)
        for neighbor in list(reversed(self.graph[node])):
            if neighbor not in visited  and neighbor not in queue:
                queue.insert(0,neighbor)
                if neighbor==self.goal_node:
                    print(f'{neighbor}')
                    print('goal found')
                    return
        self.DFS_recursion(visited,queue)

    def uniform_cost_search(self, graph_UCS):
        adjacency_dict = nx.to_dict_of_dicts(graph_UCS)
        visited = set()
        queue = []
        cost_tuple = []
        cost_tuple.append((self.start_node,0))
        queue.append(self.start_node)
        while queue:
            node = queue.pop(0)
            visited.add(node)
            print(node)
            for item in adjacency_dict[node]:
                for elements in cost_tuple:
                    if elements[0] == node:
                        cost_tuple.append((item, elements[1] + adjacency_dict[node][item]['cost']))
            cost_tuple = [(key, value) for key, value in cost_tuple if key != node]
            sorted_list = sorted(cost_tuple, key=lambda x: x[1])
            queue = []
            for item in sorted_list:
                if item[0] not in visited:
                        queue.append(item[0])
                        if item[0]==self.goal_node:
                            print(f'{item[0]}')
                            print('goal found')
                            return

if __name__ == "__main__":
    graph = {
        'S': ['A','B','C'],
        'A': ['D','E','G'],
        'B': ['G'],
        'C': ['G'],
        'D': [],
        'E': [],
        'G': []
    }

    graph_H = {}
    graph_G = {}
    graph_I = {}
    #graph G
    G = nx.DiGraph(directed=True)
    G.add_node('S')
    G.add_node('A')
    G.add_node('B')
    G.add_node('C')
    G.add_node('D')
    G.add_node('E')
    G.add_node('G')
    G.add_edge('S','A', cost = 3)
    G.add_edge('S','B', cost = 1)
    G.add_edge('S','C', cost = 8)
    G.add_edge('A','D', cost = 3)
    G.add_edge('A','E', cost = 7)
    G.add_edge('A','G', cost = 15)
    G.add_edge('B','G', cost = 20)
    G.add_edge('C','G', cost = 5)


    #graph H
    H = nx.DiGraph(directed=True)
    H.add_node('S')
    H.add_node('A')
    H.add_node('B')
    H.add_node('C')
    H.add_node('D')
    H.add_node('E')
    H.add_node('F')
    H.add_node('G1')
    H.add_node('G2')
    H.add_node('G3')
    H.add_node('H')
    H.add_node('I')
    H.add_edge('S','C', cost = 1)
    H.add_edge('C','D', cost = 1)
    H.add_edge('D','B', cost = 1)
    H.add_edge('S','B', cost = 5)
    H.add_edge('S','A', cost = 7)
    H.add_edge('B','A', cost = 9)
    H.add_edge('B','G3', cost = 3)
    H.add_edge('H','G3', cost = 4)
    H.add_edge('B','H', cost = 4)
    H.add_edge('A','F', cost = 9)
    H.add_edge('A','E', cost = 8)
    H.add_edge('F','G2', cost = 2)
    H.add_edge('E','G1', cost = 9)
    H.add_edge('G2','I', cost = 0)
    H.add_edge('I','G1', cost = 2)


    #graph I
    I = nx.DiGraph(directed=True)
    I.add_node('S')
    I.add_node('A')
    I.add_node('B')
    I.add_node('C')
    I.add_node('D')
    I.add_node('E')
    I.add_node('F')
    I.add_node('G1')
    I.add_node('G2')
    I.add_node('G3')
    I.add_edge('S','A', cost = 3)
    I.add_edge('S','B', cost = 1)
    I.add_edge('S','C', cost = 5)
    I.add_edge('A','G1', cost = 10)
    I.add_edge('A','E', cost = 7)
    I.add_edge('E','G1', cost = 2)
    I.add_edge('D','S', cost = 6)
    I.add_edge('D','B', cost = 4)
    I.add_edge('D','G2', cost = 5)
    I.add_edge('B','F', cost = 2)
    I.add_edge('B','C', cost = 2)
    I.add_edge('F','D', cost = 1)
    I.add_edge('C','G3', cost =11)
    I.add_edge('G3','F', cost = 0)

    for node in H.nodes():
        # Get the successors (children) of the current node
        successors = list(H.successors(node))
    
        # Add the node and its successors to the graph dictionary
        graph_H[node] = successors
    for node in G.nodes():
        # Get the successors (children) of the current node
        successors = list(G.successors(node))
    
        # Add the node and its successors to the graph dictionary
        graph_G[node] = successors
    for node in I.nodes():
        # Get the successors (children) of the current node
        successors = list(I.successors(node))
    
        # Add the node and its successors to the graph dictionary
        graph_I[node] = successors

    graph_algorithm_object = graph_algorithm(graph_H, start_node = 'S', goal_node = 'G')
    
    #print("\nBFS\n")
    #graph_algorithm_object.BFS()
    #print("\nBFS with recursion\n")
    #graph_algorithm_object.BFS_recursion(visited = set('S'),queue = ['S'])
    #print("\nDFS\n")
    graph_algorithm_object.DFS()
    print("\nDFS with recursion\n")
    graph_algorithm_object.DFS_recursion(visited = set('S'),queue = ['S'])
    print('\nUniform cost search\n')
    graph_algorithm_object.uniform_cost_search(graph_UCS = H)