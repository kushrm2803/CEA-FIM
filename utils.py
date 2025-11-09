def visualize_set(g, S, all_nodes):
    '''
    Draws a visualization of g, with the nodes in S much bigger/colored and the
    nodes in all_nodes drawn in a medium size. Helpful to visualize seed sets/
    conduct sanity checks.
    '''
    import networkx as nx
    node_color = []
    node_size = []
    for v in g.nodes():
        if v in S:
            node_color.append('b')
            node_size.append(300)
        elif v in all_nodes:
            node_color.append('y')
            node_size.append(100)
        else:
            node_color.append('k')
            node_size.append(20)
    import matplotlib.pyplot as plt
    nx.draw(g, node_color=node_color, node_size=node_size)
    plt.show()

def visualize_communities(g, part, S = None):
    import networkx as nx
    import random
    pos = {}
    centers = [[0, 0], [0, 1.25], [1.25, 0], [1.25, 1.25]]
    for v in g.nodes():
        pos[v] = [centers[part[v]][0] + random.random() - 0.5, centers[part[v]][1] + random.random() - 0.5]
    if not S == None:
        node_colors = []
        node_sizes = []
        for v in g.nodes():
            if v in S:
                node_colors.append('red')
                node_sizes.append(300)
            else:
                node_colors.append('blue')
                node_sizes.append(50)
    else:
        node_colors = ['blue' for _ in g.nodes()]
        node_sizes = [100 for _ in g.nodes()]
    import matplotlib.pyplot as plt
    nx.draw(g, node_color=node_colors, node_size=node_sizes, pos=pos)
    plt.show()

def greedy(items, budget, f):
    '''
    Generic greedy algorithm to select budget number of items to maximize f.
    '''
    import heapq
    if budget >= len(items):
        S = set(items)
        return S, f(S)
    upper_bounds = [(-f(set([u])), u) for u in items]
    heapq.heapify(upper_bounds)
    starting_objective = f(set())
    S  = set()
    while len(S) < budget:
        val, u = heapq.heappop(upper_bounds)
        new_total = f(S.union(set([u])))
        new_val =  new_total - starting_objective
        if new_val >= -upper_bounds[0][0] - 0.01:
            S.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return S, starting_objective

def sample_live_icm(g, num_graphs):
    import random
    import networkx as nx
    live_edge_graphs = []
    for _ in range(num_graphs):
        h = nx.Graph()
        h.add_nodes_from(g.nodes())
        for u,v in g.edges():
            if random.random() < g[u][v].get('p', 0.01):
                h.add_edge(u,v)
        live_edge_graphs.append(h)
    return live_edge_graphs
