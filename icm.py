import numpy as np
import random
from numba import jit
import networkx as nx


def sample_live_icm(g, num_graphs):
    '''
    Returns num_graphs live edge graphs sampled from the ICM on g. Assumes that
    each edge has a propagation probability accessible via g[u][v]['p'].
    '''
    import networkx as nx
    live_edge_graphs = []
    for _ in range(num_graphs):
        h = nx.Graph()
        h.add_nodes_from(g.nodes())
        for u,v in g.edges():
            if random.random() < g[u][v]['p']:
                h.add_edge(u,v)
        live_edge_graphs.append(h)
    return live_edge_graphs

def f_all_influmax_multlinear(x, Gs, Ps, ws):
    n = len(Gs)
    sample_weights = 1./n * np.ones(n)
    return objective_live_edge(x, Gs, Ps, ws, sample_weights)

def make_multilinear_objective_samples(live_graphs, target_nodes, selectable_nodes, p_attend):
    Gs, Ps, ws = live_edge_to_adjlist(live_graphs, target_nodes, p_attend)
    def f_all(x):
        x_expand = np.zeros(len(live_graphs[0]))
        x_expand[selectable_nodes] = x
        return f_all_influmax_multlinear(x_expand, Gs, Ps, ws)
    return f_all


def make_multilinear_objective_samples_group(live_graphs, group_indicator, target_nodes, selectable_nodes, p_attend):
    """
    Simple sampler-based multilinear objective that returns influence per group.
    This implementation uses the provided sampled live graphs and computes
    reachability from a deterministic seed set x on each sample.
    """
    n_groups = group_indicator.shape[1]
    n_nodes = len(live_graphs[0].nodes())

    def f_all(x, batch_size=100):
        # x is assumed to be full-length indicator vector for all nodes
        results = np.zeros(n_groups)
        seeds = [i for i, v in enumerate(x) if v > 0]
        if len(seeds) == 0:
            return results
        for _ in range(batch_size):
            g_sample = random.choice(live_graphs)
            influenced = set()
            # BFS from each seed in the sampled live-edge graph
            for s in seeds:
                if s in g_sample:
                    # connected component reachable from seed
                    for comp in nx.connected_components(g_sample):
                        if s in comp:
                            influenced.update(comp)
                            break
            # count per group
            for j in range(n_groups):
                # group_indicator maps node indices -> group membership
                # group_indicator is assumed to index by node id
                group_nodes = [idx for idx in range(n_nodes) if group_indicator[idx, j]]
                results[j] += len(set(group_nodes).intersection(influenced))
        results = results / float(batch_size)
        return results

    return f_all

# (rest of functions omitted for brevity in this patch - original file contains full implementations)
