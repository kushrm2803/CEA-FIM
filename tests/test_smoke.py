import networkx as nx
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import greedy  # Test a simple core function
from icm import sample_live_icm
from fairness_utils import calculate_beta_for_communities

def test_small_graph_greedy():
    """Smoke test: Create tiny graph, run greedy selection"""
    # Create small test graph
    g = nx.gnp_random_graph(10, 0.3)
    for u, v in g.edges():
        g[u][v]['p'] = 0.1
    
    # Simple objective function for testing
    def f(S):
        return len(S) + len(list(nx.bfs_edges(g, list(S)[0]))) if S else 0
    
    # Run greedy on small graph
    S, val = greedy(list(range(len(g))), budget=2, f=f)
    assert len(S) == 2, "Greedy should select exactly budget=2 nodes"
    assert val > 0, "Objective value should be positive"

def test_icm_samples():
    """Smoke test: Sample ICM graphs"""
    g = nx.gnp_random_graph(5, 0.4)
    for u, v in g.edges():
        g[u][v]['p'] = 0.1
    
    samples = sample_live_icm(g, num_graphs=10)
    assert len(samples) == 10
    assert all(isinstance(h, nx.Graph) for h in samples)

def test_fairness_utils():
    """Smoke test: Basic fairness calculations"""
    # Create tiny test graph with communities
    G = nx.karate_club_graph()
    partition = {n: n % 2 for n in G.nodes()}  # Simple 2-community split
    comm = [[], []]
    for n, c in partition.items():
        comm[c].append(str(n))
    
    g = nx.Graph()  # Dummy graph for attribute testing
    g.add_nodes_from(range(G.number_of_nodes()))
    nx.set_node_attributes(g, {n: 'A' if n % 2 == 0 else 'B' for n in g.nodes()}, 'color')
    
    ngIndex = {i: i for i in range(G.number_of_nodes())}
    nodes_attr = {'A': [n for n in g.nodes() if g.nodes[n]['color'] == 'A'],
                  'B': [n for n in g.nodes() if g.nodes[n]['color'] == 'B']}
    
    beta = calculate_beta_for_communities(G, partition, comm, g, ngIndex, 'color', nodes_attr)
    assert isinstance(beta, dict)
    assert len(beta) == 2  # Should have scores for both communities
    assert abs(sum(beta.values()) - 1.0) < 1e-6  # Beta values should sum to 1

if __name__ == '__main__':
    print("Running smoke tests...")
    test_small_graph_greedy()
    test_icm_samples()
    test_fairness_utils()
    print("All smoke tests passed!")