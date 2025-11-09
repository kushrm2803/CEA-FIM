"""
Fairness Enhancement Utilities for Multi-Attribute Influence Maximization
"""
import networkx as nx
import numpy as np
import math

def calculate_participation_coefficient(G, partition, comm):
    pc = {}
    for node in G.nodes():
        node_degree = G.degree(node)
        if node_degree == 0:
            pc[node] = 0
            continue
        comm_connections = {}
        for neighbor in G.neighbors(node):
            neighbor_comm = partition[neighbor]
            comm_connections[neighbor_comm] = comm_connections.get(neighbor_comm, 0) + 1
        sum_squared = sum((k_is / node_degree) ** 2 for k_is in comm_connections.values())
        pc[node] = 1 - sum_squared
    return pc

def calculate_attribute_diversity_score(node, G, g, ngIndex, attribute, nodes_attr):
    try:
        neighbors_1hop = set(G.neighbors(str(node)))
        neighbors_2hop = set()
        for n1 in neighbors_1hop:
            neighbors_2hop.update(G.neighbors(n1))
        all_neighbors = neighbors_1hop.union(neighbors_2hop)
        unique_attrs = set()
        for neighbor_str in all_neighbors:
            try:
                neighbor_idx = ngIndex[int(neighbor_str)]
                attr_val = g.nodes[neighbor_idx][attribute]
                unique_attrs.add(attr_val)
            except (KeyError, ValueError):
                continue
        total_attr_values = len(nodes_attr.keys())
        diversity_score = len(unique_attrs) / total_attr_values if total_attr_values > 0 else 0
        return diversity_score
    except:
        return 0

def calculate_beta_for_communities(G, partition, comm, g, ngIndex, attribute, nodes_attr, gamma=1.0):
    pc = calculate_participation_coefficient(G, partition, comm)
    comm_bridge_score = {}
    for c_idx, community_nodes in enumerate(comm):
        total_score = 0
        valid_nodes = 0
        for node_str in community_nodes:
            try:
                node_idx = ngIndex[int(node_str)]
                pc_score = pc.get(node_str, 0)
                diversity_score = calculate_attribute_diversity_score(node_str, G, g, ngIndex, attribute, nodes_attr)
                node_bridge_score = pc_score * (1 + diversity_score)
                total_score += node_bridge_score
                valid_nodes += 1
            except (KeyError, ValueError):
                continue
        avg_bridge_score = total_score / valid_nodes if valid_nodes > 0 else 0
        comm_bridge_score[c_idx] = avg_bridge_score
    beta_unnormalized = {c_idx: math.exp(gamma * score) for c_idx, score in comm_bridge_score.items()}
    total_beta = sum(beta_unnormalized.values())
    beta = {c_idx: value / total_beta if total_beta > 0 else 1.0 / len(comm) for c_idx, value in beta_unnormalized.items()}
    return beta

def compute_community_score_with_bridge_bonus(comm, comm_label, u, beta, t):
    size_component = len(comm[t])
    attr_component = sum(u[ca] for ca in comm_label[t])
    bridge_component = 1 + beta[t]
    score = size_component * attr_component * bridge_component
    return score

def calculate_group_influence_potential(g, nodes_attr, live_graphs, val_oracle, values):
    group_influence = {}
    for val in values:
        group_nodes = nodes_attr[val]
        if len(group_nodes) == 0:
            group_influence[val] = 1.0
            continue
        sample_size = min(5, len(group_nodes))
        sampled_nodes = np.random.choice(group_nodes, size=sample_size, replace=False)
        total_influence = 0
        for node in sampled_nodes:
            x = np.zeros(len(g.nodes()))
            x[node] = 1
            influence = val_oracle(x, 100).sum()
            total_influence += influence
        group_influence[val] = total_influence / sample_size
    return group_influence

def calculate_bridge_aware_pagerank(G, g, ngIndex, attribute, nodes_attr, values,
                                    partition=None, comm=None, damping=0.85):
    """
    Simple bridge-aware PageRank: compute standard PageRank and boost scores
    of nodes with high participation coefficient.
    Returns dict mapping node -> score.
    """
    # base pagerank
    pagerank = nx.pagerank(G, alpha=damping)
    if partition is None or comm is None:
        # no community info: return pagerank
        return pagerank
    pc = calculate_participation_coefficient(G, partition, comm)
    # boost
    bridge_scores = {node: pagerank.get(node, 0) * (1 + pc.get(node, 0)) for node in G.nodes()}
    # normalize
    total = sum(bridge_scores.values())
    if total > 0:
        bridge_scores = {k: v/total for k, v in bridge_scores.items()}
    return bridge_scores
