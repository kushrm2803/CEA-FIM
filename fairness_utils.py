"""
Fairness Enhancement Utilities for Multi-Attribute Influence Maximization
Contains bridge-node detection and enhanced PageRank calculations
"""

import networkx as nx
import numpy as np
import math


# ============================================================================
# PART 1: BRIDGE NODE DETECTION AND BETA CALCULATION
# ============================================================================

def calculate_participation_coefficient(G, partition, comm):
    """
    Calculate participation coefficient for each node.
    PC measures how evenly a node's connections are distributed across communities.
    
    PC(i) = 1 - Σ(k_is / k_i)^2
    where k_is = number of links from node i to community s
          k_i = total degree of node i
    
    High PC → node connects to many different communities (bridge node)
    Low PC → node's connections concentrated in few communities
    
    Args:
        G: NetworkX graph
        partition: dict mapping node_id → community_id
        comm: list of communities (not used but kept for consistency)
    
    Returns:
        dict: {node_id: participation_coefficient}
    """
    pc = {}
    
    for node in G.nodes():
        node_degree = G.degree(node)
        
        if node_degree == 0:
            pc[node] = 0
            continue
        
        # Count connections to each community
        comm_connections = {}
        for neighbor in G.neighbors(node):
            neighbor_comm = partition[neighbor]
            comm_connections[neighbor_comm] = comm_connections.get(neighbor_comm, 0) + 1
        
        # Calculate participation coefficient
        sum_squared = sum((k_is / node_degree) ** 2 for k_is in comm_connections.values())
        pc[node] = 1 - sum_squared
    
    return pc


def calculate_attribute_diversity_score(node, G, g, ngIndex, attribute, nodes_attr):
    """
    Calculate how many different attribute groups this node connects to.
    Higher diversity → better bridge node for fairness.
    
    Args:
        node: node ID (string format from G)
        G: NetworkX graph (with string node IDs)
        g: Original graph (with integer node IDs)
        ngIndex: mapping from original node ID to index
        attribute: attribute name to check
        nodes_attr: dict of attribute_value → list of nodes
    
    Returns:
        float: diversity score (normalized between 0 and 1)
    """
    try:
        # Get neighbors within 2 hops
        neighbors_1hop = set(G.neighbors(str(node)))
        neighbors_2hop = set()
        for n1 in neighbors_1hop:
            neighbors_2hop.update(G.neighbors(n1))
        
        all_neighbors = neighbors_1hop.union(neighbors_2hop)
        
        # Find unique attribute values in neighborhood
        unique_attrs = set()
        for neighbor_str in all_neighbors:
            try:
                neighbor_idx = ngIndex[int(neighbor_str)]
                attr_val = g.nodes[neighbor_idx][attribute]
                unique_attrs.add(attr_val)
            except (KeyError, ValueError):
                continue
        
        # Normalize by total number of possible attribute values
        total_attr_values = len(nodes_attr.keys())
        diversity_score = len(unique_attrs) / total_attr_values if total_attr_values > 0 else 0
        
        return diversity_score
    except:
        return 0


def calculate_beta_for_communities(G, partition, comm, g, ngIndex, attribute, nodes_attr, gamma=1.0):
    """
    Calculate β (bridge bonus) for each community.
    β is based on average participation coefficient of nodes in that community.
    
    Communities with more bridge nodes → higher β → higher priority for selection
    
    Constraint: Σ β_c = 1 (normalized across all communities)
    
    Args:
        G: NetworkX graph
        partition: dict mapping node_id → community_id
        comm: list of communities
        g: original graph
        ngIndex: mapping from original node ID to index
        attribute: attribute name
        nodes_attr: dict of attribute_value → list of nodes
        gamma: controls the strength of bridge bonus (higher γ → more emphasis on bridges)
    
    Returns:
        dict: {community_id: beta_value}
    """
    # Step 1: Calculate participation coefficients for all nodes
    pc = calculate_participation_coefficient(G, partition, comm)
    
    # Step 2: Calculate average PC for each community (with attribute diversity bonus)
    comm_bridge_score = {}
    
    for c_idx, community_nodes in enumerate(comm):
        total_score = 0
        valid_nodes = 0
        
        for node_str in community_nodes:
            try:
                node_idx = ngIndex[int(node_str)]
                
                # Base score from participation coefficient
                pc_score = pc.get(node_str, 0)
                
                # Bonus from attribute diversity
                diversity_score = calculate_attribute_diversity_score(
                    node_str, G, g, ngIndex, attribute, nodes_attr
                )
                
                # Combined score: PC + attribute diversity bonus
                node_bridge_score = pc_score * (1 + diversity_score)
                
                total_score += node_bridge_score
                valid_nodes += 1
            except (KeyError, ValueError):
                continue
        
        # Average bridge score for this community
        avg_bridge_score = total_score / valid_nodes if valid_nodes > 0 else 0
        comm_bridge_score[c_idx] = avg_bridge_score
    
    # Step 3: Apply exponential transformation to emphasize differences
    beta_unnormalized = {}
    for c_idx, score in comm_bridge_score.items():
        beta_unnormalized[c_idx] = math.exp(gamma * score)
    
    # Step 4: Normalize so Σ β = 1
    total_beta = sum(beta_unnormalized.values())
    beta = {}
    for c_idx, value in beta_unnormalized.items():
        beta[c_idx] = value / total_beta if total_beta > 0 else 1.0 / len(comm)
    
    return beta


def compute_community_score_with_bridge_bonus(comm, comm_label, u, beta, t):
    """
    Enhanced community scoring function:
    
    Score(C_t) = |C_t| × (Σ u[attr]) × (1 + β_t)
    
    Args:
        comm: list of communities
        comm_label: attribute labels present in each community
        u: current attribute desirability weights
        beta: bridge bonuses for each community
        t: community index
    
    Returns:
        float: community score
    """
    # Component 1: Community size
    size_component = len(comm[t])
    
    # Component 2: Attribute desirability sum
    attr_component = sum(u[ca] for ca in comm_label[t])
    
    # Component 3: Bridge bonus
    bridge_component = 1 + beta[t]
    
    # Final score
    score = size_component * attr_component * bridge_component
    
    return score


# ============================================================================
# PART 2: ENHANCED PAGERANK WITH MULTI-ATTRIBUTE AWARENESS
# ============================================================================

def calculate_group_influence_potential(g, nodes_attr, live_graphs, val_oracle, values):
    """
    Calculate the natural spreading power of each attribute group.
    Groups with lower influence potential need higher PageRank compensation.
    
    Args:
        g: NetworkX graph
        nodes_attr: dict of attribute_value → list of nodes
        live_graphs: pre-sampled ICM graphs
        val_oracle: influence oracle function
        values: list of attribute values
    
    Returns:
        dict: {attribute_value: average_influence_spread}
    """
    group_influence = {}
    
    for val in values:
        # Get nodes belonging to this attribute group
        group_nodes = nodes_attr[val]
        
        if len(group_nodes) == 0:
            group_influence[val] = 1.0
            continue
        
        # Sample a few random nodes from this group
        sample_size = min(5, len(group_nodes))
        sampled_nodes = np.random.choice(group_nodes, size=sample_size, replace=False)
        
        total_influence = 0
        for node in sampled_nodes:
            # Create seed set with single node
            x = np.zeros(len(g.nodes()))
            x[node] = 1
            
            # Measure influence spread (use fewer simulations for speed)
            influence = val_oracle(x, 100).sum()
            total_influence += influence
        
        # Average influence per node in this group
        group_influence[val] = total_influence / sample_size
    
    return group_influence


def calculate_attribute_deficit_memory(selected_attr, nodes_attr, values):
    """
    Calculate how much each attribute has been underselected.
    This creates temporal fairness.
    
    Args:
        selected_attr: dict tracking current selection counts
        nodes_attr: dict of attribute_value → list of nodes
        values: list of attribute values
    
    Returns:
        dict: {attribute_value: deficit_score}
    """
    deficit = {}
    
    total_selected = sum(selected_attr.values())
    
    for val in values:
        # Expected fair share
        group_size = len(nodes_attr[val])
        total_nodes = sum(len(nodes_attr[v]) for v in values)
        expected_fraction = group_size / total_nodes if total_nodes > 0 else 0
        
        # Actual fraction selected
        actual_fraction = selected_attr[val] / total_selected if total_selected > 0 else 0
        
        # Deficit (positive if underselected)
        deficit[val] = max(0, expected_fraction - actual_fraction)
    
    return deficit


def calculate_enhanced_personalization_vector(G, g, ngIndex, attribute, nodes_attr, 
                                               values, u, selected_attr=None,
                                               group_influence=None, alpha_size=1.0, 
                                               alpha_influence=0.5, alpha_deficit=0.3):
    """
    Enhanced personalization vector for PageRank that considers:
    1. Inverse group size (fairness by representation)
    2. Inverse influence potential (fairness by structural disadvantage)
    3. Attribute deficit memory (temporal fairness)
    4. Current attribute desirability (u values)
    
    Weight formula:
    w(node) = (1/group_size)^α₁ × (1/√influence_potential)^α₂ × 
              exp(α₃ × deficit) × u[attribute]
    
    Args:
        G, g, ngIndex, attribute, nodes_attr, values: graph and attribute data
        u: current attribute desirability weights
        selected_attr: selection history (optional)
        group_influence: pre-calculated influence potentials (optional)
        alpha_size: weight for size-based fairness (default 1.0)
        alpha_influence: weight for influence-based fairness (default 0.5)
        alpha_deficit: weight for temporal fairness (default 0.3)
    
    Returns:
        dict: normalized personalization vector {node_str: weight}
    """
    personalization = {}
    
    # Get group sizes
    group_sizes = {val: len(nodes) for val, nodes in nodes_attr.items()}
    
    # Calculate attribute deficits if selection history provided
    if selected_attr is not None:
        deficit = calculate_attribute_deficit_memory(selected_attr, nodes_attr, values)
    else:
        deficit = {val: 0 for val in values}
    
    # If influence potential not provided, assume uniform
    if group_influence is None:
        group_influence = {val: 1.0 for val in values}
    
    # Calculate weight for each node
    for node_str in G.nodes():
        try:
            # Map node from G (string) to g (integer) to get attribute
            node_g_idx = ngIndex[int(node_str)]
            attr_val = g.nodes[node_g_idx][attribute]
            
            # Component 1: Inverse group size (smaller groups get boost)
            size_component = (1.0 / group_sizes[attr_val]) ** alpha_size if group_sizes[attr_val] > 0 else 0
            
            # Component 2: Inverse influence potential (poorly connected groups get boost)
            influence_component = (1.0 / math.sqrt(group_influence[attr_val])) ** alpha_influence
            
            # Component 3: Exponential deficit (underselected groups get boost)
            deficit_component = math.exp(alpha_deficit * deficit[attr_val])
            
            # Component 4: Current desirability from EA (u values)
            desirability_component = u.get(attr_val, 1.0)
            
            # Combined weight
            personalization[node_str] = (size_component * 
                                        influence_component * 
                                        deficit_component * 
                                        desirability_component)
        
        except (KeyError, ValueError):
            # If node not found or error, give minimal weight
            personalization[node_str] = 0.001
    
    # Normalize so sum = 1 (required by networkx)
    total_weight = sum(personalization.values())
    if total_weight > 0:
        personalization = {node: weight / total_weight 
                          for node, weight in personalization.items()}
    else:
        # Fallback to uniform if all weights are 0
        uniform_weight = 1.0 / len(personalization)
        personalization = {node: uniform_weight for node in personalization.keys()}
    
    return personalization


def calculate_bridge_aware_pagerank(G, g, ngIndex, attribute, nodes_attr, values, 
                                    partition, comm, u, selected_attr=None,
                                    group_influence=None, alpha_size=1.0, 
                                    alpha_influence=0.5, alpha_deficit=0.3,
                                    alpha_bridge=0.2, damping=0.85):
    """
    Complete enhanced PageRank calculation with:
    - Multi-attribute personalization
    - Bridge node awareness
    - Temporal fairness
    - Structural fairness
    
    Args:
        G, g, ngIndex, attribute, nodes_attr, values: graph and attribute data
        partition: node → community mapping
        comm: list of communities
        u: attribute desirability weights
        selected_attr: selection history (optional)
        group_influence: pre-calculated influence potentials (optional)
        alpha_size, alpha_influence, alpha_deficit: fairness weights
        alpha_bridge: weight for bridge node bonus
        damping: PageRank damping factor
    
    Returns:
        dict: {node_str: pagerank_score}
    """
    # Step 1: Calculate base personalization from attributes
    personalization = calculate_enhanced_personalization_vector(
        G, g, ngIndex, attribute, nodes_attr, values, u, 
        selected_attr, group_influence, alpha_size, alpha_influence, alpha_deficit
    )
    
    # Step 2: Calculate bridge scores
    pc = calculate_participation_coefficient(G, partition, comm)
    
    # Step 3: Enhance personalization with bridge awareness
    enhanced_personalization = {}
    for node_str in personalization.keys():
        base_weight = personalization[node_str]
        
        # Bridge bonus from participation coefficient
        bridge_score = pc.get(node_str, 0)
        
        # Calculate attribute diversity
        try:
            diversity_score = calculate_attribute_diversity_score(
                node_str, G, g, ngIndex, attribute, nodes_attr
            )
        except:
            diversity_score = 0
        
        # Combined bridge bonus
        bridge_bonus = bridge_score * (1 + diversity_score)
        
        # Final weight with bridge awareness
        enhanced_personalization[node_str] = base_weight * (1 + alpha_bridge * bridge_bonus)
    
    # Renormalize
    total = sum(enhanced_personalization.values())
    if total > 0:
        enhanced_personalization = {node: w / total 
                                   for node, w in enhanced_personalization.items()}
    
    # Step 4: Compute PageRank with enhanced personalization
    pr = nx.pagerank(G, alpha=damping, personalization=enhanced_personalization)
    
    return pr