"""
Community-Aware Evolution Algorithm for Fair Influence Maximization (CEA-FIM).

This module implements an evolutionary algorithm for fair influence maximization
in social networks, considering community structure and node attributes.
"""

import networkx as nx
import numpy as np
import pickle
from typing import List, Set, Dict, Callable, Optional, Any, Tuple
from utils import greedy
from icm import sample_live_icm, make_multilinear_objective_samples_group, make_multilinear_gradient_group
from algorithms import algo, maxmin_algo, make_normalized, indicator
import math
import community as community_louvain
import sys
import copy
import random
import time
from time import strftime, localtime
import decimal
from decimal import Decimal

# Configuration constants
NUM_MC_SIMULATIONS = 1000  # Number of Monte-Carlo simulations for ICM
DEFAULT_POP_SIZE = 10      # Default population size
DEFAULT_MUTATION_RATE = 0.1  # Default mutation rate
DEFAULT_CROSSOVER_RATE = 0.6  # Default crossover rate
DEFAULT_MAX_GENERATIONS = 150  # Default number of generations

def multi_to_set(f: Callable, n: Optional[int] = None) -> Callable[[Set[int]], Any]:
    """Convert a function expecting a 0/1 vector to one expecting a set of nodes.
    
    Args:
        f: Function that expects a 0/1 vector
        n: Length of indicator vector (optional)
        
    Returns:
        A new function that takes a set and converts it to indicator vector
    """
    if n is None:
        n = len(g)
    def f_set(S: Set[int]) -> Any:
        return f(indicator(S, n))
    return f_set

def valoracle_to_single(f: Callable, i: int) -> Callable[[np.ndarray], float]:
    """Create a single-output value oracle from multi-output function.
    
    Args:
        f: Multi-output oracle function
        i: Index of desired output
        
    Returns:
        Single-output oracle function
    """
    def f_single(x: np.ndarray) -> float:
        return f(x, NUM_MC_SIMULATIONS)[i]
    return f_single

# Create the first random generation of population
def pop_init(pop, budget, comm, values, comm_label,nodes_attr,prank):
    '''
    pop:	Number of individuals in population (e.g. 30)
    budget:	How many seed nodes per individual (e.g. 5 or 10)
    comm:	List of communities, where each comm[t] = list of node IDs in that community
    (example : comm = [
        [0, 1],       # community 0: nodes 0 and 1
        [2, 3],       # community 1: nodes 2 and 3
        [4, 5, 6]     # community 2: nodes 4,5,6
    ])
    values:	List of possible attribute values (e.g. If attribute = gender, then {male, female})
    comm_label:	For each community t, set of which attributes appear there 
    ( example : comm_label = [
        ['Male', 'Female'],
        ['Male'],
        ['Female', 'Male']
    ])
    nodes_attr:	For each attribute value, which nodes have that value
    (example : nodes_attr = {
        'Male':   [0, 2, 3, 6],
        'Female': [1, 4, 5]
    })
    prank:	PageRank score per node (importance score from the graph)
    '''
    P = []
    # Create pop number of individual seed seets 
    for _ in range(pop):
        P_it1 = [] # list of candidate seed nodes

        # Initialize attribute control variables
        
        comm_score = {} # The score of each community (used to select communities)
        u = {} # A weight controlling how “desirable” each attribute value currently is (used to balance attributes)
        selected_attr = {} # How many nodes of each attribute have been selected so far

        # All attribute values are equally important at the start.
        for cal in values:
            u[cal] = 1
            selected_attr[cal] = 0

        # Compute initial community scores
        for t in range(len(comm)):
            sco1 = len(comm[t]) # size of community
            sco2 = 0

            for ca in comm_label[t]: # which attribute groups appear in this community
                sco2 += u[ca]

            # Bigger communities → higher sco1
            # Communities that contain more attribute groups → higher sco2
            comm_score[t] = sco1 * sco2

        comm_sel = {}

        # Select communities probabilistically
        for _ in range(budget):
            a = list(comm_score.keys()) # community id
            b = list(comm_score.values())   #score

            # normalize b (convert scores into probabilities)
            b_sum = sum(b)
            for deg in range(len(b)):
                b[deg] /= b_sum
            b = np.array(b)
            # choose one community at random such that communities with higher score → higher chance to be picked.
            tar_comm = np.random.choice(a, size=1, p=b.ravel())[0]

            # Record how many nodes to pick from each community
            if tar_comm in list(comm_sel.keys()):
                comm_sel[tar_comm] += 1 # counter is picked from each community
            else:
                comm_sel[tar_comm] = 1
                # Weight of an attribute group is only decreased for first time
                for att in comm_label[tar_comm]:
                    selected_attr[att] += len(set(nodes_attr[att])&set(comm[tar_comm])) # counts how many nodes of that attribute group exist in this community.
                    # If we already picked many nodes with attribute att,
                    # u[att] becomes smaller (because exp(-x) decreases).
                    # So next time, we’ll prefer other attributes — ensuring attribute balance.
                    # decaying factor : Nodes Within community having the attribute value / Total nodes in graphs having the attribute value
                    u[att] = math.exp(-1*selected_attr[att]/len(nodes_attr[att])) 

            # Recompute community scores
            for t in range(len(comm)):
                sco1 = len(comm[t])
                sco2 = 0

                for ca in comm_label[t]:
                    sco2 += u[ca]

                comm_score[t] = sco1 * sco2
                
        # Select top nodes within chosen communities
        for cn in list(comm_sel.keys()):
            pr = {}
            # For each selected community, we get PageRank scores of its nodes.
            for nod in comm[cn]:
                pr[nod] = prank[nod]

            pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            # Pick number of nodes equal to comm_sel of the community
            for pr_ind in range(comm_sel[cn]):
                P_it1.append(pr[pr_ind][0])

        P.append(P_it1)

    return P

# swaps nodes between paired seed sets, repairs any seed set that lost nodes
def crossover(P1, cr, budget, partition, comm_label, comm, values, nodes_attr, prank):
    '''
    P1 — list of seed sets; each seed set is a list of node IDs.
    Example: P1 = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    cr — crossover probability per nodes.
    budget — desired number of nodes per individual.
    partition — dict: node_id → community_id.
    comm — list of communities, each a list of node IDs.
    Example: comm = [[1,2], [3,4], [5,6,7]]
    comm_label — for each community id, list of attribute values present there (unique values).
    Example: comm_label = [['M'], ['F'], ['M','F']]
    values — list of attribute values (categories).
    Example: values = ['M','F']
    nodes_attr — dict: attr_value → list of node IDs with that value.
    Example: nodes_attr = {'M':[1,3,5], 'F':[2,4,6]}
    prank — dict: node_id → pagerank score (float).
    '''
    P = copy.deepcopy(P1)

    # Pairwise gene swapping
    # Pair i with len(P)-i-1 and so on
    for i in range(int(len(P)/2)):
        for j in range(len(P[i])):
            # Swap jth gene of both individuals
            if random.random() < cr:
                temp = P[i][j]
                P[i][j] = P[len(P)-i-1][j]
                P[len(P)-i-1][j] = temp

    # repair & fill missing nodes to remove duplicates or fill fewer than budget
    for i in range(len(P)):
        P[i] = list(set(P[i])) # removes duplicates
        if len(P[i]) == budget:
            continue

        # Prepare community scores
        comm_score = {}
        u = {}
        selected_attr = {}
        for cal in values:
            u[cal] = 1
            selected_attr[cal] = 0

        # Find which communities are already present in the individual
        all_comm = []
        for node in P[i]:
            all_comm.append(partition[node])
        all_comm = list(set(all_comm))

        # Update selected_attr and u based on communities already present
        for ac in all_comm:
            for ca in comm_label[ac]:
                selected_attr[ca] += len(set(nodes_attr[ca]) & set(comm[ac]))
                u[ca] = math.exp(-1 * selected_attr[ca] / len(nodes_attr[ca]))

        # Compute current community scores
        for t in range(len(comm)):
            sco1 = len(comm[t])
            sco2 = 0

            for ca in comm_label[t]:
                sco2 += u[ca]

            comm_score[t] = sco1 * sco2

        # Fill until budget reached
        while len(P[i])<budget:
            a = list(comm_score.keys())  # comm id
            b = list(comm_score.values())  # score

            # normalize b to probabilities
            b_sum = sum(b)
            for deg in range(len(b)):
                b[deg] /= b_sum
            b = np.array(b)
            tar_comm = np.random.choice(a, size=1, p=b.ravel())[0]

            if tar_comm not in all_comm:
                all_comm.append(tar_comm)

                for ca in comm_label[tar_comm]:
                    selected_attr[ca] += len(set(nodes_attr[ca]) & set(comm[tar_comm]))
                    u[ca] = math.exp(-1 * selected_attr[ca] / len(nodes_attr[ca]))

            # Build a PageRank probability distribution
            pr = {}
            for nod in comm[tar_comm]:
                pr[nod] = prank[nod]

            aa = list(pr.keys())
            bb = list(pr.values())

            # Normalize PageRank scores to probabilities.
            bb_sum = sum(bb)
            for deg in range(len(bb)):
                bb[deg] /= bb_sum
            bb = np.array(bb)

            # Pick one node from the chosen community, biased by PageRank scores.
            while True:
                tar_node = np.random.choice(aa, size=1, p=bb.ravel())[0]
                if tar_node not in P[i]:
                    P[i].append(tar_node)
                    break

            # Update community scores after adding a new node
            for t in range(len(comm)):
                sco1 = len(comm[t])
                sco2 = 0

                for ca in comm_label[t]:
                    sco2 += u[ca]

                comm_score[t] = sco1 * sco2

    return P

# mutation randomly changes a few “genes” in individuals so the search doesn’t get stuck in local optima.
def mutation(P1, mu, comm, values,nodes_attr,prank):
    '''
    P1:	    Current population — list of individuals (each individual = list of node IDs)
    Example: P1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mu:	    Mutation probability — chance of replacing a node
    comm:	List of communities, where each comm[t] = list of node IDs in that community
            Example: comm = [
                [0, 1],       # community 0
                [2, 3],       # community 1
                [4, 5, 6]     # community 2
            ]
    values:	List of possible attribute values (e.g. demographic categories)
            Example: values = ['Male', 'Female']
    nodes_attr:	For each attribute value, which nodes have that attribute
            Example: nodes_attr = {
                'Male':   [0, 2, 3, 6],
                'Female': [1, 4, 5]
            }
    prank:	PageRank score per node (importance score from the network)
    partition: Dictionary mapping each node ID → its community ID
    comm_label: For each community t, list of attribute values present in that community
            Example: comm_label = [
                ['Male', 'Female'],
                ['Male'],
                ['Female']
            ]
    '''
    P = copy.deepcopy(P1)

    # For each node within a seedset if the rolled random number is less than mutation probability mu, replace that node
    for i in range(len(P)):
        for j in range(len(P[i])):
            if random.random() < mu:
                # Initialize attribute control variables
                comm_score = {}
                u = {}
                selected_attr = {}
                for cal in values:
                    u[cal] = 1
                    selected_attr[cal] = 0
                
                # Find which communities are already present in the individual
                all_comm = []
                for node in P[i]:
                    all_comm.append(partition[node])
                all_comm.remove(partition[P[i][j]]) # removes the community of the node being replaced
                all_comm = list(set(all_comm))

                # Update attribute coverage scores
                for ac in all_comm:
                    for ca in comm_label[ac]:
                        selected_attr[ca] += len(set(nodes_attr[ca]) & set(comm[ac]))
                        u[ca] = math.exp(-1 * selected_attr[ca] / len(nodes_attr[ca]))

                # Compute community scores
                for t in range(len(comm)):
                    sco1 = len(comm[t])
                    sco2 = 0

                    for ca in comm_label[t]:
                        sco2 += u[ca]

                    comm_score[t] = sco1 * sco2

                a = list(comm_score.keys())  # comm id
                b = list(comm_score.values())  # score

                b_sum = sum(b)
                for deg in range(len(b)):
                    b[deg] /= b_sum
                b = np.array(b)
                # Randomly pick a community to mutate into — biased by community scores
                tar_comm = np.random.choice(a, size=1, p=b.ravel())[0]

                # Inside the chosen community, nodes are weighted by their PageRank scores.
                pr = {}
                for nod in comm[tar_comm]:
                    pr[nod] = prank[nod]

                aa = list(pr.keys())
                bb = list(pr.values())

                bb_sum = sum(bb)
                for deg in range(len(bb)):
                    bb[deg] /= bb_sum
                bb = np.array(bb)

                # Randomly select one node from the target community based on PageRank weights.
                while True:
                    tar_node = np.random.choice(aa, size=1, p=bb.ravel())[0]
                    if tar_node not in P[i]:
                        P[i][j] = tar_node
                        break

    return P


'''
The optimal solution for each group is estimated by running the greedy algorithm separately on each group's subgraph. This gives the highest possible influence spread for that group if you only tried to maximize coverage within it.


'''


succession = True #algorithm will run a separate greedy optimization inside each subgroup (like each region) before running the global algorithm.
solver = 'md'

group_size = {} # store the size of each attribute group
# Example: group_size['twitter']['color'] =
# [[100, 150],   # Run 1: 100 nodes are 'red', 150 nodes are 'blue'
#  [98, 152],    # Run 2: 98 nodes are 'red', 152 nodes are 'blue'
#  [101, 149]]   # Run 3: 101 nodes are 'red', 149 nodes are 'blue'
num_runs = 10 # Number of independent runs per experiment for averaging results.
algorithms = ['Greedy', 'GR', 'MaxMin-Size'] # Algorithms to compare: Greedy, Genetic Algorithm (GR), MaxMin-Size (MMS)

graphnames = ['graph_spa_500_0'] # List of graphs (network files) to run experiments on.
attributes = ['region'] # List of node attributes (demographic categories) to ensure fairness across.
# graphnames = ['rice_subset']
# attributes = ['color']

for graphname in graphnames:
    print(graphname)
    for budget in [40]: # The number of seed nodes to choose for influence maximization
        g = pickle.load(open('networks/{}.pickle'.format(graphname), 'rb'))
        ng = list(g.nodes()) # list of node IDs in the graph Example → [100, 102, 105, ...]
        ngIndex = {} # mapping from node ID to its index in ng Example → {100:0, 102:1, 105:2, ...}
        for ni in range(len(ng)):
            ngIndex[ng[ni]] = ni

        # propagation probability for the ICM
        p = 0.01
        for u, v in g.edges():
            g[u][v]['p'] = p

        g = nx.convert_node_labels_to_integers(g, label_attribute='pid') # Labelling the nodes as integers if not

        group_size[graphname] = {}

        # Counts how many unique attribute values exist.
        for attribute in attributes:
            # assign a unique numeric value for nodes who left the attribute blank
            nvalues = len(np.unique([g.nodes[v][attribute] for v in g.nodes()]))
            group_size[graphname][attribute] = np.zeros((num_runs, nvalues))

        # To record performance metrics
        fair_vals_attr = np.zeros((num_runs, len(attributes))) # average fairness violation for EA.
        greedy_vals_attr = np.zeros((num_runs, len(attributes))) # average fairness violation for greedy.
        pof = np.zeros((num_runs, len(attributes))) # Price of Fairness (ratio of greedy vs fair influence).

        include_total = False

        for attr_idx, attribute in enumerate(attributes):

            # Precomputes 1000 random “live” subgraphs used for Monte Carlo influence simulation.
            live_graphs = sample_live_icm(g, 1000)

            group_indicator = np.ones((len(g.nodes()), 1))

            val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator, list(g.nodes()),
                                                                  list(g.nodes()), np.ones(len(g))) # gives objective function for overall influence of each group
            # f_multi(x) returns total influence spread by seed set x
            def f_multi(x):
                return val_oracle(x, 1000).sum()

            f_set = multi_to_set(f_multi)

            violation_0 = [] # How much EA solution fails to meet fairness constraints
            violation_1 = [] # How much greedy solution fails to meet fairness constraints
            min_fraction_0 = [] # minimum fraction of coverage per group by EA solution
            min_fraction_1 = [] # minimum fraction of coverage per group by greedy solution
            pof_0 = [] # Ratio of total influence by greedy to total influence by EA.
            time_0 = [] # Runtime (in seconds) for the EA in each run.
            time_1 = [] # Runtime (in seconds) for the greedy in each run.

            # alpha is the tradeoff weight in fitness:
            # High α → prioritize coverage
            # Low α → prioritize fairness
            alpha = 0.5  # a*MF+(1-a)*DCV
            print('aplha ', alpha)

            for run in range(num_runs):
                print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
                # find overall optimal solution
                start_time1 = time.perf_counter()
                S, obj = greedy(list(range(len(g))), budget, f_set) # Greedy influence maximization using celf S: selected nodes, obj: achieved influence value
                end_time1 = time.perf_counter()
                runningtime1 = end_time1 - start_time1

                start_time = time.perf_counter()
                # all values taken by this attribute
                values = np.unique([g.nodes[v][attribute] for v in g.nodes()])

                # Maps each attribute value to list of node IDs having that attribute. Example → {'North': [0,1,2], 'South':[3,4], 'West':[5,6,7]}
                nodes_attr = {}  # value-node

                for vidx, val in enumerate(values):
                    nodes_attr[val] = [v for v in g.nodes() if g.nodes[v][attribute] == val]
                    group_size[graphname][attribute][run, vidx] = len(nodes_attr[val])

                # For each group (e.g., region), builds a subgraph, samples influence spread, and finds the optimal seed set within that group using greedy.
                # Calculate best potential influence spread per group by creating temporary subgraphs that only contain nodes from that group and running greedy on them, budget is propertional to group size.
                opt_succession = {}
                if succession:
                    for vidx, val in enumerate(values):
                        h = nx.subgraph(g, nodes_attr[val])
                        h = nx.convert_node_labels_to_integers(h)
                        live_graphs_h = sample_live_icm(h, 1000)
                        group_indicator = np.ones((len(h.nodes()), 1))
                        val_oracle = multi_to_set(valoracle_to_single(
                            make_multilinear_objective_samples_group(live_graphs_h, group_indicator, list(h.nodes()),
                                                                     list(h.nodes()), np.ones(len(h))), 0), len(h))
                        S_succession, opt_succession[val] = greedy(list(h.nodes()),
                                                                   math.ceil(len(nodes_attr[val]) / len(g) * budget),
                                                                   val_oracle)

                if include_total:
                    group_indicator = np.zeros((len(g.nodes()), len(values) + 1))
                    for val_idx, val in enumerate(values):
                        group_indicator[nodes_attr[val], val_idx] = 1
                    group_indicator[:, -1] = 1
                else:
                    group_indicator = np.zeros((len(g.nodes()), len(values)))
                    for val_idx, val in enumerate(values):
                        group_indicator[nodes_attr[val], val_idx] = 1

                # Creates a multi-output oracle giving influence spread for each attribute group separately.
                val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator, list(g.nodes()),
                                                                      list(g.nodes()), np.ones(len(g)))

                # build an objective function for each subgroup
                f_attr = {}
                f_multi_attr = {}
                # Build Per-Group Influence Functions
                for vidx, val in enumerate(values):
                    nodes_attr[val] = [v for v in g.nodes() if g.nodes[v][attribute] == val]
                    f_multi_attr[val] = valoracle_to_single(val_oracle, vidx)
                    f_attr[val] = multi_to_set(f_multi_attr[val])

                # get the best seed set for nodes of each subgroup
                S_attr = {}
                opt_attr = {}
                if not succession:
                    for val in values:
                        S_attr[val], opt_attr[val] = greedy(list(range(len(g))),
                                                            int(len(nodes_attr[val]) / len(g) * budget), f_attr[val])
                if succession:
                    opt_attr = opt_succession
                all_opt = np.array([opt_attr[val] for val in values])

                # Evaluation Fitness Function
                def Eval(SS):
                    # Convert Seed Set to Indices
                    S = [ngIndex[int(i)] for i in SS]
                    fitness = 0
                    # Create a 0/1 vector for the selected seed set
                    x = np.zeros(len(g.nodes))
                    x[list(S)] = 1

                    vals = val_oracle(x, 1000) # influence spread per attribute group
                    coverage_min = (vals / group_size[graphname][attribute][run]).min() # Finds the least-covered group
                    violation = np.clip(all_opt - vals, 0, np.inf) / all_opt # Measures how much each group falls short of its optimal possible influence.

                    # Combine Coverage and Fairness into Fitness
                    fitness += alpha * coverage_min
                    fitness -= (1-alpha) * violation.sum() / len(values)

                    return fitness


                # EA-start
                pop = 10 # population size
                mu = 0.1 # mutation rate
                cr = 0.6 # crossover rate
                maxgen = 150 # 150 iterations

                address = 'networks/{}.txt'.format(graphname)
                G = nx.read_edgelist(address, create_using=nx.Graph())

                partition = community_louvain.best_partition(G) # Uses Louvain algorithm to detect communities.
                
                # Build Community Data
                comm_all_label = list(set(partition.values()))
                comm = []
                for _ in range(len(comm_all_label)):
                    comm.append([])
                for key in list(partition.keys()):
                    comm[partition[key]].append(key)

                # Map Each Community’s Attribute Composition
                comm_label = []
                for c in comm:
                    temp = set()
                    for cc in c:
                        temp.add(g.nodes[ngIndex[int(cc)]][attribute])
                    comm_label.append(list(temp))

                pr = nx.pagerank(G)

                # Initialize Population
                P = pop_init(pop, budget, comm, values,comm_label,nodes_attr,pr)

                i = 0
                while i < maxgen:
                    P = sorted(P, key=lambda x: Eval(x), reverse=True) # sort by fitness

                    P_cr = crossover(P, cr, budget, partition, comm_label, comm, values, nodes_attr, pr)
                    P_mu = mutation(P, mu, comm, values,nodes_attr,pr)

                    # Evaluate children and replace parents if better
                    for index in range(pop):
                        inf1 = Eval(P_mu[index])
                        inf2 = Eval(P[index])

                        if inf1 > inf2:
                            P[index] = P_mu[index]
                    i += 1

                # Takes the best individual from the final generation.
                SS = sorted(P, key=lambda x: Eval(x), reverse=True)[0]
                SI = [ngIndex[int(si)] for si in SS]

                # EA-end

                end_time = time.perf_counter()
                runningtime = end_time - start_time

                xg = np.zeros(len(g.nodes))
                xg[list(S)] = 1

                fair_x = np.zeros(len(g.nodes))
                fair_x[list(SI)] = 1

                greedy_vals = val_oracle(xg, 1000)
                all_fair_vals = val_oracle(fair_x, 1000)

                if include_total:
                    greedy_vals = greedy_vals[:-1]
                    all_fair_vals = all_fair_vals[:-1]

                fair_violation = np.clip(all_opt - all_fair_vals, 0, np.inf) / all_opt
                greedy_violation = np.clip(all_opt - greedy_vals, 0, np.inf) / all_opt
                fair_vals_attr[run, attr_idx] = fair_violation.sum() / len(values)
                greedy_vals_attr[run, attr_idx] = greedy_violation.sum() / len(values)

                greedy_min = (greedy_vals / group_size[graphname][attribute][run]).min()
                fair_min = (all_fair_vals / group_size[graphname][attribute][run]).min()

                pof[run, attr_idx] = greedy_vals.sum() / all_fair_vals.sum()

                # Storing in order for averaging accross runs
                violation_0.append(fair_violation.sum() / len(values))
                violation_1.append(greedy_violation.sum() / len(values))
                min_fraction_0.append(fair_min)
                min_fraction_1.append(greedy_min)
                pof_0.append(greedy_vals.sum() / all_fair_vals.sum())
                time_0.append(runningtime)
                time_1.append(runningtime1)

            print("graph:", graphname, "K:", budget, "attribute", attribute)
            print("F:", Decimal(np.mean(min_fraction_0) - np.mean(violation_0)).quantize(Decimal("0.00"),
                                                                                   rounding=decimal.ROUND_HALF_UP))

            print("violation_EA:",
                  Decimal(np.mean(violation_0)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP),
                  "violation_greedy:",
                  Decimal(np.mean(violation_1)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP))

            print("min_fra_EA:",
                  Decimal(np.mean(min_fraction_0)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP),
                  "min_fra_greedy:",
                  Decimal(np.mean(min_fraction_1)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP))

            print("POF_EA:",
                  Decimal(np.mean(pof_0)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP))

            print("time_EA:", Decimal(np.mean(time_0)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP),
                  "time_greedy:", Decimal(np.mean(time_1)).quantize(Decimal("0.00"), rounding=decimal.ROUND_HALF_UP))
            print()
