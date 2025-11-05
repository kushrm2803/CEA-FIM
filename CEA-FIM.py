import networkx as nx
import numpy as np
import pickle
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
from fairness_utils import (
    calculate_beta_for_communities,
    compute_community_score_with_bridge_bonus,
    calculate_bridge_aware_pagerank,
    calculate_group_influence_potential
)

# Individual = one seed set (e.g., [3, 8, 15, 42, 50])
# Gene = one node in that seed set
# Population = collection of several such seed sets (solutions)

def multi_to_set(f, n = None):
    '''
    Input:  f (function that expects a 0/1 vector)
    Output: A new function that expects a set of nodes
    Inside: It converts the set into a 0/1 vector using indicator(S, n) and calls the original f
    Purpose: A wrapper function to pass 0/1 vector to a function that expects a set
    '''
    if n == None:
        n = len(g)
    def f_set(S):
        return f(indicator(S, n))
    return f_set

def valoracle_to_single(f, i):
    '''
    Input:  f (the multi-output function), i (index of desired output)
    Output: A new function that gives i-th value of f(x,1000)
    Purpose: To create one oracle per attribute group for fairness-based evaluation
    '''
    def f_single(x):
        return f(x, 1000)[i] #1000 is the number of Monte-Carlo simulations for ICM
    return f_single

# Create the first random generation of population
def pop_init(pop, budget, comm, values, comm_label, nodes_attr, prank, 
             G, partition, g, ngIndex, attribute, gamma=1.0):
    """
    Enhanced population initialization with bridge-node bonus.
    
    New parameters:
        G: NetworkX graph for beta calculation
        partition: node → community mapping
        g: original graph
        ngIndex: node ID to index mapping
        attribute: attribute name
        gamma: controls strength of bridge bonus (default 1.0)
    """
    P = []
    
    # Calculate β for all communities ONCE
    beta = calculate_beta_for_communities(G, partition, comm, g, ngIndex, 
                                          attribute, nodes_attr, gamma)
    
    for _ in range(pop):
        P_it1 = []
        
        # Initialize attribute control variables
        comm_score = {}
        u = {}
        selected_attr = {}
        
        for cal in values:
            u[cal] = 1
            selected_attr[cal] = 0
        
        # Compute initial community scores WITH BRIDGE BONUS
        for t in range(len(comm)):
            comm_score[t] = compute_community_score_with_bridge_bonus(
                comm, comm_label, u, beta, t
            )
        
        comm_sel = {}
        
        # Select communities probabilistically
        for _ in range(budget):
            a = list(comm_score.keys())
            b = list(comm_score.values())
            
            b_sum = sum(b)
            for deg in range(len(b)):
                b[deg] /= b_sum
            b = np.array(b)
            
            tar_comm = np.random.choice(a, size=1, p=b.ravel())[0]
            
            if tar_comm in list(comm_sel.keys()):
                comm_sel[tar_comm] += 1
            else:
                comm_sel[tar_comm] = 1
                for att in comm_label[tar_comm]:
                    selected_attr[att] += len(set(nodes_attr[att]) & set(comm[tar_comm]))
                    u[att] = math.exp(-1 * selected_attr[att] / len(nodes_attr[att]))
            
            # Recompute community scores with updated u
            for t in range(len(comm)):
                comm_score[t] = compute_community_score_with_bridge_bonus(
                    comm, comm_label, u, beta, t
                )
        
        # Select top nodes within chosen communities
        for cn in list(comm_sel.keys()):
            pr = {}
            for nod in comm[cn]:
                pr[nod] = prank[nod]
            
            pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            for pr_ind in range(comm_sel[cn]):
                P_it1.append(pr[pr_ind][0])
        
        P.append(P_it1)
    
    return P

# swaps nodes between paired seed sets, repairs any seed set that lost nodes
def crossover(P1, cr, budget, partition, comm_label, comm, values, nodes_attr, prank,
              G, g, ngIndex, attribute, gamma=1.0):  # ADD NEW PARAMETERS
    '''
    Enhanced crossover with bridge-aware community scoring
    '''
    P = copy.deepcopy(P1)
    
    # Calculate beta ONCE for this crossover operation
    beta = calculate_beta_for_communities(G, partition, comm, g, ngIndex, 
                                          attribute, nodes_attr, gamma)

    # Pairwise gene swapping
    for i in range(int(len(P)/2)):
        for j in range(len(P[i])):
            if random.random() < cr:
                temp = P[i][j]
                P[i][j] = P[len(P)-i-1][j]
                P[len(P)-i-1][j] = temp

    # repair & fill missing nodes
    for i in range(len(P)):
        P[i] = list(set(P[i]))
        if len(P[i]) == budget:
            continue

        comm_score = {}
        u = {}
        selected_attr = {}
        for cal in values:
            u[cal] = 1
            selected_attr[cal] = 0

        all_comm = []
        for node in P[i]:
            all_comm.append(partition[node])
        all_comm = list(set(all_comm))

        for ac in all_comm:
            for ca in comm_label[ac]:
                selected_attr[ca] += len(set(nodes_attr[ca]) & set(comm[ac]))
                u[ca] = math.exp(-1 * selected_attr[ca] / len(nodes_attr[ca]))

        # UPDATED: Use bridge-aware scoring
        for t in range(len(comm)):
            comm_score[t] = compute_community_score_with_bridge_bonus(
                comm, comm_label, u, beta, t
            )

        while len(P[i])<budget:
            a = list(comm_score.keys())
            b = list(comm_score.values())

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

            pr = {}
            for nod in comm[tar_comm]:
                pr[nod] = prank[nod]

            aa = list(pr.keys())
            bb = list(pr.values())

            bb_sum = sum(bb)
            for deg in range(len(bb)):
                bb[deg] /= bb_sum
            bb = np.array(bb)

            while True:
                tar_node = np.random.choice(aa, size=1, p=bb.ravel())[0]
                if tar_node not in P[i]:
                    P[i].append(tar_node)
                    break

            # UPDATED: Recompute with bridge-aware scoring
            for t in range(len(comm)):
                comm_score[t] = compute_community_score_with_bridge_bonus(
                    comm, comm_label, u, beta, t
                )

    return P


# mutation randomly changes a few “genes” in individuals so the search doesn’t get stuck in local optima.
def mutation(P1, mu, comm, values, nodes_attr, prank, partition, comm_label,
             G, g, ngIndex, attribute, gamma=1.0):  # ADD NEW PARAMETERS
    '''
    Enhanced mutation with bridge-aware community scoring
    '''
    P = copy.deepcopy(P1)
    
    # Calculate beta ONCE for this mutation operation
    beta = calculate_beta_for_communities(G, partition, comm, g, ngIndex, 
                                          attribute, nodes_attr, gamma)

    for i in range(len(P)):
        for j in range(len(P[i])):
            if random.random() < mu:
                comm_score = {}
                u = {}
                selected_attr = {}
                for cal in values:
                    u[cal] = 1
                    selected_attr[cal] = 0
                
                all_comm = []
                for node in P[i]:
                    all_comm.append(partition[node])
                all_comm.remove(partition[P[i][j]])
                all_comm = list(set(all_comm))

                for ac in all_comm:
                    for ca in comm_label[ac]:
                        selected_attr[ca] += len(set(nodes_attr[ca]) & set(comm[ac]))
                        u[ca] = math.exp(-1 * selected_attr[ca] / len(nodes_attr[ca]))

                # UPDATED: Use bridge-aware scoring
                for t in range(len(comm)):
                    comm_score[t] = compute_community_score_with_bridge_bonus(
                        comm, comm_label, u, beta, t
                    )

                a = list(comm_score.keys())
                b = list(comm_score.values())

                b_sum = sum(b)
                for deg in range(len(b)):
                    b[deg] /= b_sum
                b = np.array(b)
                tar_comm = np.random.choice(a, size=1, p=b.ravel())[0]

                pr = {}
                for nod in comm[tar_comm]:
                    pr[nod] = prank[nod]

                aa = list(pr.keys())
                bb = list(pr.values())

                bb_sum = sum(bb)
                for deg in range(len(bb)):
                    bb[deg] /= bb_sum
                bb = np.array(bb)

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

graphnames = ['twitter'] # List of graphs (network files) to run experiments on.
attributes = ['color'] # List of node attributes (demographic categories) to ensure fairness across.
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

        g = nx.convert_node_labels_to_integers(g, label_attribute='pid')

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
                                                                  list(g.nodes()), np.ones(len(g))) # objective function for overall influence
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
                S, obj = greedy(list(range(len(g))), budget, f_set) # Greedy influence maximization S: selected nodes, obj: achieved influence value
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

                # --- Start of Fairness-Biased Personalized PageRank Calculation ---
                # 1. Get the size of each attribute group from the 'nodes_attr' dictionary.
                group_sizes = {val: len(nodes) for val, nodes in nodes_attr.items()}

                # 2. Create the un-normalized personalization vector.
                # Each node's weight is the inverse of its group's size
                # --- End of Fairness-Biased Personalized PageRank Calculation ---


                print("Calculating group influence potentials...")
                group_influence = calculate_group_influence_potential(
                    g, nodes_attr, live_graphs, val_oracle, values
                )

                # Calculate enhanced PageRank with all improvements
                pr = calculate_bridge_aware_pagerank(
                    G, g, ngIndex, attribute, nodes_attr, values, 
                    partition, comm, 
                    u={val: 1.0 for val in values},  # Initial u values
                    selected_attr=None,  # No selection history yet
                    group_influence=group_influence,
                    alpha_size=1.0,        # Weight for size-based fairness
                    alpha_influence=0.5,   # Weight for structural fairness
                    alpha_deficit=0.3,     # Weight for temporal fairness
                    alpha_bridge=0.2,      # Weight for bridge node bonus
                    damping=0.85
                )
                # Initialize Population
                P = pop_init(pop, budget, comm, values, comm_label, nodes_attr, pr, G, partition, g, ngIndex, attribute, gamma=1.0)
                i = 0
                while i < maxgen:
                    P = sorted(P, key=lambda x: Eval(x), reverse=True) # sort by fitness

                    ##### BUG : Crossover never actually used #####
                    P_cr = crossover(P, cr, budget, partition, comm_label, comm, values, nodes_attr, pr,G, g, ngIndex, attribute, gamma=1.0)
                    P_mu = mutation(P, mu, comm, values ,nodes_attr, pr, partition, comm_label,G, g, ngIndex, attribute, gamma=1.0)

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