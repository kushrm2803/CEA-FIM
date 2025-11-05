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

def multi_to_set(f, n = None):
    '''
    Takes as input a function defined on indicator vectors of sets, and returns
    a version of the function which directly accepts sets
    '''
    if n == None:
        n = len(g)
    def f_set(S):
        return f(indicator(S, n))
    return f_set

def valoracle_to_single(f, i):
    def f_single(x):
        return f(x, 1000)[i]
    return f_single

def pop_init(pop, budget, comm, values, comm_label,nodes_attr,prank):
    P = []

    for _ in range(pop):
        P_it1 = []

        comm_score = {}
        u = {}
        selected_attr = {}

        for cal in values:
            u[cal] = 1
            selected_attr[cal] = 0

        for t in range(len(comm)):
            sco1 = len(comm[t])
            sco2 = 0

            for ca in comm_label[t]:
                sco2 += u[ca]

            comm_score[t] = sco1 * sco2

        comm_sel = {}

        for _ in range(budget):
            a = list(comm_score.keys())#comm number
            b = list(comm_score.values())#score

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
                    selected_attr[att] += len(set(nodes_attr[att])&set(comm[tar_comm]))
                    u[att] = math.exp(-1*selected_attr[att]/len(nodes_attr[att]))

            for t in range(len(comm)):
                sco1 = len(comm[t])
                sco2 = 0

                for ca in comm_label[t]:
                    sco2 += u[ca]

                comm_score[t] = sco1 * sco2

        for cn in list(comm_sel.keys()):
            pr = {}
            for nod in comm[cn]:
                pr[nod] = prank[nod]

            pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            for pr_ind in range(comm_sel[cn]):
                P_it1.append(pr[pr_ind][0])

        P.append(P_it1)

    return P

def crossover(P1, cr, budget, partition, comm_label, comm, values, nodes_attr, prank):
    P = copy.deepcopy(P1)

    for i in range(int(len(P)/2)):
        for j in range(len(P[i])):
            if random.random() < cr:
                temp = P[i][j]
                P[i][j] = P[len(P)-i-1][j]
                P[len(P)-i-1][j] = temp

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

        for t in range(len(comm)):
            sco1 = len(comm[t])
            sco2 = 0

            for ca in comm_label[t]:
                sco2 += u[ca]

            comm_score[t] = sco1 * sco2

        while len(P[i])<budget:
            a = list(comm_score.keys())  # comm number
            b = list(comm_score.values())  # score

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

            for t in range(len(comm)):
                sco1 = len(comm[t])
                sco2 = 0

                for ca in comm_label[t]:
                    sco2 += u[ca]

                comm_score[t] = sco1 * sco2

    return P

def mutation(P1, mu, comm, values,nodes_attr,prank):
    P = copy.deepcopy(P1)

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

                for t in range(len(comm)):
                    sco1 = len(comm[t])
                    sco2 = 0

                    for ca in comm_label[t]:
                        sco2 += u[ca]

                    comm_score[t] = sco1 * sco2

                a = list(comm_score.keys())  # comm number
                b = list(comm_score.values())  # score

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


succession = True
solver = 'md'

group_size = {}
num_runs = 10
algorithms = ['Greedy', 'GR', 'MaxMin-Size']

# graphnames = ['graph_spa_500_0']
# attributes = ['region', 'ethnicity', 'age', 'gender', 'status']
graphnames = ['twitter']
attributes = ['color']

for graphname in graphnames:
    print(graphname)
    for budget in [40]:
        g = pickle.load(open('networks/{}.pickle'.format(graphname), 'rb'))
        ng = list(g.nodes())
        ngIndex = {}
        for ni in range(len(ng)):
            ngIndex[ng[ni]] = ni

        # propagation probability for the ICM
        p = 0.01
        for u, v in g.edges():
            g[u][v]['p'] = p

        g = nx.convert_node_labels_to_integers(g, label_attribute='pid')

        group_size[graphname] = {}

        for attribute in attributes:
            # assign a unique numeric value for nodes who left the attribute blank
            nvalues = len(np.unique([g.nodes[v][attribute] for v in g.nodes()]))
            group_size[graphname][attribute] = np.zeros((num_runs, nvalues))

        fair_vals_attr = np.zeros((num_runs, len(attributes)))
        greedy_vals_attr = np.zeros((num_runs, len(attributes)))
        pof = np.zeros((num_runs, len(attributes)))

        include_total = False

        for attr_idx, attribute in enumerate(attributes):

            live_graphs = sample_live_icm(g, 1000)

            group_indicator = np.ones((len(g.nodes()), 1))

            val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator, list(g.nodes()),
                                                                  list(g.nodes()), np.ones(len(g)))

            def f_multi(x):
                return val_oracle(x, 1000).sum()

            f_set = multi_to_set(f_multi)

            violation_0 = []
            violation_1 = []
            min_fraction_0 = []
            min_fraction_1 = []
            pof_0 = []
            time_0 = []
            time_1 = []

            alpha = 0.5  # a*MF+(1-a)*DCV
            print('aplha ', alpha)

            for run in range(num_runs):
                print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
                # find overall optimal solution
                start_time1 = time.perf_counter()
                S, obj = greedy(list(range(len(g))), budget, f_set)
                end_time1 = time.perf_counter()
                runningtime1 = end_time1 - start_time1

                start_time = time.perf_counter()
                # all values taken by this attribute
                values = np.unique([g.nodes[v][attribute] for v in g.nodes()])

                nodes_attr = {}  # value-node

                for vidx, val in enumerate(values):
                    nodes_attr[val] = [v for v in g.nodes() if g.nodes[v][attribute] == val]
                    group_size[graphname][attribute][run, vidx] = len(nodes_attr[val])

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

                val_oracle = make_multilinear_objective_samples_group(live_graphs, group_indicator, list(g.nodes()),
                                                                      list(g.nodes()), np.ones(len(g)))

                # build an objective function for each subgroup
                f_attr = {}
                f_multi_attr = {}
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


                def Eval(SS):
                    S = [ngIndex[int(i)] for i in SS]
                    fitness = 0
                    x = np.zeros(len(g.nodes))
                    x[list(S)] = 1

                    vals = val_oracle(x, 1000)
                    coverage_min = (vals / group_size[graphname][attribute][run]).min()
                    violation = np.clip(all_opt - vals, 0, np.inf) / all_opt

                    fitness += alpha * coverage_min
                    fitness -= (1-alpha) * violation.sum() / len(values)

                    return fitness


                # EA-start
                pop = 10
                mu = 0.1
                cr = 0.6
                maxgen = 150

                address = 'networks/{}.txt'.format(graphname)
                G = nx.read_edgelist(address, create_using=nx.Graph())

                partition = community_louvain.best_partition(G)
                comm_all_label = list(set(partition.values()))#社团标签，非节点
                comm = []
                for _ in range(len(comm_all_label)):
                    comm.append([])
                for key in list(partition.keys()):
                    comm[partition[key]].append(key)

                comm_label = []#每个社团含有的节点属性
                for c in comm:
                    temp = set()
                    for cc in c:
                        temp.add(g.nodes[ngIndex[int(cc)]][attribute])
                    comm_label.append(list(temp))

                pr = nx.pagerank(G)

                P = pop_init(pop, budget, comm, values,comm_label,nodes_attr,pr)

                i = 0
                while i < maxgen:
                    P = sorted(P, key=lambda x: Eval(x), reverse=True)

                    P_cr = crossover(P, cr, budget, partition, comm_label, comm, values, nodes_attr, pr)
                    P_mu = mutation(P, mu, comm, values,nodes_attr,pr)

                    for index in range(pop):
                        inf1 = Eval(P_mu[index])
                        inf2 = Eval(P[index])

                        if inf1 > inf2:
                            P[index] = P_mu[index]
                    i += 1

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