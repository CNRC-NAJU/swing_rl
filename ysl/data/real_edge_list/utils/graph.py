import networkx as nx
import numpy as np
from collections import OrderedDict as od

def gen_sf_net(node, edge, alpha, depth, seed=0):
    np.random.seed(seed)
    g = nx.Graph()
    e = 0
    weight = [(n+1)**(-alpha) for n in range(node)]
    w_sum = sum(weight)
    weight = [w/w_sum for w in weight]
    while(1):
        temp = np.unique(np.random.choice(node, 2, p=weight))
        while(len(temp)!=2):
            temp = np.unique(np.random.choice(node, 2, p=weight))
        g.add_edge(temp[0], temp[1])
        e += 1
        if g.number_of_edges() == edge:
            break
    g = g.subgraph(max(nx.connected_components(g), key=len))
    g = nx.convert_node_labels_to_integers(g)
    lens = dict(nx.all_pairs_shortest_path_length(g))
    local_graphs = [nx.ego_graph(g, n, radius=depth) for n in range(g.number_of_nodes())]
    old_idxes = [[n for n in lg.nodes()] for lg in local_graphs]
    nn_idxes = [[old_idxes[n].index(nn) for nn in g.neighbors(n)] for n in range(g.number_of_nodes())]
    n_idxes = [old_idxes[n].index(n) for n in range(g.number_of_nodes())]

    return g, lens, local_graphs, old_idxes, nn_idxes, n_idxes


def gen_from_g(g, depth):
    g = nx.convert_node_labels_to_integers(g)
    lens = dict(nx.all_pairs_shortest_path_length(g))
    local_graphs = [nx.ego_graph(g, n, radius=depth) for n in range(g.number_of_nodes())]
    old_idxes = [[n for n in lg.nodes()] for lg in local_graphs]
    nn_idxes = [[old_idxes[n].index(nn) for nn in g.neighbors(n)] for n in range(g.number_of_nodes())]
    n_idxes = [old_idxes[n].index(n) for n in range(g.number_of_nodes())]
    nn_nnn_idxes = [[idx for idx in range(len(old_idxes[n])) if idx!=old_idxes[n].index(n)] for n in range(g.number_of_nodes())]

    return g, lens, local_graphs, old_idxes, nn_nnn_idxes, nn_idxes, n_idxes


def plot_graph(g, pos, mask_gen, label=False):
    node_color = ['tab:blue' if mask else 'tab:red' for mask in mask_gen]
    nodes = nx.draw_networkx_nodes(g, pos, node_color=node_color, node_size = 10)
    nx.draw_networkx_edges(g, pos, edge_color='black')
    if label:
        nx.draw_networkx_labels(g, pos, font_color='white')
    nodes.set_edgecolor('black')


def shk_network(nt, n0, p, q, r, s):

    # nt: the number of nodes desired
    # n0: the number of nodes initialized
    # p, q, r, s: parameters

    # initialize
    g = nx.Graph()
    pos = np.random.random((nt, 2))
    n = n0

    for ni in range(n0):
        for nj in range(ni+1, n0):
            dist = np.sqrt(np.sum(np.power(pos[ni] - pos[nj], 2)))
            g.add_edge(ni, nj, weight = dist)
    g = nx.minimum_spanning_tree(g)

    for (n1, n2, d) in g.edges(data=True):
        d.clear()

    # init edges
    m = n0 * (1 - s) * (p + q) 

    for _ in range(int(m)):
        dists = list(nx.shortest_path_length(g))
        max_fr = 0
        max_ni, max_nj = -1, -1
        for ni in range(n):
            for nj in range(ni+1, n):
                temp_fr = (dists[ni][1][nj]+1.)**r / np.sqrt(np.sum(np.power(pos[ni] - pos[nj], 2.)))
                if max_fr < temp_fr and not g.has_edge(ni, nj):
                    max_ni = ni
                    max_nj = nj
                    max_fr = temp_fr
        if max_ni == -1:
            continue
        g.add_edge(max_ni, max_nj)

    # network growth
    for n in range(n0, nt):
        coin_s = np.random.random(1)
        if coin_s[0] < s and g.number_of_edges() > 0:
            # select an existing edge at random
            target = list(g.edges())[np.random.randint(g.number_of_edges())]
            # and add a node
            pos[n] = np.sum(pos[list(target)], axis = 0) / 2.
            while len(pos[pos==pos[n]]) != 2: # if that position is already occupied, get another one
                target = list(g.edges())[np.random.randint(g.number_of_edges())]
                pos[n] = np.sum(pos[list(target)], axis = 0) / 2.
            if(len(pos[pos==pos[n]])!=2):
                raise ValueError(pos)
            g.add_edge(n, target[0])
            g.add_edge(n, target[1])
            # removed the selected edge
            g.remove_edge(*target)

        else:
            # set a node n at random position 
            # pos[n] = np.random.random((2,))
            # find a node m for which d_{nm} is minimal
            e_dists = np.sqrt(np.sum(np.power(pos[:n] - pos[n], 2.), axis = 1))
            arg_min = np.argmin(e_dists)
            g.add_edge(arg_min, n)

            # with probability p, find another node m for which fr_{nm} is maximal and add an edge
            coin_p = np.random.random(1)
            if coin_p[0] < p:
                dists = np.fromiter(od(sorted(nx.shortest_path_length(g, source = n).items())).values(), dtype=float)
                dists = dists[:-1]
                frs = (dists + 1.)**r / e_dists
                frs[arg_min] = -1
                arg_max = np.argmax(frs)
                g.add_edge(arg_max, n)

            coin_q = np.random.random(1)
            if coin_q[0] < q and g.number_of_nodes() > 2:
                # select an existing node at random
                rand_node = np.random.randint(n)
                e_dists = np.sqrt(np.sum(np.power(pos[:n+1] - pos[rand_node], 2.), axis = 1))
                dists = np.fromiter(od(sorted(nx.shortest_path_length(g, source = rand_node).items())).values(), dtype=float)
                dists = dists[:-1]
                e_dists = e_dists[:-1]
                e_dists[rand_node] = 1e-8 # doesn't matter actually.

                frs = (dists + 1.)**r / e_dists

                frs[rand_node] = -1
                for nb in g.neighbors(rand_node):
                    if nb == n:
                        continue
                    frs[nb] = -1
                 
                arg_max = np.argmax(frs)
                if frs[arg_max] != -1:
                    g.add_edge(arg_max, rand_node)

    return g, pos