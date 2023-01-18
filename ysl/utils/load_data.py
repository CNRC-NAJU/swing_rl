import numpy as np
import networkx as nx

from utils.solver import *
from utils.graph import *

def load_uk(protocol=0, lossless=True):
    # UK grid
    g = nx.DiGraph()

    num_nodes = 313

    mass = np.zeros((num_nodes, ))
    gamma = np.zeros((num_nodes, ))
    power_gen = np.zeros((num_nodes, ))
    power_con = np.zeros((num_nodes, ))

    phase = np.zeros((num_nodes, ))
    dphase = np.zeros((num_nodes, ))

    with open('data/real_edge_list/uk_data/bus.txt') as f:
        for line in f:
            line = line.split('\t')[1:]
            n = int(line[0])-1
            g.add_node(n)
            power_con[n] = -float(line[2])*0.01

    nodes = g.nodes()
    adjm = nx.to_numpy_array(g, sorted(g.nodes(), key=lambda n: n))
    alpha = np.copy(adjm)

    with open('data/real_edge_list/uk_data/line.txt') as f:
        for line in f:
            line = line.split('\t')[1:]
            if lossless:
                w = 1./float(line[3])
            else:
                w = 1./np.absolute(float(line[2]) - 1j * float(line[3]))
            g.add_edge(int(line[0])-1, int(line[1])-1, weight=w)
            fn, tn = int(line[0])-1, int(line[1])-1
            adjm[fn][tn] += w
            adjm[tn][fn] += w
            angle = 0
            if not lossless:
                angle = np.arctan2(float(line[2]), float(line[3]))
            alpha[fn][tn] = angle
            alpha[tn][fn] = angle

    mask_gen = {n: False for n in nodes}
    with open('data/real_edge_list/uk_data/gen.txt') as f:
        for line in f:
            line = line.split('\t')[1:]
            n = int(line[0])-1
            mask_gen[n] = True
            power_gen[n] = float(line[1])*0.01

    pos = {n: (0, 0) for n in nodes}
    n = 0
    with open('data/real_edge_list/uk_data/bus_pos.txt') as f:
        for line in f:
            line = line.split('\t')[1:]
            pos[n] = (float(line[0]), float(line[1]))
            n += 1

    nx.set_node_attributes(g, mask_gen, 'mask_gen')
    nx.set_node_attributes(g, pos, 'pos')

    pos = [[p[0], p[1]] for _, p in nx.get_node_attributes(g, 'pos').items()]
    mask_gen = np.array([b for _, b in nx.get_node_attributes(g, 'mask_gen').items()])

    mass[mask_gen] = np.loadtxt('data/real_edge_list/uk_data/gen_inertia.txt')
    mass[mask_gen==False] = np.min(mass[mass!=0])

    gamma[mask_gen] += np.loadtxt('data/real_edge_list/uk_data/gen_damp.txt')
    gamma += np.loadtxt('data/real_edge_list/uk_data/bus_damp.txt')

    power = power_gen + power_con

    protocols = []
    # protocol_0: no truncation, lossless
    protocols.append({
        'gamma_min': 0, 
        'mass_min': 0, 
        'K_max': 100000000, 
        })
    # protocol_1: lossless and truncation with gamma_min = 0.1, mass_min = 0.1, K_max = 1000
    protocols.append({
        'gamma_min': 0.1, 
        'mass_min': 0.1, 
        'K_max': 1000, 
        })
    # protocol_2: lossless and truncation with gamma_min = 0.1, mass_min = 0.01, K_max = 1000
    protocols.append({
        'gamma_min': 0.1, 
        'mass_min': 0.01, 
        'K_max': 1000, 
        })
    # protocol_3: lossless and truncation with gamma_min = 0.1, mass_min = 0.01, K_max = 100
    protocols.append({
        'gamma_min': 0.1, 
        'mass_min': 0.01, 
        'K_max': 100, 
        })
    # protocol_4: lossless and truncation with gamma_min = 0.1, mass_min = 0.1, K_max = 100
    protocols.append({
        'gamma_min': 0.1, 
        'mass_min': 0.1, 
        'K_max': 100, 
        })
    # protocol_5: lossless and truncation with gamma_min = 0.1, mass_min = 0.1, K_max = 100
    protocols.append({
        'gamma_min': 0.1, 
        'mass_min': 1, 
        'K_max': 100, 
        })
    # protocol_6: lossless and truncation with gamma_min = 0.1, mass_min = 0.1, K_max = 100
    protocols.append({
        'gamma_min': 0.1, 
        'mass_min': 0.5, 
        'K_max': 100, 
        })

    gamma_min, mass_min, K_max = protocols[protocol].values()
    gamma[gamma<gamma_min] = gamma_min
    mass[mass<mass_min] = mass_min
    adjm[adjm>K_max] = K_max

    phase = np.loadtxt('data/real_edge_list/uk_data/vars/protocol_'+str(protocol)+'/steady_phase.txt')
    dphase = np.loadtxt('data/real_edge_list/uk_data/vars/protocol_'+str(protocol)+'/steady_dphase.txt')
    power = np.loadtxt('data/real_edge_list/uk_data/vars/protocol_'+str(protocol)+'/init_power.txt')

    # gen_type = np.genfromtxt('data/real_edge_list/uk_data/gen_type.txt', dtype='str')


    return g, adjm, alpha, mask_gen, phase, dphase, power, mass, gamma, pos, power_gen<15, protocols[protocol]['mass_min']




def load_ieee_testgrid():
    # IEEE test grid
    g = nx.DiGraph()

    num_nodes = 313

    mass = np.zeros((num_nodes, ))
    gamma = np.zeros((num_nodes, ))
    power_gen = np.zeros((num_nodes, ))
    power_con = np.zeros((num_nodes, ))

    phase = np.zeros((num_nodes, ))
    dphase = np.zeros((num_nodes, ))

    with open('data/real_edge_list/uk_data/bus.txt') as f:
        for line in f:
            line = line.split('\t')[1:]
            n = int(line[0])-1
            g.add_node(n)
            power_con[n] = -float(line[2])*0.01

    nodes = g.nodes()
    adjm = nx.to_numpy_array(g, sorted(g.nodes(), key=lambda n: n))

    with open('data/real_edge_list/uk_data/line.txt') as f:
        for line in f:
            line = line.split('\t')[1:]
            w = 1./float(line[3])
            g.add_edge(int(line[0])-1, int(line[1])-1, weight=w)
            fn, tn = int(line[0])-1, int(line[1])-1
            adjm[fn][tn] += w
            adjm[tn][fn] += w

    mask_gen = {n: False for n in nodes}
    with open('data/real_edge_list/uk_data/gen.txt') as f:
        for line in f:
            line = line.split('\t')[1:]
            n = int(line[0])-1
            mask_gen[n] = True
            power_gen[n] = float(line[1])*0.01

    pos = {n: (0, 0) for n in nodes}
    n = 0
    with open('data/real_edge_list/uk_data/bus_pos.txt') as f:
        for line in f:
            line = line.split('\t')[1:]
            pos[n] = (float(line[0]), float(line[1]))
            n += 1

    nx.set_node_attributes(g, mask_gen, 'mask_gen')
    nx.set_node_attributes(g, pos, 'pos')

    pos = [[p[0], p[1]] for _, p in nx.get_node_attributes(g, 'pos').items()]
    mask_gen = np.array([b for _, b in nx.get_node_attributes(g, 'mask_gen').items()])


    mass[mask_gen] = np.loadtxt('data/real_edge_list/uk_data/gen_inertia.txt')
    mass[mask_gen==False] = np.min(mass[mass!=0])
    # mass *= 10

    gamma[mask_gen] += np.loadtxt('data/real_edge_list/uk_data/gen_damp.txt')
    gamma += np.loadtxt('data/real_edge_list/uk_data/bus_damp.txt')

    power = power_gen + power_con

    phase = np.loadtxt('data/real_edge_list/uk_data/vars/steady_phase.txt')
    dphase = np.loadtxt('data/real_edge_list/uk_data/vars/steady_dphase.txt')
    power = np.loadtxt('data/real_edge_list/uk_data/vars/init_power.txt')


    return g, adjm, mask_gen, phase, dphase, power, mass, gamma, pos



def load_shk(case=2, seed=123):

    # init config
    np.random.seed(seed)
    g = nx.Graph()
    if case==1:
        g = nx.read_edgelist('data/real_edge_list/es_edge_list.txt', nodetype=int)
        K = 3
        gamma = 0.5
        net_type='es w K=3'
    elif case==2:
        g, pos = shk_network(20, 1, 0.3, 0.1, 1, 0.2)
        K = 4
        # K = 3
        gamma = 0.5
        net_type='SHK (1, 0.3, 0.1, 1, 0.2) w K=4 N=20'
        # net_type='SHK (1, 0.3, 0.1, 1, 0.2) w K=3'
    elif case==3:
        g, pos = shk_network(20, 1, 0.3, 0.1, 10, 0.2)
        K = 4
        # K = 3
        gamma = 0.5
        net_type='SHK (1, 0.3, 0.1, 10, 0.2) w K=4 N=20'
        # net_type='SHK (1, 0.3, 0.1, 10, 0.2) w K=3'
    elif case==4:
        g, pos = shk_network(100, 1, 0.3, 0.1, 1, 0.2)
        K = 4
        gamma = 0.5
        net_type = 'SHK (1, 0.3, 0.1, 1, 0.2) w K=4, N=100'
    elif case==5:
        g, pos = shk_network(300, 1, 0.3, 0.1, 1, 0.2)
        K = 4
        gamma = 0.5
        net_type = 'SHK (1, 0.3, 0.1, 1, 0.2) w K=4, N=100'


    power = [-1. for _ in range(int(g.number_of_nodes()/2))]
    power.extend([1. for _ in range(int(g.number_of_nodes()/2))])
    np.random.shuffle(power)
    power = np.array(power)

    phase = np.zeros((g.number_of_nodes(), ), dtype=float)
    dphase = np.zeros((g.number_of_nodes(), ), dtype=float)
    mass = np.ones((g.number_of_nodes(), ), dtype=float)
    gamma *= np.ones((g.number_of_nodes(), ), dtype=float)

    mask_gen = power==1

    # adj = [list(g.neighbors(n)) for n in range(g.number_of_nodes())]
    adjm = nx.to_numpy_array(g) * K


    return g, adjm, mask_gen, phase, dphase, power, mass, gamma, pos, net_type