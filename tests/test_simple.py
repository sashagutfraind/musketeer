'''
Multiscale Entropic Network Generator 2 (MUSKETEER2)

Copyright (c) 2011-2023 by Alexander Gutfraind and Ilya Safro. 
All rights reserved.

Use and redistribution of this file is governed by the license terms in
the LICENSE file found in the project's top-level directory.


Test Scripts

'''

import os
import time
import numpy as np
import numpy.random as npr
import random, sys
import networkx as nx
#import matplotlib
#matplotlib.use('PDF')
#import matplotlib.pylab as pylab
#import pylab

import pytest

import pdb

import musketeer.graphutils as graphutils
import musketeer.algorithms as algorithms


def alternatives_builder1():
    import alternatives
    original = nx.uniform_random_intersection_graph(n=1000, m=10, p=0.1)
    ws       = alternatives.watts_strogatz_replicate(original, params={'k':4})
    liu_chung = alternatives.expected_degree_replicate(original)
    er        = alternatives.er_replicate(original)
    rand_noise1 = alternatives.random_noise_replicate(original, params={'epsilon':.24, 'preserve_degree':False})
    rand_noise2 = alternatives.random_noise_replicate(original, params={'epsilon':.24, 'preserve_degree':True})
    
    assert graphutils.graph_sanity_test(ws)
    assert graphutils.graph_sanity_test(liu_chung)
    assert graphutils.graph_sanity_test(er)
    assert graphutils.graph_sanity_test(rand_noise1)
    assert graphutils.graph_sanity_test(rand_noise2)

#def alternatives_builder_kron():
#    kron = kronecker_replicate(original=nx.path_graph(10), params={'num_iterations':4})
#    
#    assert graphutils.graph_sanity_test(kron)


def alternatives_sf():
    import alternatives
    G = nx.path_graph(1000)
    replica = scalefree_replicate(G)
    print('Original:')
    print((G.number_of_nodes(), G.number_of_edges()))
    print('Replica:')
    print((replica.number_of_nodes(), replica.number_of_edges()))

    G = nx.erdos_renyi_graph(900, p=0.01)
    replica = alternatives.scalefree_replicate(G)
    print('Original:')
    print((G.number_of_nodes(), G.number_of_edges()))
    print('Replica:')
    print((replica.number_of_nodes(), replica.number_of_edges()))


def test_command():
    MUSKETEER_EXAMPLE_CMD = 'python musketeer.py -p "{\'edge_edit_rate\':[0.1,0.01]}" -f data-samples/karate.adjlist -t adjlist -o karate_replica.edges'
    assert 0 == os.system(MUSKETEER_EXAMPLE_CMD)

def test_integrity():
    print('Integrity testing ...')
    graphs = {'karate': nx.generators.karate_club_graph(),
              'er200_025': nx.erdos_renyi_graph(n=200, p=0.25, seed=17),
              'er200_0001': nx.erdos_renyi_graph(n=200, p=0.001, seed=42)}

    params = {'verbose':True,
              'node_edit_rate': [0],
              'edge_edit_rate': [0],
              'node_growth_rate': [0],
              'verbose':False}
    for name,G in list(graphs.items()):
        print(name)
        replica = algorithms.generate_graph(original=G, params=params)

        diff    = graphutils.graph_graph_delta(G, replica)
        assert diff['new_nodes'] == []
        assert diff['del_edges'] == []
        assert diff['new_nodes'] == []
        assert diff['del_edges'] == []
        
    print('Integrity test: PASSED')

def test_iterated(seed=None, testparams=None, params=None):        
    print('Starting iterative replication test...')
    if seed == None:
        seed = npr.randint(1E6)
        print('Setting random number generator seed: %d'%seed)
        random.seed(seed)
        npr.seed(seed)
    if params == None:
        params = {}
    defparams = {'edge_edit_rate':[0.05, 0.05], 'node_edit_rate':[0.05, 0.05], 'verbose':False, }
    defparams.update(params)
    params = defparams
    print('params')
    print(params)
    if testparams == None:
        testparams = {}
    if 'G' not in testparams:
        nn = 1000
        p  = 0.01
        G = nx.erdos_renyi_graph(nn, p)
        G.add_edges_from(nx.path_graph(nn).edges())
    else:
        G = testparams['G']
    alg = testparams.get('algorithm', algorithms.generate_graph)

    num_rounds = testparams.get('num_rounds', 10)
    print('Round: 1. Initial graph ' + getattr(G, 'name', '_'))
    for trial in range(2, num_rounds+1):
        new_G = alg(original=G, params=params)
        seed = npr.randint(1E6)
        print('Round: %d. New seed: %d'%(trial,seed))
        random.seed(seed)
        npr.seed(seed)
    print('PASSED')


def test_meshtest():
    G = graphutils.load_graph('data-samples/mesh33.edges')
    params = {'verbose':False,
              'node_growth_rate':[0.01], 
    }

    replica = algorithms.generate_graph(original=G, params=params)

    assert replica.number_of_nodes() >= G.number_of_nodes()
    assert replica.number_of_nodes() <= G.number_of_nodes()*1.10

    assert replica.number_of_edges() >= G.number_of_edges()
    assert replica.number_of_edges() <= G.number_of_edges()*1.10


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_smoketest(directed, weighted):
    print(f'Smoke testing: directed={directed}, weighted={weighted}')
    graphs = {'karate': nx.generators.karate_club_graph(),
              'er200_025': nx.erdos_renyi_graph(n=200, p=0.25, seed=42),
              'er200_0001': nx.erdos_renyi_graph(n=200, p=0.001, seed=42)}
    for name, G in graphs.items():
        if weighted:
            H = nx.Graph()
            H.add_edges_from((u, v, {'weight':1.0}) for u, v in G.edges())
        else:
            H = G
        if directed:
            H = nx.to_directed(H)
        graphs[name] = H

    params = {'verbose':False,
              'node_edit_rate': [0.1/(1.+i) for i in range(100)],
              'edge_edit_rate': [0.1/(1.+i) for i in range(100)],
              'node_growth_rate': [0.1/(1.+i) for i in range(100)]}
    for name,G in (graphs.items()):
        print(name)
        replica = algorithms.generate_graph(original=G, params=params)
        assert list(nx.selfloop_edges(G)) == []        

    return

