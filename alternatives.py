'''
Multiscale Entropic Network Generator 2 (MUSKETEER2)

Copyright (c) 2011-2015 by Alexander Gutfraind and Ilya Safro. 
All rights reserved.

Use and redistribution of this file is governed by the license terms in
the LICENSE file found in the project's top-level directory.


Alternative Network Generation Algorithms

'''


import os
import time
import numpy as np
import numpy.random as npr
import random, sys
import networkx as nx
import matplotlib
#matplotlib.use('PDF')
import pdb
import pickle
import subprocess

np.seterr(all='raise')

timeNow = lambda : time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())


def er_replicate(original, params=None):
#generate the ER graph with the same density
    n = nx.number_of_nodes(original)
    p = nx.density(original)

    return nx.erdos_renyi_graph(n, p)


def ergm_replicate(original, params=None):
#generate the ERGM;  there is apparently no code for doing this
    pass

def expected_degree_replicate(original, params=None):
#generate the Chung-Lu expected degree model
    replica = nx.generators.expected_degree_graph(w=list(nx.degree(original).values()), selfloops=False)

    return replica


def random_noise_replicate(original, params=None):
    epsilon      = params['epsilon']
    preserve_degree = params.get('preserve_degree', False)
    preserve_connected = params.get('preserve_connected', nx.is_connected(original))
    #potentially we could even repeat the edge rewiring multiple times
    G = original.copy() 
    ne = original.number_of_edges()
    if ne == 0:
        return G

    if preserve_degree and not preserve_connected:
        edited_edges = random.sample(G.edges(), npr.binomial(ne, epsilon))
        random.shuffle(edited_edges)
        num_edits = len(edited_edges) / 2
        for idx in range(num_edits):
            edgeA = edited_edges[idx]
            edgeB = edited_edges[idx + num_edits]

            newA  = (edgeA[0],edgeB[0])
            newB  = (edgeA[1],edgeB[1])

            G.remove_edges_from([edgeA,edgeB])
            G.add_edges_from([newA,newB])
    elif preserve_degree and preserve_connected:
        nswap = epsilon/2. * ne
        nx.connected_double_edge_swap(G, nswap=nswap) #modified in place
    else:
        nodes = G.nodes()
        edited_edges = random.sample(G.edges(), npr.binomial(ne, epsilon))
        for edge in edited_edges:
            G.remove_edge(*edge)
            pair = random.sample(nodes, 2)
            G.add_edge(pair[0],pair[1])
    
    G.remove_edges_from(G.selfloop_edges())
    return G


def kronecker_replicate(original=None, params=None):
#generate the kronecker graph
#wishlist: this algorithm should have the option to work like a Python generator, so that fitting is only done once per input original
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('output/krondump'):
        os.mkdir('output/krondump')

    kronfit_path = params.get('kronfit_path', 'krontools/kronfit')
    krongen_path = params.get('krongen_path', 'krontools/krongen')
    if not os.path.exists(krongen_path):
        raise Exception('krongen is not found in path "%s".  Please compile krongen (SNAP library) and specify path wtih the parameter "krongen_path"'%krongen_path)
    
    base_path =  'output/krondump/kron_%d'%npr.randint(1E6)
    stdout_path  = base_path+'_out.txt'
    stderr_path  = base_path+'_err.txt'

    if params==None:
        params = {}

    if 'matrix' not in params:
        original_path = base_path + '_input.elist'
        nx.write_edgelist(nx.convert_node_labels_to_integers(original), original_path, data=False)
        matrix_path   = base_path+'_mat.txt'
        num_iterations = params.get('num_iterations', 50)
        print('Fitting (%d iterations)...'%num_iterations)
        #fitter_cmdl=["krontools/kronfit", "-gi:%d -i:%s -o:%s > %s 2> %s &"%(num_iterations,original_path,matrix_path,stdout_path,stderr_path)]
        fitter_cmdl=[kronfit_path, "-gi:%d"%num_iterations, "-i:%s"%original_path, 
                      "-o:%s"%matrix_path, ">", "%s"%stdout_path, "2>", "%s"%stderr_path, "&"]
        ret = subprocess.call(fitter_cmdl)
        assert ret == 0
        with open(matrix_path, 'r') as mat_file:
            mat_string = mat_file.readlines()[0][1:-2]
    else:
        mat_string = params['matrix']
        #matrix must have the format 'p11, .., p1n; p21, .., p2n; .. ; pn1, .., pnn'

    if params.get('just_do_fitting', False):
        return mat_string

    dimension = mat_string.split(';')[0].count(',') + 1
    num_generator_iterations = int(np.round(np.log(original.number_of_nodes())/np.log(dimension)))
    #tab-separated edgelist
    replica_path = base_path+'_replica.elist'
    replicator_cmdl=[krongen_path,
                     "-i:%d"%num_generator_iterations,
                     "-m:'%s'"%mat_string.replace(' ', ''),
                     "-o:%s"%replica_path, 
                     ">>", 
                     "%s"%stdout_path, 
                     "2>>", 
                     "%s"%stderr_path, 
                     #"&"
                     ]
    #ret = subprocess.call(replicator_cmdl)
    ret = os.system(' '.join(replicator_cmdl))
    assert ret == 0
        
    replica = nx.read_edgelist(replica_path) #this format does not show any singletons, so we will have to rebuild them ...
    assert not replica.is_directed()  #the algorithm naturally generates digraphs
    replica = nx.convert_node_labels_to_integers(replica)
    for node in range(int(dimension**num_generator_iterations)):
        if not replica.has_node(node):
            replica.add_node(node)
    return replica

def scalefree_replicate(original, params=None):
    n = nx.number_of_nodes(original)
    m = int(round(nx.number_of_edges(original)/float(n)))  
    #every node brings m edges

    return nx.barabasi_albert_graph(n=n, m=m)


def test1():
    original = nx.uniform_random_intersection_graph(n=1000, m=10, p=0.1)
    ws = watts_strogatz_replicate(original, params={'k':4})
    liu_chung = expected_degree_replicate(original)
    er = er_replicate(original)
    rand_noise1 = random_noise_replicate(original, params={'epsilon':.24, 'preserve_degree':False})
    rand_noise2 = random_noise_replicate(original, params={'epsilon':.24, 'preserve_degree':True})

    kron = kronecker_replicate(original=nx.path_graph(10), params={'num_iterations':4})
    
    import simpletesters
    assert graphutils.graph_santity_test(ws)
    assert graphutils.graph_santity_test(liu_chung)
    assert graphutils.graph_santity_test(er)
    assert graphutils.graph_santity_test(rand_noise1)
    assert graphutils.graph_santity_test(rand_noise2)
    assert graphutils.graph_santity_test(kron)

    print('Test 1 passed!')

def test2_sf():
    G = nx.path_graph(1000)
    replica = scalefree_replicate(G)
    print('Original:')
    print(G.number_of_nodes(), G.number_of_edges())
    print('Replica:')
    print(replica.number_of_nodes(), replica.number_of_edges())

    G = nx.erdos_renyi_graph(900, p=0.01)
    replica = scalefree_replicate(G)
    print('Original:')
    print(G.number_of_nodes(), G.number_of_edges())
    print('Replica:')
    print(replica.number_of_nodes(), replica.number_of_edges())

def watts_strogatz_replicate(original, params=None):
#warning: for simplicity of coding, the replica uses nodes labeled 0..n-1
    if params == None:
        params = {}
    n = nx.number_of_nodes(original)
    k = params.get('k', 4)
    p = nx.density(original)

    return nx.watts_strogatz_graph(n, k, p)


if __name__ == '__main__': 
    pass
    #test1()
    #test2_sf()
    #kronecker_replicate(original=nx.path_graph(10))
    #kronecker_replicate(params={'matrix':'0.1 0.2; 0.3 0.4'})
