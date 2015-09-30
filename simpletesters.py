'''
Multiscale Entropic Network Generator 2 (MUSKETEER2)

Copyright (c) 2011-2015 by Alexander Gutfraind and Ilya Safro. 
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
import pdb
import pickle
import graphutils

np.seterr(all='raise')

timeNow = lambda : time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
       
def integrity_test():
    import algorithms
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

def iterated_test(seed=None, testparams=None, params=None):        
    import algorithms
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

def smoke_test():
    import algorithms
    print('Smoke testing ...')
    graphs = {'karate': nx.generators.karate_club_graph(),
              'er200_025': nx.erdos_renyi_graph(n=200, p=0.25, seed=42),
              'er200_0001': nx.erdos_renyi_graph(n=200, p=0.001, seed=42)}

    params = {'verbose':False,
              'node_edit_rate': [0.1/(1.+i) for i in range(100)],
              'edge_edit_rate': [0.1/(1.+i) for i in range(100)],
              'node_growth_rate': [0.1/(1.+i) for i in range(100)]}
    for name,G in list(graphs.items()):
        print(name)
        #print '  nn=%d,ne=%d'%(G.number_of_nodes(), G.number_of_edges())
        replica = algorithms.generate_graph(original=G, params=params)
        #print '  nn=%d,ne=%d'%(replica.number_of_nodes(), replica.number_of_edges())
        assert G.selfloop_edges() == []

    assert 0 == os.system(graphutils.MUSKETEER_EXAMPLE_CMD)
        

    print('Smoke test: PASSED')
    print()
    return

#we use a dict, for possible future detailed testing code, such as testing for numerical range
valid_params = {'accept_chance_edges':None,
                'algorithm':None,
                'algorithm_for_coarsening':None, #TODO: document
                'algorithm_for_uncoarsening':None, #TODO: document
                'coarsening_density_limit':None, #TODO: document
                'component_is_edited':None, #TODO: document
                'deep_copying':None,
                'deferential_detachment_factor':None,
                'do_coarsen_tester':None, #TODO: document
                'do_uncoarsen_tester':None, #TODO: document
                'dont_cutoff_leafs':None,
                'edge_edit_rate':None,
                'edge_growth_rate':None,
                'edge_welfare_fraction':None,
                'edit_edges_tester':None, #TODO: document
                'edit_nodes_tester':None, #TODO: document
                'enforce_connected':None,
                'fine_clustering':None,
                'locality_algorithm':None,
                'locality_bias_correction':None,
                'long_bridging':None, #TODO: document
                'maintain_edge_attributes':None,
                'maintain_node_attributes':None,
                'matching_algorithm':None, #TODO document
                'memoriless_interpolation':None,    
                'metric_runningtime_bound':None, #TODO: document
                'minorizing_node_deletion':None, #TODO: document
                'new_edge_horizon':None,
                'node_edit_rate':None,
                'node_growth_rate':None,
                'num_deletion_trials':None, #TODO: document
                'num_insertion_trials':None, #TODO: document
                'num_insertion_searches_per_distance':None, #TODO: document
                'num_pairs_to_sample':None, #TODO: document
                'num_snapshots':None,
                'num_trial_particles':None, #TODO: document
                'num_v_cycles':None,
                'post_processor':None,
                'preserve_clustering_on_deletion':None,
                'reporting_precision':None, #TODO: document
                'revise_graph_tester':None, #TODO: document
                'retain_intermediates':None,
                'seed_threshold_1':None, #TODO: document
                'seed_threshold_2':None, #TODO: document
                'search_method':None, #TODO: document 
                'skip_param_sanity_check':False, #TODO: document
                'stats_report_on_all_levels':None, #presumes retain_intermediates
                'triangle_distribution_limit':None, #TODO: document
                'suppress_warnings':None,
                'verbose':None,
                'weighted_step':None,
                }

def validate_params(params):
    bad_params = False
    for k in params:
        if k not in valid_params and (not str(k).startswith('_')):
            if params.get('verbose', True):
                print('Unknown or undocumented parameter: %s'%k)
                print('Hint: for a list of valid parameters, see valid_params variable in simpletesters.py')
                print()
            bad_params = True
    
    if params.get('memoriless_interpolation', False) and params.get('deep_copying', True):
        if params.get('verbose', True):
            print('Memoriless_interpolation=True requires deep_copying=False')
        bad_params = True

    if params.get('stats_report_on_all_levels', False) and not params.get('retain_intermediates'):
        if params.get('verbose', True):
            print('Making retain_intermediates=True')
        params['retain_intermediates'] = True


    if bad_params and not params.get('skip_param_sanity_check', False):
        if params.get('verbose', True):
            print('Existing ...')
        sys.exit(1)


if __name__ == '__main__': 
    smoke_test()
    integrity_test()
    iterated_test()
