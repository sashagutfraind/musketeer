'''
Multiscale Entropic Network Generator 2 (MUSKETEER2)

Copyright (c) 2011-2023 by Alexander Gutfraind and Ilya Safro.
All rights reserved.

Use and redistribution of this file is governed by the license terms in
the LICENSE file found in the project's top-level directory.

Core Algorithms

'''

import os, copy
import time
import numpy as np
import numpy.random as npr
import random, sys
import networkx as nx
import pdb
import gc
#module might be refered in params['algorithm']

from . import graphutils

np.seterr(all='raise')

timeNow = lambda : time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
random_one = lambda arr: random.sample(arr, 1)[0]
max_int = np.iinfo(np.int32(10)).max

def chart_paths(G, source, new_edge_horizon, search_method='particles', params=None):
    """
    determine the path structure for paths from source to a neighbor through a second path with a random walk
    """
    #weighted_random = params.get('weighted_random_walk_method', lambda G,src,valid_nodes: random_one(valid_nodes)) #weighted_step
    weighted_step = params.get('weighted_step', False)
    sm = sum
    rds = random.shuffle
    find_next = weighted_step_advanced
    G_adj = G.adj
    G_neighborsSet = lambda u: set(G_adj[u].keys())


    estimate_of_paths   = np.zeros(new_edge_horizon+1)
    num_trial_particles = params.get('num_trial_particles', 50)
    if search_method == 'particles':  #random walk from a neighbor to find any OTHER nbr
        source_nbs = G_neighborsSet(source)
        num_misses = 0
        for cur_loc in [random_one(source_nbs) for i in range(num_trial_particles)]:
            blocked = set([source, cur_loc])
            for d in range(2, new_edge_horizon+1):
                cur_loc = find_next(G=G, start_node=cur_loc, weighted_step=weighted_step, blocked=blocked, rds=rds, sm=sm)
                if cur_loc == None:  #stuck in a self-made corner
                    # num_misses += 1
                    #print 'corner'
                    #assert len(blocked) == d
                    break
                #blocked.add(cur_loc)
                if cur_loc in source_nbs:
                    estimate_of_paths[d] += 1
                    #print 'reached nb'
                    #assert len(blocked) == d + 1
                    break
                elif d == new_edge_horizon:
                    # num_misses += 1
                    #print 'escaped'
                    break
        if sum(estimate_of_paths) > 0:
            estimate_of_paths /= sum(estimate_of_paths)
        num_missed_neighbors = None #not applicable
    elif search_method == 'particles_shortest':  #random walk estimates the shortest alternative path to a neighbor
        source_nbs = G_neighborsSet(source)
        nb_steps = dict.fromkeys(list(G_adj[source].keys()), np.inf)
        num_misses = 0
        #wishlist: the distance depends on the starting nb.  we need to be consistent and use the same starting nb
        for cur_loc in [random_one(source_nbs) for i in range(num_trial_particles)]:
            blocked = set([source, cur_loc])
            for d in range(2, new_edge_horizon+1):
                cur_loc = find_next(G=G, start_node=cur_loc, weighted_step=weighted_step, blocked=blocked, rds=rds, sm=sm)
                if cur_loc == None:  #stuck in a self-made corner
                    #num_misses += 1
                    #print 'corner'
                    #assert len(blocked) == d
                    break
                #blocked.add(cur_loc)
                if cur_loc in source_nbs and d <= nb_steps[cur_loc]:
                    nb_steps[cur_loc] = d
                    #assert len(blocked) == d + 1
                    break
                elif d == new_edge_horizon:
                    #num_misses += 1
                    #print 'escaped'
                    break
        for nb in nb_steps:
            d = nb_steps[nb]
            if d < np.inf:
                estimate_of_paths[d] += 1
        if sum(estimate_of_paths) > 0:
            estimate_of_paths /= sum(estimate_of_paths)
        num_missed_neighbors = None #not applicable
    elif search_method == 'particles3':     #routes, determines how many neighbors were never reached
        source_nbs = dict.fromkeys(list(G_adj[source].keys()), False) #which neighbors were reached
        for cur_loc in [random_one(source_nbs) for i in range(num_trial_particles)]:
            blocked = set([source, cur_loc])
            for d in range(2, new_edge_horizon+1):
                cur_loc = find_next(G=G, start_node=cur_loc, weighted_step=weighted_step, blocked=blocked, rds=rds, sm=sm)
                if cur_loc == None:  #stuck in a self-made corner
                    break
                if cur_loc in source_nbs:
                    estimate_of_paths[d] += 1
                    source_nbs[cur_loc] = True
                    break
                #blocked.add(next_loc) #self-avoiding
                #wishlist: maybe break if find all the neighbors
        num_missed_neighbors = sum(1 for nb in source_nbs if not source_nbs[nb])
        if sum(estimate_of_paths) > 0:
            estimate_of_paths = (source_nbs.__len__() - num_missed_neighbors) * estimate_of_paths / sum(estimate_of_paths)
        num_misses = None #N/A
    else:
        raise ValueError ('Unknown search method')

    #return estimate_of_paths, num_misses
    #WARNING: many of our results were based on setting this to 0.
    #we do not correctly estimate the number of misses ...

    #print estimate_of_paths, num_misses
    #pdb.set_trace()
    #return estimate_of_paths, 0

    return {'estimate_of_paths':estimate_of_paths,
            'num_missed_neighbors':num_missed_neighbors,
            'num_attempts':num_trial_particles,
            'num_misses':num_misses,
            }


def check_and_fix_connectivity(G, params):
    new_edges = set()
    ccs = [cc for cc in nx.connected_components(G)]
    if len(ccs) > 1:
        giant_comp = ccs[0]
        for cc in ccs[1:]:
            u,w = random.sample(giant_comp, 1)[0], random.sample(cc, 1)[0]
            G.add_edge(u,w)
            new_edges.add((u,w))
    return new_edges

def clean_c_data(G, c_data):
    """ 
    this method serves no function other than trapping bugs: 
    it removes data which should not be used in uncoarsening
    """
    aggregates    = c_data['aggregates']
    trapped_edges = c_data['trapped_edges']
    home_nodes    = c_data['home_nodes']
    merged_edges  = c_data['merged_edges']

    deleted_seeds   = [node for node in aggregates   if not G.has_node(node)]
    deleted_c_edges = [edge for edge in merged_edges if not G.has_edge(*edge)]

    for node in deleted_seeds:
        trapped_edges.pop(node)
        for guest in aggregates[node]:
            home_nodes.pop(guest)
        aggregates.pop(node)
    for edge in deleted_c_edges:
        merged_edges.pop(edge)

    return c_data

def compute_topology_data(G, level, params):
    """
    measures statistics of the topology of graph G
    #wishlist: for generation of multiple replicas, it would be helpful to do this computation once for the speedup
    """
    if params.get('verbose', True):
        sys.stdout.write('Topology estimation ... ')
        sys.stdout.flush()
    tpl_data = {}

    if G.is_directed():
        tpl_data['enforce_connected'] = False
    else:
        tpl_data['enforce_connected'] = nx.is_connected(G)

    #estimates the probability of friending a node at distance d
    new_edge_horizon     = params.get('new_edge_horizon', estimate_horizon(G))  #no edges added to nodes beyond the horizon
    num_pairs_to_sample  = params.get('num_pairs_to_sample', 100)  #revise comment #no edges added to nodes beyond the horizon
    locality_algorithm   = params.get('locality_algorithm', chart_paths)  #which method to use for computing locality

    num_nodes_beyond_the_horizon = 0.
    overall_estimates    = np.zeros(new_edge_horizon+1)

    #for each node u in a sample, select one neighbor, and compute the distance up to H steps
    #    now see how many of the other neighbors of u have been reached.
    #    those not reached could be a kind of "beyond-the-horizon-edges", which are also possible
    for source in random.sample(G.nodes(), min(G.number_of_nodes(), num_pairs_to_sample)):
        source_degree = G.degree(source)
        if source_degree == 0:
            continue
        elif source_degree == 1:
            #num_nodes_beyond_the_horizon += 1
            continue
            #WARNING: many of our results are based on setting this to 0
            #source_degree==1 is an important indicator that many edges are, in effect, chance edges
        else:
            #target_nb = random.choice(G.neighbors(source)) #we will pick any one
            locality_data = locality_algorithm(G, source, new_edge_horizon, params=params, search_method=params.get('search_method', 'particles'))
            #estimate_of_paths (and num_nodes_beyond_the_horizon) should have the norm of the number of nodes actually reached (not reached), b/c the source has different degrees
            estimate_of_paths           = locality_data['estimate_of_paths']
            num_missed_neighbors        = locality_data['num_missed_neighbors']
            num_attempts                = locality_data['num_attempts']
            num_misses                  = locality_data['num_misses']
            overall_estimates  += estimate_of_paths
            if num_missed_neighbors != None:
                num_nodes_beyond_the_horizon += num_missed_neighbors
            else:
                num_nodes_beyond_the_horizon += num_misses
    try:
        locality_bias_correction = params['locality_bias_correction'][level]
    except:
        locality_bias_correction = 0.
    #locality_bias_correction = 0

    if locality_bias_correction > 0: #shift weight downward b/c this estimator under-rates correlations between neighbors
        overall_estimates[-1]          += locality_bias_correction     * num_nodes_beyond_the_horizon
        num_nodes_beyond_the_horizon   *= (1-locality_bias_correction)
        for dis in range(len(overall_estimates)-1, 2, -1):
            overall_estimates[dis-1] += locality_bias_correction     * overall_estimates[dis]
            overall_estimates[dis]   *= (1-locality_bias_correction)
    else:                            #shift weight upwards b/c this estimator over-rates correlations between neighbors
        for dis in range(len(overall_estimates)-1):
            overall_estimates[dis+1]   += -locality_bias_correction     * overall_estimates[dis]
            overall_estimates[dis]     *= (1+locality_bias_correction)
        num_nodes_beyond_the_horizon += -locality_bias_correction     * overall_estimates[-1]
        overall_estimates[-1]        *= (1+locality_bias_correction)

    accept_chance_edges = params.get('accept_chance_edges', 1.0)
    assert accept_chance_edges >= 0 and accept_chance_edges <= 1.0
    if sum(overall_estimates) > 0 or (num_nodes_beyond_the_horizon > 0 and accept_chance_edges > 0):
        if accept_chance_edges > 0:
            norm = accept_chance_edges*num_nodes_beyond_the_horizon + sum(overall_estimates)
            chance_edge_prob  = float(accept_chance_edges*num_nodes_beyond_the_horizon)/norm
        else:
            norm = sum(overall_estimates)
            chance_edge_prob  = 0.
        locality_acceptor = overall_estimates/norm
    else: #fallback
        locality_acceptor = [0., 0.] + [0.5/(2**d) for d in range(1, min(new_edge_horizon,G.number_of_nodes()-2))]
        chance_edge_prob  = 0.
        if G.number_of_edges() > 10 and nx.density(G) > 0.2:
            print_warning(params, 'Warning: unable to estimate edge locality.')
            print_warning(params, 'Consider setting allow_chance_edges to positive values')
    assert locality_acceptor[0] == 0.
    assert locality_acceptor[1] == 0.
    if locality_bias_correction > 0 and locality_acceptor[2] > 0.8:
        print_warning(params, 'Warning: extreme locality at distance 2.  Might make it difficult to insert edges')

    tpl_data['locality_acceptor'] = locality_acceptor
    tpl_data['chance_edge_prob']  = chance_edge_prob

    #print tpl_data

    if params.get('verbose', True):
        sys.stdout.write('Done topology.'+os.linesep)

    return tpl_data

def do_coarsen(G, params):
    G_coarse = nx.empty_graph()
    aggregates = {} #nodes within new nodes.  seed->fine_nodes
    trapped_edges = {} #edges within new nodes.  seed->fine_edges
    home_nodes    = {} #node->seed
    merged_edges  = {} #edge->internal edges

    algorithm_for_coarsening      = params.get('algorithm_for_coarsening', seed_finder_matching) #alt: seed_finder_weight_alg
    seeds, home_nodes, aggregates = algorithm_for_coarsening(G, params)
    if params.get('verbose', True):
        print('nn: %d ne: %d (seeds: %d)'%(G.number_of_nodes(),G.number_of_edges(),len(seeds)))

    free_edges = set()        #edges not within any coarse node.  they will be retained in the coarse graph (many-to-one mapping)
    for seed in seeds:
        G_coarse.add_node(seed)
        G_coarse.nodes[seed]['weight'] = sum(G.nodes[nb].get('weight', 1.) for nb in aggregates[seed])

        trapped_edges[seed] = G.subgraph(aggregates[seed]).edges(data=False)

        for nb in aggregates[seed]:
            for nbnb in G.neighbors(nb):
                if nbnb in aggregates[seed] or (nbnb,nb) in free_edges:
                    continue
                free_edges.add((nb,nbnb))

    for u,v in free_edges:
        s1 = home_nodes[u]
        s2 = home_nodes[v]
        uv_edge_wt = G.edges[u, v].get('weight', 1.0)
        if (s1,s2) in merged_edges:
            merged_edges[(s1,s2)].append((u,v))
            G_coarse.edges[s1, s2]['weight'] += uv_edge_wt
            assert (v,u) not in merged_edges[(s1,s2)]
        elif (s2,s1) in merged_edges:
            merged_edges[(s2,s1)].append((u,v))
            G_coarse.edge[s2][s1]['weight'] += uv_edge_wt
            assert (v,u) not in merged_edges[(s2,s1)]
        else:
            G_coarse.add_edge(s1,s2, weight=uv_edge_wt)
            merged_edges[(s1,s2)] = [(u,v)]
            assert (v,u) not in merged_edges[(s1,s2)]

    for u in G:
        assert u in home_nodes
        assert home_nodes[u] in seeds
    for (u,v) in G.edges():
        hu = home_nodes[u]
        hv = home_nodes[v]
        if hu == hv:
            assert not G_coarse.has_edge(hu,hv)
            assert (u,v) in trapped_edges[hu] or (v,u) in trapped_edges[hv]
        else:
            assert G_coarse.has_edge(hu,hv)
            assert (hu,hv) in merged_edges or (hv,hu) in merged_edges

    c_data = {'aggregates':aggregates, 'trapped_edges':trapped_edges, 'home_nodes':home_nodes, 'merged_edges':merged_edges}
    if 'do_coarsen_tester' in params:
        params['do_coarsen_tester'](G, G_coarse, c_data)

    return G_coarse, c_data


def do_uncoarsen(G_coarse, c_data, params):
    if callable(params.get('algorithm_for_uncoarsening', False)):
        return params['algorithm_for_uncoarsening'](G_coarse, c_data, params)

    aggregates    = c_data['aggregates']
    trapped_edges = c_data['trapped_edges']
    home_nodes    = c_data['home_nodes']
    merged_edges  = c_data['merged_edges']

    G_fine = nx.empty_graph()
    G_fine.add_nodes_from(home_nodes)

    for seed in trapped_edges:
        for u,v in trapped_edges[seed]:
            if u in G_fine and v in G_fine:
                G_fine.add_edge(u,v)
            #u or v must have been deleted

    for s1,s2 in G_coarse.edges():
        if (s1,s2) in merged_edges:
            s1s2 = merged_edges[(s1,s2)]
        else:
            s1s2 = merged_edges[(s2,s1)]

        for u,v in s1s2:
            assert u in G_fine
            assert v in G_fine
            G_fine.add_edge(u,v)

    if 'do_uncoarsen_tester' in params:
        params['do_uncoarsen_tester'](G_coarse, G_fine, c_data)

    return G_fine



def edit_edges_sequential(G, edge_edit_rate, edge_growth_rate, tpl_data, params):
    """
    edit edges: first delete, then insert
    """
    verbose = params.get('verbose', True)
    try:
        edit_rate = edge_edit_rate != [] and float(edge_edit_rate[0]) or 0.
        if edit_rate < 0. or edit_rate > 1.: raise
    except:
        print_warning(params, 'Bad or truncated edge edit rate information!  Defaulting to 0')
        edit_rate = 0.
    try:
        growth_rate = edge_growth_rate != [] and float(edge_growth_rate[0]) or 0.
    except:
        print_warning(params, 'Bad or truncated edge growth rate information!  Defaulting to 0')
        growth_rate = 0.
    if verbose:
        print('  Edge rates: edit %f, growth %f'%(edit_rate,growth_rate))
    if G.number_of_nodes() == 0:
        if verbose:
            print('Num nodes = 0 ... editing canceled')
        return G

    new_edge_horizon   = params.get('new_edge_horizon', estimate_horizon(G))  #no edges added to nodes beyond the horizon
    if new_edge_horizon in params and nx.density(G) > 0.2 and G.number_of_nodes() > 500:
        print_warning(params, 'Warning: using a large horizon (%d) on a large graph might use a lot of time'%new_edge_horizon)

    if 'enforce_connected' in params:
        enforce_connected = params['enforce_connected']
    else:
        enforce_connected = tpl_data['enforce_connected']
    dont_cutoff_leafs = params.get('dont_cutoff_leafs', False)
    #do we allow leafs to be cut off completely?
    #   this option should be used sparingly, as it disrupts deferential detachment and decreases clustering

    all_nodes = G.nodes()
    added_edges_set = set()
    deled_edges_set = set()
    target_edges_to_delete = npr.binomial(max(G.number_of_edges(), 1), edit_rate)
    target_edges_to_add    = npr.binomial(max(G.number_of_edges(), 1), edit_rate)  #should be here, since NumEdges will change
    if growth_rate > 0:
        target_edges_to_add    += int(round(G.number_of_edges() * growth_rate))
    else:
        target_edges_to_delete += int(round(G.number_of_edges() * (-growth_rate)))

    deprived_nodes = [] #list of nodes that lost edges, including repetitions
    deferential_detachment_factor = params.get('deferential_detachment_factor', 0.0)
    d = []
    for degree in G.degree():
        d.append(degree[1])
    avg_degree = np.average(d)  #inexact deferential detachment, but with a much higher sampling efficiency
    num_deletion_trials           = params.get('num_deletion_trials', int(round(avg_degree**2)) )

    G_adj = G.adj
    G_degree    = lambda u: len(G_adj[u])
    G_neighbors = lambda u: list(G_adj[u].keys())
    for trial_num in range(max(20, num_deletion_trials*target_edges_to_delete)):
        if len(deled_edges_set) == target_edges_to_delete:
            break
        u = random.choice(list(all_nodes))
        degree_of_u = G_degree(u)
        if degree_of_u == 0: #will take care of this later
            continue
        #too random w = random.choice(G_neighbors(u))
        w = find_node_to_unfriend(G, head=u, params=params, existing_nbs=G_neighbors(u))
        if w == None:
            continue
        degree_of_w = G_degree(w)
        #perhaps a slight improvement is to multiply not by avg_degree but by avg_nb_degree
        if npr.rand()*deferential_detachment_factor > avg_degree/float(degree_of_u*degree_of_w):
            continue
        if dont_cutoff_leafs and (degree_of_u == 1 or degree_of_w == 1):
            continue
        #this improves clustering but it is a an unprincipled approach
        #if strong_clustering_structure(G, u, w, params):
        #    continue
        G.remove_edge(u,w)
        deled_edges_set.add((u,w))
        deprived_nodes += [u,w]

    if enforce_connected:
        new_edges = check_and_fix_connectivity(G, params)
        added_edges_set.update(new_edges)

    num_remaining_edges_to_add = target_edges_to_add - len(added_edges_set)
    edge_welfare_fraction = params.get('edge_welfare_fraction', 0.0)
    long_bridging         = params.get('long_bridging', False)
    #whether it should try to build edges to nodes which lost them;  not supported for all edges or for edges lost during node deletion

    for trial_num in range(max(20, 3*target_edges_to_add)):
        if num_remaining_edges_to_add <= 0:  #we might overshoot, hence <= 0 not ==
            break
        if npr.rand() > edge_welfare_fraction or len(deprived_nodes) == 0:
            #wishlist: avoid converting to list
            head = random.choice(list(all_nodes))
        else:
            head = random.choice(list(deprived_nodes))
        if G_degree(head) == 0 and tpl_data['chance_edge_prob'] == 0.0:
            continue
        tail = find_node_to_friend_hits(G=G, head=head, tpl_data=tpl_data, params=params, existing_nbs=G_neighbors(head))
        if tail == None or tail == head or G.has_edge(head,tail):
            continue
        #sys.stdout.write('%d,%.3f'%(G_degree(head), nx.clustering(G, head)))
        G.add_edge(head,tail)
        #sys.stdout.write(',%.3f\n'%nx.clustering(G, head))
        added_edges_set.add((head,tail))
        num_remaining_edges_to_add -= 1

    num_edges_added   = len(added_edges_set)
    num_edges_deleted = len(deled_edges_set)
    #print(num_edges_added, num_edges_deleted, G.number_of_edges())
    if num_edges_added > 20 and (num_edges_added-target_edges_to_add)/float(num_edges_added)> 0.2:
        print_warning(params, 'Warning: Excessive number of edges were added. Is the graph treelike (low AvgDegree and connected)? AvgDegree=%.1f.'%graphutils.a_avg_degree(G))
        #this might be caused by node edits.  in that case, try minorizing_node_deletion
    if num_edges_added > 20 and (target_edges_to_add-num_edges_added)/float(num_edges_added)> 0.2:
        print_warning(params, 'Warning: Excessive number of edges failed to add.   Consider setting locality_bias_correction to negative values.')
        if nx.density(G) > 0.6:
            print_warning(params, 'Is the graph too dense? Density=%.2f'%nx.density(G))
    if num_edges_deleted > 20 and abs(target_edges_to_delete-num_edges_deleted)/float(num_edges_deleted)> 0.2:
        print_warning(params, 'Warning: Excessive number of edges were deleted.')
        if nx.density(G) > 0.6:
            print_warning(params, 'Is the graph too dense? Density=%.2f'%nx.density(G))
    if verbose:
        print('\tadded edges: %d, deleted edges: %d'%(num_edges_added,num_edges_deleted))

    if 'edit_edges_tester' in params:
        params['edit_edges_tester'](G, added_edges_set, deled_edges_set, tpl_data)


    return G



def edit_nodes_sequential(G, node_edit_rate, node_growth_rate, tpl_data, params):
    verbose = params.get('verbose', True)
    if verbose:
        print('nn: %d'%G.number_of_nodes())
    try:
        edit_rate = node_edit_rate != [] and float(node_edit_rate[0]) or 0.
        if edit_rate < 0. or edit_rate > 1.: raise
    except:
        print_warning(params, 'Bad or truncated node edit rate information!  Defaulting to 0')
        edit_rate = 0.
    try:
        growth_rate = node_growth_rate != [] and float(node_growth_rate[0]) or 0.
    except:
        print_warning(params, 'Bad or truncated node growth rate information!  Defaulting to 0')
        growth_rate = 0.
    if verbose:
        print('  Node rates: edit %f, growth %f'%(edit_rate,growth_rate))
    if G.number_of_nodes() == 0:
        if verbose:
            print('Num nodes = 0 ... editing canceled')
        return G

    new_edge_horizon   = params.get('new_edge_horizon', estimate_horizon(G))  #no edges added to nodes beyond the horizon
    if new_edge_horizon in params and new_edge_horizon > 5 and nx.density(G) > 0.2 and G.number_of_nodes() > 500:
        print_warning(params, 'Warning: using a large horizon (%d) on a large graph might use a lot of time'%new_edge_horizon)

    num_deleted_nodes = npr.binomial(G.number_of_nodes(), edit_rate)
    num_added_nodes   = npr.binomial(G.number_of_nodes(), edit_rate)
    if growth_rate > 0:
        num_added_nodes   += int(round(G.number_of_nodes() * growth_rate))
    else:
        num_deleted_nodes += int(round(G.number_of_nodes() * (-growth_rate)))
    if num_deleted_nodes > G.number_of_nodes():
        print_warning(params, 'Warning: excess negative growth rate. Deletion of nodes will destroy all the nodes of the graph.  Editing aborted at this level.')
        return G

    G_adj = G.adj
    G_degree    = lambda u: G_adj[u].__len__()
    G_neighbors = lambda u: list(G_adj[u].keys())
    #we cache edges-to-add to avoid skewing these statistics during the editing process
    original_nodes = G.nodes()
    added_node_info = {}
    for i in range(num_added_nodes):
        source_node = random.choice(list(original_nodes))
        new_node    = new_node_label(G)
        added_node_info[new_node] = G_degree(source_node)
        #G.node[new_node]['resampling_source'] = source_node

    num_edges_added = 0
    num_edges_deleted = 0
    failed_searches = 0
    for new_node in added_node_info:
        G.add_node(new_node)
        num_remaining_nbs_to_add = added_node_info[new_node]
        if num_remaining_nbs_to_add == 0:
            continue
        #uncomment below and use 'enforce_connected':False to see the aggregates.  WARNING: comment back when done
        #continue

        anchor_node = random.choice(list(original_nodes))
        G.add_edge(new_node, anchor_node)
        num_edges_added += 1
        num_remaining_nbs_to_add -= 1
        for trial_num in range(max(40, 3*num_remaining_nbs_to_add)):
            if num_remaining_nbs_to_add == 0:
                break
            v = find_node_to_friend_hits(G=G, head=new_node, tpl_data=tpl_data, params=params, existing_nbs=G_neighbors(new_node))
            if v == None or v == new_node or v in G_neighbors(new_node):
                continue
            G.add_edge(new_node, v)
            num_edges_added += 1
            num_remaining_nbs_to_add -= 1
        if num_remaining_nbs_to_add > 0:
            failed_searches += 1
    added_nodes_set = set(added_node_info.keys())

    deled_nodes_set = set()
    minorizing_node_deletion = params.get('minorizing_node_deletion', False)
    for u in random.sample(G.nodes(), num_deleted_nodes):
        num_edges_deleted += G_degree(u)
        if minorizing_node_deletion: #connect the neighbors into a tree
            nbrs = G_neighbors(u)
            random.shuffle(nbrs)
            for nb_idx,nb in enumerate(nbrs[:-1]):
                new_edge = (nb, nbrs[nb_idx+1])
                if not G.has_edge(*new_edge):
                    #assert new_edge[0] != new_edge[1]
                    G.add_edge(*new_edge)
                    num_edges_added += 1
        G.remove_node(u)
        deled_nodes_set.add(u)

    if len(original_nodes) !=0 and len(original_nodes)>10 and (float(failed_searches)/len(original_nodes)) > .2:
        print_warning(params, 'Warning: > 20%% of searches failed when attempting to insert edges.')
        if nx.density(G) > 0.6:
            print_warning(params, 'Is the graph too dense? Density=%.2f'%nx.density(G))
    if verbose:
        print('\tadded nodes: %d, deleted nodes: %d'%(num_added_nodes,num_deleted_nodes))
        print('\tadded edges: %d, deleted edges: %d'%(num_edges_added,num_edges_deleted))

    if 'edit_nodes_tester' in params:
        params['edit_nodes_tester'](G, added_nodes_set, deled_nodes_set, tpl_data)

    return G

def estimate_horizon(G):
    density = nx.density(G)
    if density == 0 or G.number_of_nodes() < 3:
        return 4

    return 20

def find_node_to_friend_basic(G, head, tpl_data, params, existing_nbs=None):
    locality_acceptor = tpl_data['locality_acceptor']
    #implicit: chance_edge_prob  = tpl_data['chance_edge_prob']
    weighted_step = params.get('weighted_step', False)
    sm = sum
    rds = random.shuffle
    find_next = weighted_step_advanced

    all_nodes = None
    tail = None
    if existing_nbs == None:
        existing_nbs = set()
    else:
        existing_nbs = set(existing_nbs)
    G_adj = G.adj
    G_neighborsSet = lambda u: set(G_adj[u].keys())

    num_insertion_trials = params.get('num_insertion_trials', 30)
    num_insertion_searches_per_distance = params.get('num_insertion_searches_per_distance', 20)
    for trial in range(num_insertion_trials):
        #sample from np.random.multinomial()
        toss = npr.rand()
        for dis, prob in enumerate(locality_acceptor):
            toss -= prob
            if toss < 0:
                break
        for search_num in range(num_insertion_searches_per_distance):
            if toss < 0 and len(existing_nbs) > 0:
                cur_loc = find_next(G=G, start_node=head, weighted_step=weighted_step, blocked=(), rds=rds, sm=sm)
                blocked = set([head, cur_loc])
                next_loc = None
                tail = None
                for d in range(2, dis+1):
                    next_loc = find_next(G=G, start_node=cur_loc, weighted_step=weighted_step, blocked=blocked, rds=rds, sm=sm)
                    if next_loc == None:  #stuck in a self-made corner
                        break
                    #blocked.add(next_loc) #self-avoiding
                    cur_loc = next_loc
                if d == dis and next_loc != None:  #sanity tests ensure that no node "None" exists
                    tail = next_loc
                    #print 'tail %s at distance %d steps'%(tail,dis)
            else:
                for i in range(G.number_of_nodes()):
                    if all_nodes == None:
                        all_nodes = G.nodes()
                    candidate = random.choice(all_nodes)
                    if candidate != head and (candidate not in existing_nbs):
                        tail = candidate
                        break
                #print 'tail %s by RANDOM selection'%(tail,)
            if tail != None:
                return tail
    return tail


def find_node_to_friend_hits(G, head, tpl_data, params, existing_nbs=None):
    locality_acceptor = tpl_data['locality_acceptor']
    #implicit: chance_edge_prob  = tpl_data['chance_edge_prob']
    weighted_step = params.get('weighted_step', False)
    sm = sum
    rds = random.shuffle
    find_next = weighted_step_advanced

    all_nodes = None
    tail = None
    if existing_nbs == None:
        existing_nbs = set()
    else:
        existing_nbs = set(existing_nbs)
    G_adj = G.adj
    G_neighborsSet = lambda u: set(G_adj[u].keys())

    #num_insertion_trials = params.get('num_insertion_trials', 10 )
    num_insertion_searches_per_distance = params.get('num_insertion_searches_per_distance', 30)
    tail = None
    hits = {}

    if len(existing_nbs) == 0:
        return None
    #for trial in xrange(num_insertion_trials):
    most_hits           = -1
    most_hits_candidate = None
    if True:
        #sample from np.random.multinomial()
        toss = npr.rand()
        for dis, prob in enumerate(locality_acceptor):
            toss -= prob
            if toss < 0:
                break
        for search_num in range(num_insertion_searches_per_distance):
            cur_loc = find_next(G=G, start_node=head, weighted_step=weighted_step, blocked=(), rds=rds, sm=sm)
            blocked = set([head, cur_loc])
            #hits[cur_loc] = 1
            next_loc = None
            d = 2 #in case the loop is not even started
            for d in range(2, dis+1):
                cur_loc = find_next(G=G, start_node=cur_loc, weighted_step=weighted_step, blocked=blocked, rds=rds, sm=sm)
                if cur_loc == None:  #stuck in a self-made corner
                    break
            if d == dis and cur_loc != None:
                cur_loc_hits  = hits.get(cur_loc, 0) + 1
                hits[cur_loc] = cur_loc_hits
                if (cur_loc_hits >= most_hits) and (cur_loc not in existing_nbs):
                    most_hits           = cur_loc_hits
                    most_hits_candidate = cur_loc
                #sanity tests ensure that no node "None" exists
            #if d == dis and next_loc != None:  #sanity tests ensure that no node "None" exists
            #    tail = next_loc
            #    #print 'tail %s at distance %d steps'%(tail,dis)
    if most_hits_candidate != None:
        return most_hits_candidate

    hits = sorted(list(hits.items()), key=lambda item: item[1]-item[0])
    for candidate, h in hits:
        if candidate not in existing_nbs:
            return candidate
    return None


def find_node_to_unfriend(G, head, params, existing_nbs=None):
    if len(existing_nbs) == 0:
        return None
    else:
        return random_one(existing_nbs)

    #locality_acceptor = tpl_data['locality_acceptor']
    #implicit: chance_edge_prob  = tpl_data['chance_edge_prob']
    #weighted_step = params.get('weighted_step', False)
    #sm = sum  #maybe avoid a weighted walk?
    #rds = random.shuffle
    #find_next = weighted_step_advanced

    #tail = None
    #if existing_nbs == None:
    #    existing_nbs = set(existing_nbs)
    #G_adj = G.adj
    #G_neighborsSet = lambda u: set(G_adj[u].keys())
    #num_insertion_searches_per_distance = params.get('num_insertion_searches_per_distance', 30) #wishlist: use a separate parameter
    #tail = None
    #hits = {}

    #new_edge_horizon     = params.get('new_edge_horizon', estimate_horizon(G))
    #for search_num in xrange(num_insertion_searches_per_distance):
    #    cur_loc = find_next(G=G, start_node=head, weighted_step=weighted_step, blocked=(), rds=rds, sm=sm)
    #    blocked = set([head, cur_loc])
    #    for d in xrange(2, new_edge_horizon+1):
    #        cur_loc = find_next(G=G, start_node=cur_loc, weighted_step=weighted_step, blocked=blocked, rds=rds, sm=sm)
    #        if cur_loc == None:  #stuck in a self-made corner
    #            break
    #        #blocked.add(cur_loc) #self-avoiding
    #        if cur_loc in hits:
    #            hits[cur_loc] += 1
    #        else:
    #            hits[cur_loc] = 1
    ##print hits
    ##hits = hits.items()
    ##hits.sort(lambda x,y: y[1]-x[1])
    ##for candidate in hit_nodes:
    ##    if candidate in existing_nbs:
    ##        return candidate
    ##return None


def flush_graph(G):
    """
    the algorithm relabels the nodes of the graph at random
    -> data on aggregates becomes obsolete and is so never used
    """
    node_map = {}
    for node in G:
        while True:
            new_name = new_node_label(G)
            if (new_name not in G) and (new_name not in node_map):
                break
        node_map[node] = new_name
    G = nx.relabel_nodes(G, node_map, copy=True)
    return G


def generate_graph(original, params=None):
    """
    main entry point
    : returns replica graph
    """
    if params == None:
        params = {}
        print_warning(params, 'WARNING: empty parameter input. Running with default parameters.')

    if params.get('algorithm', False): #alternative algorithm, for testing purposes
        params2 = params.copy()
        alg_info = params2.pop('algorithm')
        if callable(alg_info):
            alg_method = alg_info
        elif type(alg_info) is str:
            alg_method = eval(alg_method)
        elif (type(alg_info) is list) or (type(alg_info) is tuple):
            alg_method           = alg_info[0]
            params2['algorithm'] = alg_info[1]
        else:
            raise ValueError ('algorithm parameter should be either callable, the name of a function, or (func,(nested_algorithm))')
        return alg_method(original=original, params=params2)

    validate_params(params)

    node_edit_rate   = params.get('node_edit_rate', [])
    edge_edit_rate   = params.get('edge_edit_rate', [])
    node_growth_rate = params.get('node_growth_rate', [])
    edge_growth_rate = params.get('edge_growth_rate', [])

    original._musketeer_data = {}
    G = original.copy()
    if params.get('verbose', True):
        sys.stdout.write('Checking original graph ... ')
        graphutils.graph_sanity_test(G, params)
        sys.stdout.write('Done.'+os.linesep)
    start_time = time.time()

    replica, model_map = revise_graph(G=G, level=0,
                                node_edit_rate=node_edit_rate,
                                node_growth_rate=node_growth_rate,
                                edge_edit_rate=edge_edit_rate,
                                edge_growth_rate=edge_growth_rate,
                                params=params)
    if params.get('verbose', True):
        print('replica is finished. nn: %d.  time: %.2f sec.'%(replica.number_of_nodes(), time.time()-start_time))
        print()

    replica      = resample_attributes(G, replica, model_map, params)
    replica.name = getattr(original, 'name', 'graph') + '_replica_' + timeNow()
    replica._musketeer_data = original._musketeer_data #WARNING shallow copy, to allow information to be passed from G to replicas

    graphutils.graph_sanity_test(replica, params)

    del G
    gc.collect()

    return replica



def interpolate_edges(G, c_data, model_map, fine_model_map, params):
    """
    constructs an "interior" of newly added edges
    """
    aggregates    = c_data['aggregates']
    merged_edges  = c_data['merged_edges']
    deep_copying  = params.get('deep_copying', True)
    if not deep_copying: assert len(model_map) == 0

    authentic_edges = list(merged_edges.items())
    edited_edges = []
    for (s1,s2) in G.edges(): #wishlist: faster loop?
        if ((s1,s2) not in merged_edges) and ((s2,s1) not in merged_edges):
            edited_edges.append((s1,s2))
    for (s1,s2) in edited_edges:
        trapped_in_s1 = aggregates[s1]
        trapped_in_s2 = aggregates[s2]

        model_aggregate_1 = model_map.get(s1, None)
        model_aggregate_2 = model_map.get(s2, None)
        new_pairs = set()
        if deep_copying:
            merged_model_edge_contents = merged_edges.get((model_aggregate_1, model_aggregate_2), None)
            if merged_model_edge_contents == None:
                merged_model_edge_contents = merged_edges.get((model_aggregate_2, model_aggregate_1), None)
        exact_interpolation = deep_copying and (merged_model_edge_contents != None)
        if exact_interpolation: #occurs only if s1 and s2 are deep-copied from an edge that existed in the original graph
            reversed_map = {}
            for trapped_node in list(trapped_in_s1) + list(trapped_in_s2):
                reversed_map[fine_model_map[trapped_node]] = trapped_node
            for mA, mB in merged_model_edge_contents:
                u = reversed_map[mA]
                v = reversed_map[mB]
                new_pairs.add((u,v))
        else:
            if authentic_edges != []:
                random_model_edge, random_model_edge_contents = random.choice(list(authentic_edges))
                num_target_edges = len(random_model_edge_contents)
            else:
                num_target_edges = 1
            num_failures = 0
            while len(new_pairs) < num_target_edges and num_failures < max(10, 3*num_target_edges):
                u = random.choice(list(trapped_in_s1))
                v = random.choice(list(trapped_in_s2))
                if ((u,v) not in new_pairs) and ((v,u) not in new_pairs):
                    new_pairs.add((u,v))
                else:
                    num_failures += 1
        merged_edges[(s1,s2)] = [(u,v) for u,v in new_pairs]
        #might optionally pass an attribute 'new' in the edges
    return c_data


def interpolate_nodes(G, c_data, model_map, params):
    """
    constructs the interior of nodes just added to the graph G, preparing it for uncoarsening
    interpolation is of two kinds: (node has just appeared in level i)                       -> we randomly select a "model node".  then we insert its internal structure from the existing graph.
                                   (deep copying: node was part of an aggregate in level i+) -> we refer to model_map to determine the source_node of its internal structure; we copy that structure;
    model_map:      u in Gi      -> v \in original_i
    fine_model_map: u in G_{i-1} -> v \in original_{i-1}
    at level i, fine_model_map contains information about level i-1, while model_map is the same information about
    """
    aggregates    = c_data['aggregates']
    trapped_edges = c_data['trapped_edges']
    home_nodes    = c_data['home_nodes']
    merged_edges  = c_data['merged_edges']
    if not params.get('deep_copying', True):
        assert len(model_map) == 0


    #this_level_node A -> node in G_i original of which is the model of A
    fine_model_map = {}

    authentic_nodes = list(aggregates.keys())
    assert authentic_nodes != []
    edited_nodes = [node for node in G if node not in aggregates]
    num_new_nodes = 0
    num_new_edges = 0
    for node in edited_nodes:
        source_aggregate = model_map.get(node, random.choice(list(authentic_nodes)))
        sources_edges    = trapped_edges[source_aggregate]
        sources_nodes    = aggregates[source_aggregate]

        renamed_nodes = {}
        for node_hosted_by_source in sources_nodes:
            new_hosted_node                        = new_node_label(home_nodes)
            renamed_nodes[node_hosted_by_source]   = new_hosted_node
            fine_model_map[new_hosted_node]        = node_hosted_by_source
            num_new_nodes += 1

        my_trapped_nodes = list(renamed_nodes.values())

        my_trapped_edges = []
        for edge_hosted_by_source in sources_edges:
            my_trapped_edges.append( (renamed_nodes[edge_hosted_by_source[0]], renamed_nodes[edge_hosted_by_source[1]]) )
            num_new_edges += 1

        aggregates[node]    = my_trapped_nodes
        trapped_edges[node] = my_trapped_edges
        for new_hosted_node in my_trapped_nodes:
            home_nodes[new_hosted_node] = node
        #might optionally pass an attribute 'new' in the node
        #num_added_nodes += len(my_trapped_nodes)
    if params.get('verbose', True):
        print('  from new aggregates: %d nodes, %d edges'%(num_new_nodes,num_new_edges))
    #print 'added: %d'%num_added_nodes

    if params.get('deep_copying', True):
        return c_data, fine_model_map
    else:
        return c_data, {}


def musketeer_on_subgraphs(original, params=None):
    """
    special case for disconnected graph
    """
    components = nx.connected_component_subgraphs(original)
    merged_G   = nx.Graph()

    component_is_edited = params.get('component_is_edited', [True]*len(components))

    for G_num, G in enumerate(components):
        if component_is_edited[G_num]:
            replica = generate_graph(original=G, params=params)
        else:
            replica = G

        merged_G = nx.union(merged_G, replica)

    merged_G.name = getattr(original, 'name', 'graph') + '_replica_' + timeNow()

    return merged_G

def musketeer_snapshots(original, params=None):
    """
    applies replication sequentially, generating snapshots of the original.
    returns the final snapshot.
    snapshots (0 to last) are held in the .snapshot attribute
    """
    graphs = [original]

    num_snapshots = params['num_snapshots']

    for graph_num in range(num_snapshots):
        G = graphs[-1]
        replica = generate_graph(original=G, params=params)
        replica.name = 'snapshot_%d'%graph_num
        graphs.append(replica)

    if params.get('verbose', True):
        print('Snapshots complete.')
        print()

    replica.snapshots = graphs

    return replica


def musketeer_iterated_cycle(original, params=None):
    """
    applies replication sequentially, and returns the final snapshot.
    """
    num_cycles = params['num_v_cycles']

    params2 = params.copy()
    params2['edge_edit_rate']   = [r/float(num_cycles) for r in params2.get('edge_edit_rate', [])]
    params2['edge_growth_rate'] = [r/float(num_cycles) for r in params2.get('edge_growth_rate', [])]
    params2['node_edit_rate']   = [r/float(num_cycles) for r in params2.get('node_edit_rate', [])]
    params2['node_growth_rate'] = [r/float(num_cycles) for r in params2.get('node_growth_rate', [])]

    replica = original
    for graph_num in range(num_cycles):
        replica = generate_graph(original=replica, params=params2)
        replica.name = getattr(original, 'name', 'graph') + '_replica_w%d_'%graph_num + timeNow()

    return replica


def new_node_label(G):
    """
    constructs a new label for a node
    """
    #G is either a graph or a dict/list of existing labels
    num_trials = 100
    label = None
    for t in range(num_trials):
        label = npr.randint(max_int)
        if label not in G:
            break
    if label == None:
        raise Exception ('Could not find a unique label for a newly-added node')
    return label

def print_warning(params, str):
    if not params.get('suppress_warnings', False):
        print(str)

def resample_attributes(G, replica, model_map, params):
    """
    inserts attributes to new nodes and edges by COPYING data from existing nodes and edges
    note that to save the data, the replica should be saved as particular formats (e.g. gml, dot, edgelist[for edges only])
    """
    maintain_node_attributes = params.get('maintain_node_attributes', False)
    maintain_edge_attributes = params.get('maintain_edge_attributes', False)
    deep_copying             = params.get('deep_copying', True)

    original_nodes = G.nodes()
    G_adj = G.adj
    G_degree    = lambda u: G_adj[u].__len__()
    G_neighbors = lambda u: list(G_adj[u].keys())
    if maintain_node_attributes:
        if params.get('verbose', True):
            print('Updating node attributes ...')
        for node in replica:
            pdb.set_trace()
            if replica.node[node] != {}:
                continue
            elif node in G:
                replica.node[node] = G.node[node].copy()
            elif deep_copying:
                model_node = model_map.get(node, random.choice(original_nodes))
                replica.node[node] = G.node[model_node].copy()
            else:
                model_node = random.choice(original_nodes)
                replica.node[node] = G.node[model_node].copy()

    if maintain_edge_attributes and G.number_of_edges() > 0:
        if params.get('verbose', True):
            print('Updating edge attributes ...')
        for edge in replica.edges():
            if replica.get_edge_data(*edge) != {}:
                continue
            elif G.has_edge(*edge):
                edata = G.get_edge_data(*edge).copy()
                replica.add_edges_from([(edge[0], edge[1], edata)])
                replica.add_edges_from([(edge[1], edge[0], edata)])
            else:
                modelA = model_map.get(edge[0], None)
                modelB = model_map.get(edge[1], None)
                if deep_copying and G.has_edge(modelA, modelB):
                    nodeA, nodeB = modelA, modelB
                else:
                    for trial in range(G.number_of_edges()):
                        nodeA = random.choice(original_nodes)
                        if G_degree(nodeA) == 0:
                            continue
                        nodeB = random.choice(G_neighbors(nodeA))
                        break

                edata = G.get_edge_data(nodeA, nodeB).copy()
                replica.add_edges_from([(edge[0], edge[1], edata)])
                replica.add_edges_from([(edge[1], edge[0], edata)])

    return replica

def revise_graph(G, level, node_edit_rate, node_growth_rate, edge_edit_rate, edge_growth_rate, params):
    """
    revises a graph at a particular resolution and deeper
    """
    no_more_coarse = len(node_edit_rate) < 2 and len(edge_edit_rate) < 2 and len(node_growth_rate) < 2 and len(edge_growth_rate) < 2
    excess_density = G.number_of_edges() == 0 or nx.density(G) > params.get('coarsening_density_limit', 0.9)
    if no_more_coarse or excess_density:
        if excess_density:
            if params.get('verbose', True):
                print('Coarsening stopped due to excess density')
                if not no_more_coarse:
                    print('Editing at deeper levels is impossible.  CHANGE editing parameters.')
        if params.get('verbose', True):
            print('Final coarsening level. nodes: %d, edges: %d'%(G.number_of_nodes(), G.number_of_edges()))
            print('---------------------------------------------')
        if params.get('memoriless_interpolation', False):
            G_prime = flush_graph(G)
        else:
            G_prime = G
        fine_model_map = {}
        G_prime.coarser_graph = None
        coarse = None
        #return G, {}
    else:
        coarse, c_data               = do_coarsen(G, params)
        coarse, model_map            = revise_graph(coarse, level+1, node_edit_rate[1:], node_growth_rate[1:], edge_edit_rate[1:], edge_growth_rate[1:], params)
        c_data_prime                 = copy.deepcopy(c_data)  #we are now going to start playing with it;  wishlist: support injection
        c_data_prime, fine_model_map = interpolate_nodes(coarse, c_data_prime, model_map, params)  #must precede edges, since otherwise cannot find an attachment for some edges
        c_data_prime                 = interpolate_edges(coarse, c_data_prime, model_map, fine_model_map, params)
        clean_c_data(coarse, c_data_prime)
        G_prime                      = do_uncoarsen(coarse, c_data_prime, params)

    #editing of the current level
    tpl_data = compute_topology_data(G, level, params)

    G_prime  = edit_nodes_sequential(G_prime, node_edit_rate, node_growth_rate, tpl_data, params)
    G_prime  = edit_edges_sequential(G_prime, edge_edit_rate, edge_growth_rate, tpl_data, params)

    if 'revise_graph_tester' in params:
        params['revise_graph_tester'](G, G_prime, c_data_prime)
    if params.get('retain_intermediates', False):
        G_prime.model_graph, G_prime.coarser_graph = G, coarse
    else:
        G_prime.model_graph, G_prime.coarser_graph = None, None

    return G_prime, fine_model_map


def seed_finder_matching(G, params):
    """
    uses a matching algorithm.
    the algorithm must (1) return a dictionary of nodes u->v [iff v->u] (2) that consistitue a maximal matching
    """
    seeds         = set()
    aggregates    = {}    #nodes within new nodes.  seed->fine_nodes
    home_nodes    = {}    #the reverse map: node->seed
    #free_nodes    = set(G.nodes())

    #invalid code
    #matching_algorithm  = params.get('matching_algorithm', nx.maximal_matching)  #random matching - returns a set of edges

    #all the matching algorithms return a dictionary u->v and v->u
    if not G.is_directed():
        matching_algorithm  = params.get('matching_algorithm', graphutils.drake_hougardy)
        #matching_algorithm  = params.get('matching_algorithm', nx.max_weight_matching)
    else:
        matching_algorithm  = params.get('matching_algorithm', graphutils.drake_hougardy_flattened)

    matching = matching_algorithm(G)

    for nodeA in matching:
        nodeB = matching[nodeA]
        assert G.has_edge(nodeA, nodeB)
        #wishlist: choose A or B by maximal weight of the neighbors?
        if nodeA in seeds or nodeB in seeds:
            continue
        seeds.add(nodeA)
        aggregates[nodeA] = [nodeA, nodeB]
        home_nodes[nodeB] = nodeA #sic
        home_nodes[nodeA] = nodeA #sic

    for node in G: #sweep for missed nodes
        if node in matching:
            continue
        seeds.add(node)
        aggregates[node] = [node]
        home_nodes[node] = node

    return seeds, home_nodes, aggregates


def seed_finder_weight_alg(G, params):
    """
    coarsening inspired by Safro et al.
        Graph minimum linear arrangement by multilevel weighted edge contractions
    1. nodes have weights that represent nodes previously included in them
    2a. nodes with high expected weight (>= 2 of average) automatically become seeds
        expected weight = self weight + total weight of neighbors
    2b. we go over the remaining nodes in order of decreasing future weight
        and make them seeds when (sum of neighbor expect_wts who are seeds) / (sum over all nbs) < 0.4
    3. the remaining nodes are assigned to a neighboring seeds that has the greatest wt
    """

    seeds         = set()
    aggregates    = {}    #nodes within new nodes.  seed->fine_nodes
    home_nodes    = {}    #the reverse map: node->seed
    free_nodes    = set(G.nodes())
    seed_threshold_1 = params.get('seed_threshold_1', 2.0)
    seed_threshold_2 = params.get('seed_threshold_2', 0.4)

    G_adj = G.adj
    G_degree    = lambda u: G_adj[u].__len__()
    G_neighbors = lambda u: list(G_adj[u].keys())
    expected_wt_dict = {}
    G_node_weights = nx.get_node_attributes(G, 'weight')
    for node in G:
        if G_degree(node) == 0: #singletons mess up seed finding by pulling down the average wt, but then become seeds themselves
            seeds.add(node)
        else:
            expected_wt_dict[node] = G_node_weights.get(node, 1.) + sum(G_node_weights.get(nb, 1.) for nb in G_neighbors(node))

    if G.number_of_edges() > 0:
        mean_exp_wt = np.fromiter(expected_wt_dict.values(), float).mean()
    else:
        mean_exp_wt = 0.

    expected_wt = list(expected_wt_dict.items())
    expected_wt.sort(key=lambda x:x[1], reverse=True)
    for node,wt in expected_wt:
        if wt > seed_threshold_1*mean_exp_wt:
            seeds.add(node)
        elif sum(expected_wt_dict[nb] for nb in seeds.intersection(G_neighbors(node)))/sum(expected_wt_dict[nb] for nb in G_neighbors(node)) < seed_threshold_2:
            seeds.add(node)
    free_nodes.symmetric_difference_update(seeds)
    for seed in seeds:
        aggregates[seed] = [seed]
        home_nodes[seed] = seed

    all_edges = G.edges()
    for node in free_nodes:
        nearby_seeds = seeds.intersection(G_neighbors(node))
        seed_attraction = [(seed,all_edges[(node,seed)].get('weight', 1.0)) for seed in nearby_seeds]
        my_aggregate = max(seed_attraction, key=lambda x:x[1])[0]
        aggregates[my_aggregate].append(node)
        home_nodes[node] = my_aggregate

    return seeds, home_nodes, aggregates


def strong_clustering_structure(G, u, w, params):
    """
    detects strong clustering around (u,w) and decides whether to skip deletion of the edge
    """
    if not params.get('preserve_clustering_on_deletion', False):  #wishlist: make this a parameter from 0 to 1
        return False

    num_mutual_nbs = len(set(G.neighbors(u)).intersection(G.neighbors(w)))
    if num_mutual_nbs == 0:
        return False
    elif npr.rand() * num_mutual_nbs < 0.5:
        return False
    else:
        return True



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
                print('Hint: for a list of valid parameters, see valid_params variable')
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


def weighted_step_advanced(G, start_node, weighted_step, blocked, rds, sm=None):
    nb_wt = list(G[start_node].items())
    if not weighted_step:
        #TODO: in most cases, we are fine just doing 10 random selections, and then giving up - we would just not guarantee that None means fully blocked
        l = list(nb_wt)
        rds(l)  #shuffling is O(n); selection is O(1)
        nb_wt = dict(l)
        for nb,wt in list(nb_wt.items()):
             if nb not in blocked:
                 return nb
        return None
    total_wt = 0.
    for idx, (nb,nb_data) in enumerate(nb_wt):
        if nb in blocked:
            nb_wt[idx] = (nb,0.)
            continue
        try:
            wt = nb_data['weight']
        except:
            wt = 1.
        nb_wt[idx] = (nb,wt)
        total_wt += wt
    if total_wt == 0.:
        return None
    draw = npr.rand() * total_wt
    #draw *= npr.rand() * sm(wt for nb,wt in nb_wt)
    for nb,wt in nb_wt:
        draw -= wt
        if draw < 0:
            return nb
    return nb_wt[-1][0]

