# -*- coding: utf-8 -*-
'''
Multiscale Entropic Network Generator 2 (MUSKETEER2)

Copyright (c) 2011-2018 by Alexander Gutfraind and Ilya Safro.
All rights reserved.

Use and redistribution of this file is governed by the license terms in
the LICENSE file found in the project's top-level directory.

Code to assist the generator

'''

'''
Contains code derived from NetworkX - Copyright (c) by various authors.
'''

import os
import time
import numpy as np
import scipy.sparse
import numpy.random as npr
import random, sys
import networkx as nx
import pdb
import cPickle
import algorithms
import community

np.seterr(all='raise')

timeNow = lambda : time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())

MUSKETEER_EXAMPLE_CMD = 'python musketeer.py -p "{\'edge_edit_rate\':[0.1,0.01]}" -f data-samples/karate.adjlist -t adjlist -o output/karate_replica.edges'

None_node = None  #key to indicate a not real node.

#those are short methods (like lambda, but supporting pickling)
def a_avg_degree(G):
    return np.average(nx.degree(G).values())
def a_degree_connectivity(G):
    return np.average(nx.average_degree_connectivity(G).values())
def a_s_metric(G):
    return nx.s_metric(G, normalized=False)
def a_eccentricity(G):
    return np.average(nx.eccentricity(G).values())
def a_avg_shortest(G):
    return average_all_pairs_shortest_path_estimate(G, max_num_sources=100)
def a_avg_harmonic(G):
    return average_all_pairs_inverse_shortest_path_estimate(G, max_num_sources=100)
def a_avg_between(G):
    return np.average(nx.betweenness_centrality(G, normalized=True).values())

METRIC_ERROR = -999999
#any negative value is suitable, but not None or -np.inf (might complicate code for statistical analysis)

def average_all_pairs_shortest_path(G):
    if nx.number_connected_components(G)>1:
        return METRIC_ERROR
    true_d = nx.all_pairs_shortest_path_length(G)
    nx_sum = 0.0
    reachable_pairs = 0.0
    for key1 in true_d:
      for key2 in true_d[key1]:
        nx_sum += true_d[key1][key2]
        reachable_pairs += 1.0
    nx_mean_distance = nx_sum/reachable_pairs
    return nx_mean_distance

def average_all_pairs_shortest_path_GT(G):
    if nx.number_connected_components(G)>1:
        return METRIC_ERROR

    import graph_tool, graph_tool.topology
    gtG = make_gt_graph(G)
    #gtDistances = graph_tool.topology.shortest_distance(g=gtG['G'], source=None, target=None, weights=gtG['edge_weights'], dense=True)
    gtDistances = graph_tool.topology.shortest_distance(g=gtG['G'], source=None, target=None, dense=False)
    total_distance = 0.
    for u in gtG['G'].vertices():
        from_u = gtDistances[u].a
        total_distance += sum(from_u)
    nn = gtG['G'].num_vertices()
    gt_mean_distance = total_distance / (nn * nn)

    return gt_mean_distance

def average_all_pairs_shortest_path_estimate(G, max_num_sources=100):
    if nx.number_connected_components(G)>1:
        return METRIC_ERROR

    num_sources = min(G.number_of_nodes(), max_num_sources)
    baseline_sources = []
    if hasattr(G, '_musketeer_data'):
        if 'all_pairs_shortest_path_estimate_SOURCES' in G._musketeer_data:
            baseline_sources = G._musketeer_data['all_pairs_shortest_path_estimate_SOURCES']
            baseline_sources = filter(lambda s: s in G, baseline_sources)
    else:
        G._musketeer_data = {}
    sampled_sources = baseline_sources + random.sample(G.nodes(), num_sources - len(baseline_sources))
    G._musketeer_data['all_pairs_shortest_path_estimate_SOURCES'] = sampled_sources
    total_distance = 0.0
    for source in sampled_sources:
        lengths = nx.single_source_shortest_path_length(G, source)
        total_distance += np.average(lengths.values())
    total_distance /= num_sources
    return total_distance

def average_all_pairs_shortest_path_estimate_GT(G, max_num_sources=100):
    if nx.number_connected_components(G)>1:
        return METRIC_ERROR

    import graph_tool, graph_tool.topology
    gtG = make_gt_graph(G)
    num_sources = min(G.number_of_nodes(), max_num_sources)
    cum_distance = 0.0
    for source in random.sample(G.nodes(), num_sources):
        gt_source     = gtG['G'].vertex(gtG['node_to_num'][source])
        gtDistances   = graph_tool.topology.shortest_distance(g=gtG['G'], source=gt_source, target=None, dense=False)
        cum_distance += np.average(gtDistances.a)
    cum_distance /= num_sources
    return cum_distance

def average_all_pairs_inverse_shortest_path_estimate(G, max_num_sources=100):
#estimates the ''efficiency'' of the graph: the harmonic mean of the distances
#this is well-defined even in disconnected graphs
    if G.number_of_edges()<1:
        return METRIC_ERROR

    num_sources = min(G.number_of_nodes(), max_num_sources)
    baseline_sources = []
    if hasattr(G, '_musketeer_data'):
        if 'all_pairs_inverse_shortest_path_estimate_SOURCES' in G._musketeer_data:
            baseline_sources = G._musketeer_data['all_pairs_inverse_shortest_path_estimate_SOURCES']
            baseline_sources = filter(lambda s: s in G, baseline_sources)
    else:
        G._musketeer_data = {}
    sampled_sources = baseline_sources + random.sample(G.nodes(), num_sources - len(baseline_sources))
    G._musketeer_data['all_pairs_inverse_shortest_path_estimate_SOURCES'] = sampled_sources
    tally = 0.0
    for source in sampled_sources:
        lengths = nx.single_source_shortest_path_length(G, source)
        tally += sum([1.0/lengths[node] for node in lengths if node!=source])
    tally = num_sources*(1.0/tally)
    return tally

def average_flow_closeness(G):
  if nx.number_connected_components(G)>1:
    return METRIC_ERROR
  else:
    length=nx.algorithms.current_flow_closeness_centrality(G)
    sum = 0.0
    count = 0.0
    for key1 in length.keys():
      sum = sum + length[key1]
      count = count + 1
    return sum/count

def average_eigenvector_centrality(G):
  #warning: this algorithm might not be suitable for disconnected graphs, since it creates additional zero eigenvalues
  if nx.number_connected_components(G)>1:
    return METRIC_ERROR
  else:
    length=nx.algorithms.eigenvector_centrality(G, 500, 0.0001)
    sum = 0.0
    count = 0.0
    for key1 in length.keys():
      sum = sum + length[key1]
      count = count + 1
    return sum/count

def algebraic_distance_dense(G, params={}):
    '''
    takes: graph G, computational parameters

    returns:
    a distance dictionary, d[node1][node2]  giving the distance between the nodes

    ref:
            RELAXATION-BASED COARSENING AND
            MULTISCALE GRAPH ORGANIZATION
            DORIT RON, ILYA SAFRO, AND ACHI BRANDT
    '''

    metric              = params.get('metric', 'Linfinity')
    num_relaxations_r   = params.get('num_relaxations_r', 10)
    num_test_vectors_K  = params.get('num_test_vectors_K', 20)
    lazy_walk_param_w   = params.get('lazy_walk_param_w', 0.3)

    if metric != 'Linfinity':
        raise Exception('Metric other than Linifinity not implemented')

    #singletons = filter(lambda u: G.degree(u)==0, G.nodes())
    #if len(singletons) == 0:
    #    Gcopy = G
    #else:
    #    Gcopy = G.copy()
    #    Gcopy.remove_nodes_from(singlestons)
    #H = nx.convert_node_labels_to_integers(Gcopy, label_attribute='orig_label')
    H = nx.convert_node_labels_to_integers(G, label_attribute='orig_label')
    #for singletons, we set diag=1
    all_nodes = H.nodes()  #future: use this for sorting

    distance = {}
    for node1 in all_nodes:
        distance[node1] = {}
        for node2 in H.neighbors(node1):
            if node1 < node2: #save time
               distance[node1][node2] = -np.inf

    #wishlist: sparse matrices
    #the wrong laplacian (uses degrees) LAP      = nx.laplacian_matrix(H)
    #should consider degree? diag_vec     = np.array([H.node[u].get('weight', 1) for u in H])
    #diag_vec     = np.array([H.degree(u) for u in H])
    diag_vec = []
    for u in all_nodes:
        val = 0. + sum(edge.get('weight', 1.) for edge in H.edge[u].values())
        if val == 0.:
            val == 1.
        diag_vec.append(val)
    diag_vec     = np.array(diag_vec)
    diag_vec_inv = 1./diag_vec  #[1./val for val in diag_vec]
    DIAG     = np.diag(diag_vec)
    DIAGinv  = np.diag(diag_vec_inv)
    ADJ      = nx.adj_matrix(G, nodelist=all_nodes)
    w_times_Dinv_times_LAP = lazy_walk_param_w * np.dot(DIAGinv,DIAG-ADJ)

    for t in xrange(num_test_vectors_K):
        x = npr.rand(H.number_of_nodes(),1) - 0.5
        x = x / x.sum()

        for iteration in xrange(num_relaxations_r):
            x = (1-lazy_walk_param_w)*x + np.dot(w_times_Dinv_times_LAP, x)

        #maximize over the trial vectors: d(i,j) = max_{t=1..K} |x_t(i) - x_t(j)|
        for node1 in all_nodes:
            for node2 in H.neighbors(node1):
                dis = np.abs((x[node1]-x[node2])[0,0])
                if node1 < node2 and dis > distance[node1][node2]: #to save time, compute just the upper triangle of the matrix
                    distance[node1][node2] = dis


    #generate the distance dictionary in the original node labels, and including the diagonal and lower triangle
    ret = {}
    for node1 in G:
        ret[node1] = {node1:0.}
    for u in H:
        node1 = H.node[u]['orig_label']
        ret[node1] = {}
        for v in H.neighbors(u):
            node2 = H.node[v]['orig_label']
            if u < v:
                d = distance[u][v]
            elif v < u:
                d = distance[v][u]
            else:
                d = 0.
            ret[node1][node2] = d
            ret[node2][node1] = d
    return ret


def algebraic_distance_sparse(G, params={}):
    '''
    takes: graph G, computational parameters

    returns:
    a distance dictionary, d[node1][node2]  giving the distance between the nodes

    ref:
            RELAXATION-BASED COARSENING AND
            MULTISCALE GRAPH ORGANIZATION
            DORIT RON, ILYA SAFRO, AND ACHI BRANDT
    '''

    metric              = params.get('metric', 'Linfinity')
    num_relaxations_r   = params.get('num_relaxations_r', 10)
    num_test_vectors_K  = params.get('num_test_vectors_K', 50)
    lazy_walk_param_w   = params.get('lazy_walk_param_w', 0.5)

    singletons = filter(lambda u: G.degree(u)==0, G.nodes())
    if len(singletons) == 0:
        Gcopy = G
    else:
        Gcopy = G.copy()
        Gcopy.remove_nodes_from(singlestons)
    H = nx.convert_node_labels_to_integers(Gcopy, label_attribute='orig_label')  #this does preserve node and edge weights

    if metric != 'Linfinity':
        raise Exception('Metric other than Linifinity is not implemented')

    distance = {}
    for node1 in H:
        distance[node1] = {}
        for node2 in H:
            if node1 < node2: #save time
               distance[node1][node2] = -np.inf

    nn = H.number_of_nodes()
    #wishlist: sparse matrices
    ##LAP      = nx.laplacian_matrix(H)
    #diag_vec = [H.node['weight'] for node in H]
    #diag_vec     = np.array([H.node[u].get('weight', 1) for u in H])  #should consider degree?
    diag_vec     = np.array([H.degree(u) for u in H])
    diag_vec_inv = 1./diag_vec  #[1./val for val in diag_vec]
    full_range = range(nn)
    DIAG     = scipy.sparse.csr_matrix((diag_vec,     (full_range,full_range)))
    DIAGinv  = scipy.sparse.csr_matrix((diag_vec_inv, (full_range,full_range)))
    ADJ      = nx.to_scipy_sparse_matrix(G, nodelist=non_singletons, format='csr')
    #w_times_Dinv_times_LAP = lazy_walk_param_w * np.dot(np.diag([1./el for el in diag_vec]),DIAG-LAP)
    w_times_Dinv_times_LAP = lazy_walk_param_w * DIAGinv.dot(DIAG-ADJ)

    for t in xrange(num_test_vectors_K):
        x = npr.rand(H.number_of_nodes(),1)

        for iteration in xrange(num_relaxations_r):
            x = (1-lazy_walk_param_w)*x + w_times_Dinv_times_LAP.dot(x)

        #maximize over the trial vectors: d(i,j) = max_{t=1..K} |x_t(i) - x_t(j)|
        for node1 in H:
            for node2 in (H if all_pairs_distance else H.neighbors(node1)):
                dis = np.abs((x[node1]-x[node2])[0,0])
                if node1 < node2:
                    old_dis = distance[node1][node2]
                    if dis > old_dis: #to save time, compute just the upper triangle of the matrix
                        distance[node1][node2] = dis


    #generate the distance dictionary in the original node labels, and including the diagonal and lower triangle
    ret = {}
    for node1 in G:
        ret[node1] = {node1:0.} #important for singletons
    for u in H:
        node1 = H.node[u]['orig_label']
        ret[node1] = {node1:0.}
        for v in (H if all_pairs_distance else H.neighbors(u)):
            node2 = H.node[v]['orig_label']
            if u < v:
                d = distance[u][v]
            elif v < u:
                d = distance[v][u]
            else:
                d = 0.
            ret[node1][node2] = d
            ret[node2][node1] = d
    return ret
algebraic_distance = algebraic_distance_dense
#algebraic_distance = algebraic_distance_sparse

def bfs_distance_with_horizon(G, source, horizon=4, blocked_node=None):
#computes distance from every node to every neighbor at distance at most <horizon> hops
#    nodes further away are considered infinitely away
#no path is allowed through blocked_node
    G_adj = G.adj
    G_neighbors = lambda u: G_adj[u].keys()

    fringe = set(G_neighbors(source))
    distance_source  = {source:0}
    for d in xrange(1, horizon+1):
        new_fringe = []
        for v in fringe:
            if v not in distance_source and v!=blocked_node:
                distance_source[v] = d
                new_fringe += G_neighbors(v)
        fringe = set(new_fringe)

    return distance_source


def color_by_3d_distances(G, verbose):
    import matplotlib.pylab as pylab
    #cm=pylab.get_cmap('Paired')
    #cm=pylab.get_cmap('gist_rainbow')
    cm=pylab.get_cmap('RdBu')  #UFL

    if verbose:
        print 'Computing edge colors ...'
    max_dis = 0
    positions = {}
    for u,v,data in G.edges_iter(data=True):
        try:
            u_pos = positions[u]
        except:
            #u_pos = np.array([float(p) for p in G.node[u]['pos'][1:-1].split(',')])
            u_pos = np.array([float(p) for p in G.node[u]['pos'].split(',')])
            positions[u] = u_pos
        try:
            v_pos = positions[v]
        except:
            #v_pos = np.array([float(p) for p in G.node[v]['pos'][1:-1].split(',')])
            v_pos = np.array([float(p) for p in G.node[v]['pos'].split(',')])
            positions[v] = v_pos

        dis = np.sqrt(np.sum(np.power(u_pos-v_pos,2)))
        max_dis = max(max_dis, dis)

        data['dis'] = dis

    for u,v,data in G.edges_iter(data=True):
        dis = data.pop('dis')
        #data['color'] = '"%.3f %.3f %.3f"'%tuple(cm(dis/max_dis)[:3])
        data['color'] = '%.3f %.3f %.3f'%tuple(cm(dis/max_dis)[:3])
        #data['weight'] = 1.0

    return G


def color_new_nodes_and_edges(G, original, params=None):
#add red color to new components.
#use the option 'post_processor':graphutils.color_new_nodes_and_edges
    for node in G:
        G.node[node]['label'] = ''
        #d['style'] = 'filled'
        if node in original:
            G.node[node]['color']='black'
        else:
            G.node[node]['color']='blue'
    for u,v,d in G.edges_iter(data=True):
        if original.has_edge(u,v):
            d['color']='black'
        else:
            d['color']='blue'

    return G

def compare_nets(old_G, new_G, metrics=None, params={}):
    '''
    Report on the differences between two networks
    '''
    if metrics == None:
        metrics = default_metrics
    verbose   = params.get('verbose', True)
    runningtime_bound   = params.get('metric_runningtime_bound', 2)

    precision   = params.get('reporting_precision', 2)
    #formatstring = '\t%.'+str(precision)+'f\t%.'+str(precision)+'f\t%.'+str(precision)+'f%%'

    #TODO: at the moment, graph_graph_delta cannot find edges which were deleted then inserted back: it changes the edge attribute data
    errors = {}
    if verbose:
        delta = graph_graph_delta(old_G, new_G)
        num_changed_nodes = len(delta['new_nodes']) + len(delta['del_nodes'])
        num_changed_edges = len(delta['new_edges']) + len(delta['del_edges'])
        if old_G.number_of_nodes() > 0:
            print 'New or deleted Nodes: %d (%.1f%%)'%(num_changed_nodes, 100*float(num_changed_nodes)/old_G.number_of_nodes())
            print 'New or deleted Edges: %d (%.1f%%)'%(num_changed_edges, 100*float(num_changed_edges)/old_G.number_of_edges())
            print
        print 'Name\t\t\tOld G\t\tNew G\t\tRelative Error'
        print 'statistics start ------------------------------------------------------------'
    for met_info in metrics:
        met_name = met_info['name']
        met_func = met_info['function']
        met_wt = met_info['weight']
        if met_info['optional'] > 0 or met_info['runningtime'] > runningtime_bound:
            continue
        try:
            if verbose:
                sys.stdout.write(met_name.center(20))
                sys.stdout.flush()
            old_value = met_func(old_G)
            new_value = met_func(new_G)
            if old_value != 0. and abs(old_value-METRIC_ERROR) > 1 and abs(new_value-METRIC_ERROR) > 1:
                error = met_wt*float(new_value-old_value)/old_value
            else:
                error = np.NaN
            if verbose:
                outstr = ''
                if np.abs(old_value) < 0.1 or np.abs(old_value) > 1000:
                    outstr += ('\t%.'+str(precision)+'e')%old_value
                else:
                    outstr += ('\t%.'+str(precision)+'f    ')%old_value
                if np.abs(new_value) < 0.1 or np.abs(new_value) > 1000:
                    outstr += ('\t%.'+str(precision)+'e')%new_value
                else:
                    outstr += ('\t%.'+str(precision)+'f    ')%new_value
                outstr += ('\t%.'+str(precision)+'f%%')%(100*error)
                print outstr
                #print formatstring%(old_value,new_value,100*error)
            errors[met_name] = (old_value, new_value, error)
        except Exception,inst:
            print
            print 'Warning: could not compute '+met_name + ': '+str(inst)
    mean_error = np.average([np.abs(v[2]) for v in errors.values() if (v[2]!=np.NaN) and abs(v[2]-METRIC_ERROR) > 1])
    if verbose:
        print 'statistics end ------------------------------------------------------------'
        print 'Mean absolute difference: %.2f%%'%(100*mean_error)

    return mean_error, errors

def degree_assortativity(G):
#this wrapper helps avoid error due to change in interface name
    if hasattr(nx, 'degree_assortativity_coefficient'):
        return nx.degree_assortativity_coefficient(G)
    elif hasattr(nx, 'degree_assortativity'):
        return nx.degree_assortativity(G)
    else:
        raise ValueError, 'Cannot compute degree assortativity: method not available'

def drake_hougardy_slow(G):
#uses an implementation close to the pseudo-code in the paper
    assert not G.is_directed()
    H = nx.Graph()
    for u,v,d in G.edges(data=True):
        H.add_edge(u,v,d)
    #H = G.copy() #deep
    H_adj = H.adj
    H_degree    = lambda u: H_adj[u].__len__()
    H_outedges  = lambda u: H.edge[u]


    Matchings = ([],[])
    Weights   = [0.,0.]
    ind = 0
    #edges           = set(G.edges())
    #inspected_nodes = dict.from_keys(G.nodes(), False)
    ni             = G.nodes_iter()  #use G, not H
    #qx = ni.next()

    while H.number_of_edges() > 0:
        try:
            x = ni.next()
        except StopIteration:
            break
        while x in H and H_degree(x) > 0:
            nbs = H_outedges(x)
            nb_weights = [(nb,nbs[nb].get('weight', 1.0)) for nb in nbs]
            y,wt_y = max(nb_weights, key=lambda x:x[1])
            Matchings[ind].append((x,y))
            Weights[ind] += wt_y
            H.remove_node(x)
            x = y
            ind = (ind + 1)%2
    if Weights[0] > Weights[1]:
        return dict(Matchings[0] + [(y,x) for (x,y) in Matchings[0]])
    else:
        return dict(Matchings[1] + [(y,x) for (x,y) in Matchings[1]])

def drake_hougardy(G, maximize=True):
    '''Compute a weighted matching of G using the Drake-Hougardy path growing algorithm.[1]
    The matching is guaranteed to have weight >= 0.5 of the maximumal weight matching

    Parameters
    ----------
    G: NetworkX undirected graph
    maximize: add an additional step to find edges missed by the matching, to return a maximal matching [2]

    Returns
    -------
    mate : dictionary
       The matching is returned as a dictionary, mate, such that
       mate[v] == w if node v is matched to node w.  Unmatched nodes do not occur as a key in mate.
       for convenience, iff mate[v] == w then mate[w] == v

    References
    ----------
    .. [1] "A simple approximation algorithm for the weighted matching problem"
        Doratha E. Drake, Stefan Hougardy. Information Processing Letters, 2002.
    .. [2] "Linear time local improvements for weighted matchings in graphs"
        Doratha E. Drake, Stefan Hougardy. Report.
    '''
    assert not G.is_directed()
    G_adj = G.adj
    G_outedges  = lambda u: G.edge[u]
    mx = max

    Matchings = ([],[])
    Weights   = [0.,0.]
    ind       = 0
    ni        = G.nodes_iter()  #use G, not G
    inspected = set()  #iff u in inspected, it has been already included in the matching.
    num_inspected_halfedges = 0
    num_edges = G.number_of_edges()

    nodes = G.nodes()
    try:
        npr.shuffle(nodes)    #wishlist: why does it fail on npr.shuffle([u'903', 1]) ??
    except:
        random.shuffle(nodes)  #wishlist: might be too slow...
    for x in nodes:
        if num_inspected_halfedges >= 2*num_edges:
            break
        while x not in inspected:
            inspected.add(x)
            nbs = G_outedges(x)
            num_inspected_halfedges += nbs.__len__()
            nb_weights = [(nb,nbs[nb].get('weight', 1.0)) for nb in nbs if nb not in inspected]
            if nb_weights.__len__() == 0:
                continue
            y,wt_y = mx(nb_weights, key=lambda pair:pair[1])
            Matchings[ind].append((x,y))
            Weights[ind] += wt_y
            x = y
            ind = (ind + 1)%2


    if Weights[0] > Weights[1]:
        best_matching = dict(Matchings[0] + [(y,x) for (x,y) in Matchings[0]])
    else:
        best_matching = dict(Matchings[1] + [(y,x) for (x,y) in Matchings[1]])

    if not maximize:
        return best_matching

    for x in G:
        if x in best_matching:
            continue
        nbs = G_outedges(x)
        nb_weights = [(nb,nbs[nb].get('weight', 1.0)) for nb in nbs if nb not in best_matching]
        if nb_weights.__len__() == 0:
            continue
        y,wt_y = mx(nb_weights, key=lambda pair:pair[1])
        best_matching[x] = y
        best_matching[y] = x
    return best_matching

def graph_graph_delta(G, new_G, **kwargs):
#lists the changes in the two graphs, and reports the Jaccard similarity coefficient for nodes and for edges
    new_nodes = []
    del_nodes = []
    new_edges = []
    del_edges = []

    for node in G:
        if node not in new_G:
            del_nodes.append(node)
    for edge in G.edges():
        if not new_G.has_edge(*edge):
            del_edges.append(edge)

    for node in new_G:
        if node not in G:
            new_nodes.append(node)
    for edge in new_G.edges():
        if not G.has_edge(*edge):
            new_edges.append(edge)

    ret = {'new_nodes':new_nodes, 'del_nodes':del_nodes, 'new_edges':new_edges, 'del_edges':del_edges}

    num_nodes_original = G.number_of_nodes()
    num_edges_original = G.number_of_edges()
    if num_nodes_original + len(new_nodes) > 0:
        jaccard_nodes = float(num_nodes_original-len(del_nodes))/(num_nodes_original + len(new_nodes))
    else:
        jaccard_nodes = 0.
    if num_edges_original + len(new_edges) > 0:
        jaccard_edges = float(num_edges_original-len(del_edges))/(num_edges_original + len(new_edges))
    else:
        jaccard_edges = 0.
    ret['jaccard_nodes'] = jaccard_nodes
    ret['jaccard_edges'] = jaccard_edges

    return ret

def graph_sanity_test(G, params=None):
    ok = True
    if G.number_of_nodes() == 0:
        print 'Warning: no nodes'
        ok = False
    elif G.has_node(None):
        print 'Node with label "None" is in the graph.'
        ok = False
    elif G.number_of_edges() == 0:
        print 'Warning: no edges'
        ok = False
    elif G.is_directed():
        print 'Warning: the algorithm DOES NOT support directed graphs for now'
        ok = False

    if ok:
        selfloops = G.selfloop_edges()
        if selfloops != []:
            print 'Warning: self-loops detected (%d)'%len(selfloops)
            print 'Deleting!'
            G.remove_edges_from(selfloops)
            ok = False

    return ok


def load_graph(path, params={}, list_types_and_exit=False):
    '''reads graph from path, using automatic detection of graph type
       to attempt AUTODETECTION use params['graph_type'] = AUTODETECT
    '''

    loaders = {
            'adjlist':nx.read_adjlist,
            'adjlist_implicit':read_adjlist_implicit,
            'adjlist_implicit_prefix':read_adjlist_implicit_prefix,
            'graph6':nx.read_graph6,
            'shp':nx.read_shp,
            'dot':nx.drawing.nx_agraph.read_dot,
            'xdot':nx.drawing.nx_agraph.read_dot,
            'sparse6':nx.read_sparse6,
            'edges':nx.read_edgelist,
            'elist':nx.read_edgelist,
            'edgelist':nx.read_edgelist,
            'graphml':nx.read_graphml,
            'gexf':nx.read_gexf,
            'leda':nx.read_leda,
            'weighted_edgelist':nx.read_weighted_edgelist,
            'gml':nx.read_gml,
            'multiline_adjlist':nx.read_multiline_adjlist,
            'yaml':nx.read_yaml,
            'gpickle':nx.read_gpickle,
            'pajek':nx.read_pajek,}

    raw_loaders = {
            'adjlist':nx.parse_adjlist,
            'elist':nx.parse_edgelist,
            'edgelist':nx.parse_edgelist,
            'gml':nx.parse_gml,
            'leda':nx.parse_leda,
            'multiline_adjlist':nx.parse_multiline_adjlist,
            'pajek':nx.parse_pajek,
            }

    known_extensions = {
            'gml':'gml',
            'dot':'dot',
            'xdot':'dot',
            'edges':'edgelist',
            'elist':'edgelist',
            'edgelist':'edgelist',
            'welist':'weighted_edgelist',
            'wdgelist':'weighted_edgelist',
            'weighted_edgelist':'weighted_edgelist',
            'adj':'adjlist',
            'alist':'adjlist',
            'adjlist':'adjlist',
            'adjlistImp':'adjlist_implicit',
            'adjlistImpPre':'adjlist_implicit_prefix',
            'pajek':'pajek',
            'net':'pajek',
            }

    if list_types_and_exit:
        return loaders.keys()

    def sane_graph(G, params={}):
        if G.number_of_nodes() > 0:
            return True
        else:
            return False

    G = None
    graph_type = params.get('graph_type', 'AUTODETECT')
    read_params= params.get('read_params', {})
    skip_sanity= params.get('skip_sanity', False)

    if not os.path.exists(path):
        raise ValueError, 'Path does not exist: %s'%path

    if graph_type in loaders:
        if graph_type in ['edges', 'elist', 'edgelist']:
            print "Default weight is 1. To indicate weight, each line should use the format: node1 node2 {'weight':positive_wt}"
        if graph_type in ['adjlist']:
            print 'Adjlist format: WARNING! assuming that the lines are: "u neighbor1 neighbor2 etc".  Implicit "u" is not allowed'
        try:
            G = loaders[graph_type](path=path, **read_params)
            if not sane_graph(G) and not skip_sanity:
                print 'Warning: Sanity test failed!'
                print
                graph_type = None
        except:
            print 'Graph read error.'
            raise

    if G == None and graph_type != 'AUTODETECT':
        raise Exception,'Unable to load graphs of type '+str(graph_type)

    extension_guess = os.path.splitext(path)[1][1:]
    if G == None and extension_guess in known_extensions:
        print 'Attempting auto-detection of graph type.'

        if params.get('verbose', True):
            print 'Warning: Trying to auto-detect graph type by extension'
        graph_type = known_extensions[extension_guess]
        if params.get('verbose', True):
            print 'Guessing type: '+str(graph_type)
        try:
            G = loaders[graph_type](path=path)
            assert sane_graph(G) or skip_sanity
        except Exception, inst:
            print 'Graph read error.  This might be caused by malformed edge data or unicode errors.'
            print inst

    if G == None and graph_type in raw_loaders:
        if params.get('verbose', True):
            print 'Trying raw read...'
        try:
            f = open(path, 'rb')
            lines = f.readlines()
            G = raw_loaders[graph_type](lines=lines)
            del lines
            assert sane_graph(G) or skip_sanity
        except Exception, inst:
            print 'Graph read error:'
            print inst
        finally:
            try:
                f.close()
            except:
                pass

    if G == None:
        if params.get('verbose', True):
            print 'Warning: Trying to guess graph type iteratively: this often FAILS'
        for graph_type in loaders:
            try:
                if params.get('verbose', True):
                    sys.stdout.write(graph_type + '? ')
                G = loaders[graph_type](path=path)
                if sane_graph(G) or skip_sanity:
                    if params.get('verbose', True):
                        print(' Yes!')
                        print 'Successfully detected type: '+str(graph_type)
                    break
                else:
                    if params.get('verbose', True):
                        print(' No.')
                    G = None
                #wishlist: attempt edgelist before adjlist
            except:
                if params.get('verbose', True):
                    print(' No.')

    if G == None:
        raise Exception, 'Could not load graph.  None of the available loaders succeeded.'

    #postprocessing
    if graph_type == 'dot':
        G.name = os.path.split(path)[1]  #otherwise the output is terrible
    try:
        if not hasattr(G, 'name') or G.name == '':
            G.name = os.path.split(path)[1]
    except:
        pass

    return G

def make_gt_graph(nxG):
    import graph_tool, graph_tool.topology
    node_to_num  = dict(zip(nxG.nodes(), range(nxG.number_of_nodes())))
    gtG          = graph_tool.Graph(directed=nxG.is_directed())
    edge_weights = gtG.new_edge_property("double")
    gtG.add_vertex(nxG.number_of_nodes())
    for u,v,data in nxG.edges(data=True):
        e = gtG.add_edge(node_to_num[u], node_to_num[v])
        edge_weights[e] = data.get('weight', 1.0)
    return {'G':gtG, 'edge_weights':edge_weights, 'node_to_num':node_to_num}

def powerlaw_mle(G, xmin=6.):
    #estimate the power law exponent based on Clauset et al., http://arxiv.org/abs/0706.1062,
    #for simplicity, we avoid the MLE calculation of Eq. (3.5) and instead use the approximation of Eq. (3.7)
    #the power law is only applied for nodes of degree > xmin, so it's not suitable for others
    degseq = G.degree().values()

    #print np.array(degseq).transpose()

    if xmin < 6:
        print 'Warning: the estimator uses an approximation which is not suitable for xmin < 6'

    degseqLn = [np.log(xi/(xmin-0.5)) for xi in degseq if xi >= xmin]
    degseqLn.sort() #to reduce underflow.

    #print degseqLn
    return 1. + len(degseqLn) / sum(degseqLn)



def read_adjlist_implicit(path, comments = '#', delimiter = None,
                  create_using = None, nodetype = int):
    """Parse lines of a graph adjacency list representation.

    Parameters
    ----------
    lines : list or iterator of strings
        Input data in adjlist format
        The line number is IMPLICIT, starting with line 1

    create_using: NetworkX graph container
       Use given NetworkX graph for holding nodes or edges.

    nodetype : Python type, optional
       Convert nodes to this type.

    comments : string, optional
       Marker for comment lines

    delimiter : string, optional
       Separator for node labels.  The default is whitespace.

    create_using: NetworkX graph container
       Use given NetworkX graph for holding nodes or edges.


    Returns
    -------
    G: NetworkX graph
        The graph corresponding to the lines in adjacency list format.

    Examples
    --------
    >>> lines = ['2 5',
    ...          '3 4',
    ...          '5',
    ...          '',
    ...          '']
    >>> G = nx.parse_adjlist(lines, nodetype = int)
    >>> G.nodes()
    [(1, 2), (1, 5), (2, 3), (2, 4), (3, 5)]

    See Also
    --------
    read_adjlist

    """
    if create_using is None:
        G=nx.Graph()
    else:
        try:
            G=create_using
            G.clear()
        except:
            raise TypeError("Input graph is not a NetworkX graph type")

    with open(path, 'r') as file:
        lines = file.readlines()

    linenum = 1
    for line in lines:
        p=line.find(comments)
        if p>=0:
            line = line[:p]
        if not len(line):
            continue
        vlist=line.strip().split(delimiter)
        #u=vlist.pop(0)
        # convert types
        u = linenum
        if nodetype is not None:
            try:
                u=nodetype(linenum)
            except:
                raise TypeError("Failed to convert node (%s) to type %s"\
                                %(u,nodetype))
        G.add_node(u)
        if nodetype is not None:
            try:
                vlist=map(nodetype,vlist)
            except:
                raise TypeError("Failed to convert nodes (%s) to type %s"\
                                    %(','.join(vlist),nodetype))
        G.add_edges_from([(u, v) for v in vlist])
        linenum += 1
    return G


def read_adjlist_implicit_prefix(path, comments = '#', create_using=None):
    '''
    reads network files formatted as:

    "
    15606 45878
    2 3 6 7
    1 4 6 9
    "
    and so on.
    first line: num_nodes num_edges
    second lines (and the rest of the lines in the file):
    [implicit node = line number - 1] neighbor1 neighbor2 ...
    empty lines are degree=0 nodes
    '''

    if create_using == None:
        G = nx.Graph()
    else:
        G = create_using()

    try:
        with open(path, 'r') as file_handle:
            header_data = file_handle.next().split(' ')
            node_num = 1
            for line in file_handle:
                p=line.find(comments)
                if p>=0:
                    line = line[:p]
                if not len(line):
                    continue
                line = line.strip()
                if line == '':
                    G.add_node(node_num)
                else:
                    G.add_edges_from([(node_num,int(v)) for v in line.split(' ')])
                node_num += 1
    except Exception,inst:
        if 'node_num' not in locals():
            raise
        raise IOError, 'Parse error on line %d'%(node_num+1)

    expected_num_nodes = int(header_data[0])
    expected_num_edges = int(header_data[1])

    if G.number_of_nodes() != expected_num_nodes or G.number_of_edges() != expected_num_edges:
        raise IOError, 'Failed integrity check to input. Expected nn=%d,ne=%d; Read nn=%d,ne=%d'%(expected_num_nodes,expected_num_edges,G.number_of_nodes(),G.number_of_edges())

    return G

def safe_pickle(path, data, params=None):
    with open(path, 'wb') as f:
        cPickle.dump(data, f)
        if type(params) != type({}) or params.get('verbose', True):
            print 'pickled to: '+str(path)


def test_algebraic_distance():
    #TODO: need new tests: edges are always=1 for normal distance
    #given two start vectors, a mesh should unfold
    print 'Testing Algebraic distance'
    #test1: nodes nearby on the path graph should land nearby
    print 'test path ...'
    G1 = nx.path_graph(10)
    distance1 = algebraic_distance(G1, params={'all_pairs_distance':False})  #usual regime
    true_distance1 = []
    alg_distance1 = []
    for node1 in G1:
        for node2 in G1.neighbors(node1):
            if node1 > node2:
                continue
            true_distance1.append(abs(node1-node2))
            alg_distance1.append(distance1[node1][node2])

    assert distance1[0][1] == distance1[1][0]
    val1 = np.corrcoef(true_distance1, alg_distance1)[0,1]
    print 'correlation: %.2f'%val1
    assert val1 > 0.8
    print 'passed.'

    print 'test grid'
    G2=nx.grid_graph(dim=[10,10])
    distance2 = algebraic_distance(G2, params={'all_pairs_distance':True})
    true_distance2 = []
    alg_distance2 = []
    for node1 in G2:
        for node2 in G2:
            if sum(node1) > sum(node2):
                continue
            true_distance2.append(abs(node1[0]-node2[0]) + abs(node1[1]-node2[1]))
            alg_distance2.append(distance2[node1][node2])

    val2 = np.corrcoef(true_distance2, alg_distance2)[0,1]
    print 'correlation: %.2f'%val2
    assert val2 > 0.5

    val1 = np.corrcoef(true_distance1, alg_distance1)[0,1]

    distance2sp = algebraic_distance_sparse(G2, params={'all_pairs_distance':True})
    err2 = 0
    for node1 in G2:
        for node2 in G2:
            if sum(node1) > sum(node2):
                continue
            err2 += abs(distance2sp[node1][node2] - distance2[node1][node2])
    err2 = err2/G.number_of_edges()
    print '  mean gap for pair: %.f'%(err2)
    assert err2 < 0.1

    print 'passed.'

def test_average_path_length():
    print 'Testing avg path length estimator'
    G = nx.barabasi_albert_graph(300, 5)
    #G = nx.cycle_graph(300)

    estimated_avg = average_all_pairs_shortest_path_estimate(G, max_num_sources=200)

    true_lengths = nx.all_pairs_shortest_path_length(G)
    true_avg = np.average([np.average(true_lengths[node].values()) for node in G])

    print 'Estimate: %f'%estimated_avg
    print 'True:     %f'%true_avg

    assert abs(estimated_avg-true_avg)/true_avg < 0.03
    print 'PASSED'

def test_bfs():
    print 'Testing BFS'
    G = nx.path_graph(5)
    distances_path0 = bfs_distance_with_horizon(G, source=0, horizon=2)
    assert distances_path0[0] == 0
    assert distances_path0[1] == 1
    assert distances_path0[2] == 2
    assert 3 not in distances_path0
    assert 4 not in distances_path0
    distances_path1 = bfs_distance_with_horizon(G, source=1, horizon=2)
    assert distances_path1[0] == 1
    assert distances_path1[1] == 0
    assert distances_path1[2] == 1
    assert distances_path1[3] == 2
    assert 4 not in distances_path1

    ER100 = nx.erdos_renyi_graph(100, 0.02)
    true_d       = nx.all_pairs_shortest_path_length(ER100)
    cc1 = nx.connected_components(ER100)[0]
    for node1 in cc1:
        horizon_d_node1    = bfs_distance_with_horizon(ER100, source=node1, horizon=4)
        horizon_dinf_node1 = bfs_distance_with_horizon(ER100, source=node1, horizon=1000)
        for node2 in cc1:
            if node2 in horizon_d_node1:
                assert true_d[node1][node2] == horizon_d_node1[node2]
            assert true_d[node1][node2] == horizon_dinf_node1[node2]
    print 'PASSED'

    s='''
    import networkx as nx
    import graphutils
    #G = nx.grid_2d_graph(10, 10)
    G = nx.erdos_renyi_graph(200, 0.2)
    #graphutils.bfs_distance_with_horizon(G, source=(1,5), horizon=10)
    graphutils.bfs_distance_with_horizon(G, source=15, horizon=10)
    '''
    import timeit
    t=timeit.Timer(stmt=s)
    num_trials = 100
    print '%f usec/pass'%(t.timeit(number=num_trials)/num_trials)

def test_graphtool_distance():
    G = nx.connected_watts_strogatz_graph(n=2000, k=4, p=0.02, tries=100, seed=None)
    print 'NX: %.3f'%average_all_pairs_shortest_path(G)
    print 'GT: %.3f'%average_all_pairs_shortest_path_GT(G)
    print 'NX sampling: %.3f'%average_all_pairs_shortest_path_estimate(G)
    print 'GT sampling: %.3f'%average_all_pairs_shortest_path_estimate_GT(G)

def test_inverse_mean_path_length():
    print 'Testing BFS'
    G = nx.erdos_renyi_graph(100, 0.02)
    eff_est = average_all_pairs_inverse_shortest_path_estimate(G, max_num_sources=100)
    print 'Estimate: '+str(eff_est)
    eff_tru = average_all_pairs_inverse_shortest_path_estimate(G, max_num_sources=G.number_of_nodes())
    print 'True:     '+str(eff_tru)
    assert abs(eff_est-eff_tru)/eff_tru < 0.05
    print 'PASSED'

def test_powerlaw_mle():
    print 'Testing Power law MLE estimator'
    G = nx.barabasi_albert_graph(100, 5)
    print 'nn: %d, alpha: %f'%(G.number_of_nodes(),powerlaw_mle(G))
    G = nx.barabasi_albert_graph(1000, 5)
    print 'nn: %d, alpha: %f'%(G.number_of_nodes(),powerlaw_mle(G))
    G = nx.barabasi_albert_graph(10000, 5)
    print 'nn: %d, alpha: %f'%(G.number_of_nodes(),powerlaw_mle(G))
    G = nx.barabasi_albert_graph(100000, 5)
    print 'nn: %d, alpha: %f'%(G.number_of_nodes(),powerlaw_mle(G))
    print 'Expected: 2.9 (or thereabout)'


def write_dot_helper(G, path, encoding='utf-8'):
    #a simplified implementation of dot writer
    #needed in the Windows platform where pygraphviz is not available
    #loses label information
    with open(path, mode='wb') as f:
        header = 'strict graph ' + getattr(G, 'name', 'replica') + ' {\n'.encode(encoding)
        f.write(header)
        for line in nx.generate_edgelist(G, ' -- ', False):
            line =' %s;\n'%line
            f.write(line.encode(encoding))
        f.write('}\n'.encode(encoding))

def write_graph(G, path, params={}, list_types_and_exit=False):
    '''reads graph from path, using automatic detection of graph type
    '''

    writers = {
            'adjlist':nx.write_adjlist,
            'dot':nx.drawing.nx_agraph.write_dot,
            'xdot':nx.drawing.nx_agraph.write_dot,
            'edges':nx.write_edgelist,
            'elist':nx.write_edgelist,
            'edgelist':nx.write_edgelist,
            'weighted_edgelist':nx.write_weighted_edgelist,
            'graphml':nx.write_graphml,
            'gml':nx.write_gml,
            'gpickle':nx.write_gpickle,
            'pajek':nx.write_pajek,
            'yaml':nx.write_yaml}
    if os.name == 'nt':
        writers['dot'] = write_dot_helper
        writers['xdot'] = write_dot_helper

    if list_types_and_exit:
        return writers.keys()

    write_params = params.get('write_params', {})
    skip_sanity  = params.get('skip_sanity', False)

    graph_type = os.path.splitext(path)[1][1:]

    if graph_type in writers:
        try:
            writers[graph_type](G=G, path=path, **write_params)
        except Exception, inst:
            print 'Graph write error:'
            print inst

            print 'Attempting to write to DOT format'
            nx.drawing.nx_agraph.write_dot(G, path)
            print 'Done.'
    else:
        raise Exception,'Unable to write graphs of type: '+str(graph_type)


#runningtime based on the power on the V or E.  e.g. 1 linear, 2 quadratic etc.
default_metrics = []
default_metrics += [{'name':'num nodes',          'weight':1, 'optional':0, 'runningtime': 1, 'function':nx.number_of_nodes}]
default_metrics += [{'name':'density',            'weight':1, 'optional':2, 'runningtime': 1, 'function':nx.density}]
default_metrics += [{'name':'num edges',          'weight':1, 'optional':0, 'runningtime': 1, 'function':nx.number_of_edges}]
default_metrics += [{'name':'num comps',          'weight':1, 'optional':0, 'runningtime': 1, 'function':nx.number_connected_components}]
default_metrics += [{'name':'clustering',         'weight':1, 'optional':0, 'runningtime': 1, 'function':nx.average_clustering}]
default_metrics += [{'name':'average degree',     'weight':1, 'optional':0, 'runningtime': 1, 'function':a_avg_degree}]
default_metrics += [{'name':'degree assortativity', 'weight':1, 'optional':2, 'runningtime': 1, 'function':degree_assortativity}]
default_metrics += [{'name':'degree connectivity', 'weight':1, 'optional':2, 'runningtime': 1, 'function':a_degree_connectivity}]

default_metrics += [{'name':'total deg*deg',       'weight':1, 'optional':0, 'runningtime': 1, 'function':a_s_metric}]
#wishlist: make default
default_metrics += [{'name':'mean ecc',           'weight':1, 'optional':1, 'runningtime': 3.5, 'function':a_eccentricity}]
#default_metrics += [{'name':'L eigenvalue sum',   'weight':1, 'optional':0, 'runningtime': 3, 'function':lambda G: sum(nx.spectrum.laplacian_spectrum(G)).real}]
default_metrics += [{'name':'average shortest path',   'weight':1, 'optional':0, 'runningtime': 3, 'function':a_avg_shortest}]
default_metrics += [{'name':'harmonic mean path',   'weight':1, 'optional':0, 'runningtime': 3, 'function':a_avg_harmonic}]
#flow_closeness appears to be broken in NX 1.6
default_metrics += [{'name':'avg flow closeness',   'weight':1, 'optional':1, 'runningtime': 3, 'function':average_flow_closeness}]
#wishlist: make optional for speed
default_metrics += [{'name':'avg eigvec centrality',   'weight':1, 'optional':0, 'runningtime': 3, 'function':average_eigenvector_centrality}]
#wishlist: make default
default_metrics += [{'name':'avg between. central.',   'weight':1, 'optional':1, 'runningtime': 4, 'function':a_avg_between}]
default_metrics += [{'name':'modularity',          'weight':1, 'optional':0, 'runningtime': 2, 'function':community.louvain_modularity}]
default_metrics += [{'name':'powerlaw exp',          'weight':1, 'optional':0, 'runningtime': 3, 'function':powerlaw_mle}]
#'optional' runs from 0 (always used) to 5 (never)


if __name__ == '__main__':
    pass
    #test_graphtool_distance()
    #test_algebraic_distance()
    #test_bfs()
    #test_average_path_length()
    #test_inverse_mean_path_length()
    #test_powerlaw_mle()
