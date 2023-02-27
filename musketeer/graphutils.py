'''
Multiscale Entropic Network Generator 2 (MUSKETEER2)

Copyright (c) 2011-2023 by Alexander Gutfraind and Ilya Safro.
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
import numpy.random as npr
import random, sys
import networkx as nx
import pdb
import _pickle as cPickle

#wishlist: make this load lazily to speedup loading
import yaml  

np.seterr(all='raise')

timeNow = lambda : time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())

None_node = None  #key to indicate a not real node.

METRIC_ERROR = np.NaN  #for consist reporting of error in graph metrics

def a_avg_degree(G):
    if nx.number_of_nodes(G) > 0:
        return 2.0*nx.number_of_edges(G)/nx.number_of_nodes(G)
    else:
        return 0.0
def a_degree_connectivity(G):
    return np.fromiter(nx.average_degree_connectivity(G).values(), float).mean()
def a_s_metric(G):
    return nx.s_metric(G, normalized=False)
def a_eccentricity(G):
    return np.fromiter(nx.eccentricity(G).values(), float).mean()
def a_avg_shortest(G):
    return average_all_pairs_shortest_path_estimate(G, max_num_sources=100)
def a_avg_harmonic(G):
    return average_all_pairs_inverse_shortest_path_estimate(G, max_num_sources=100)
def a_avg_between(G):
    return np.fromiter(nx.betweenness_centrality(G, normalized=True).values(), float).mean()


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
        total_distance += np.fromiter(lengths.values(), float).mean()
    total_distance /= num_sources
    return total_distance



def average_all_pairs_inverse_shortest_path_estimate(G, max_num_sources=100):
    """
    estimates the ''efficiency'' of the graph: the harmonic mean of the distances
    this is well-defined even in disconnected graphs
    """
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

def bfs_distance_with_horizon(G, source, horizon=4, blocked_node=None):
    """
    computes distance from every node to every neighbor at distance at most <horizon> hops
    nodes further away are considered infinitely away
    no path is allowed through blocked_node
    """
    G_adj = G.adj
    G_neighbors = lambda u: G_adj[u].keys()

    fringe = set(G_neighbors(source))
    distance_source  = {source:0}
    for d in range(1, horizon+1):
        new_fringe = []
        for v in fringe:
            if v not in distance_source and v!=blocked_node:
                distance_source[v] = d
                new_fringe += G_neighbors(v)
        fringe = set(new_fringe)

    return distance_source

def color_by_3d_distances(G):
    """
    adds edge color attribute based on distance
    G.nodes should have comma-sep pos field
    """
    import matplotlib.pylab as pylab
    #cm=pylab.get_cmap('Paired')
    #cm=pylab.get_cmap('gist_rainbow')
    cm=pylab.get_cmap('RdBu')  #UFL

    if verbose:
        print('Computing edge colors ...')
    max_dis = 0
    positions = {} #cache
    for u,v,data in G.edges(data=True):
        try:
            u_pos = positions[u]
        except:
            u_pos = np.array([float(p) for p in G.nodes[u]['pos'].split(',')])
            positions[u] = u_pos
        try:
            v_pos = positions[v]
        except:
            v_pos = np.array([float(p) for p in G.nodes[v]['pos'].split(',')])
            positions[v] = v_pos

        dis = np.sqrt(np.sum(np.power(u_pos-v_pos,2)))
        max_dis = max(max_dis, dis)

        data['dis'] = dis

    for u,v,data in G.edges(data=True):
        dis = data.pop('dis')
        data['color'] = '%.3f %.3f %.3f'%tuple(cm(dis/max_dis)[:3])

    return G

#color_by_3d_distances(G, False)

def color_new_nodes_and_edges(G, original, params=None):
    """
    add red color to new nodes and edges (for visualization)
    use the option 'post_processor':graphutils.color_new_nodes_and_edges
    """
    for node in G:
        G.nodes[node]['label'] = ''
        #d['style'] = 'filled'
        if node in original:
            G.nodes[node]['color']='black'
        else:
            G.nodes[node]['color']='blue'
    for u,v,d in G.edges(data=True):
        if original.has_edge(u,v):
            d['color']='black'
        else:
            d['color']='blue'

    return G


def compare_nets(old_G, new_G, metrics=None, params={}):
    """
    Report on the differences between two networks
    :param metrics: are functions, alternatively uses the defaultset
    """
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
            print('New or deleted Nodes: %d (%.1f%%)'%(num_changed_nodes, 100*float(num_changed_nodes)/old_G.number_of_nodes()))
            print('New or deleted Edges: %d (%.1f%%)'%(num_changed_edges, 100*float(num_changed_edges)/old_G.number_of_edges()))
            print()
        print('Name\t\t\tOld G\t\tNew G\t\tRelative Error')
        print('statistics start ------------------------------------------------------------')
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
            if (old_value != 0.) and (not np.isnan(old_value)) and (not np.isnan(new_value)):
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
                print(outstr)
                #print formatstring%(old_value,new_value,100*error)
            errors[met_name] = (old_value, new_value, error)
        except Exception as inst:
            print()
            print('Warning: could not compute '+met_name + ': '+str(inst))
    mean_error = np.average([np.abs(v[2]) for v in errors.values() if (v[2]!=np.NaN) and (not np.isnan(v[2]))])
    if verbose:
        print('statistics end ------------------------------------------------------------')
        print('Mean absolute difference: %.2f%%'%(100*mean_error))

    return mean_error, errors

def degree_assortativity(G):
#this wrapper helps avoid error due to change in interface name
    if hasattr(nx, 'degree_assortativity_coefficient'):
        return nx.degree_assortativity_coefficient(G)
    elif hasattr(nx, 'degree_assortativity'):
        return nx.degree_assortativity(G)
    else:
        raise ValueError ('Cannot compute degree assortativity: method not available')


def drake_hougardy(G, maximize=True):
    """
    Compute a weighted matching of G using the Drake-Hougardy path growing algorithm.[1]
    The matching is guaranteed to have weight >= 0.5 of the maximumal weight matching

    :param G: NetworkX undirected graph
    maximize: add an additional step to find edges missed by the matching, to return a maximal matching [2]

    :returns: mate Dictionary 
       The matching is returned as a dictionary, mate, such that
       mate[v] == w if node v is matched to node w.  Unmatched nodes do not occur as a key in mate.
       for convenience, iff mate[v] == w then mate[w] == v

    References
    ----------
    .. [1] "A simple approximation algorithm for the weighted matching problem"
        Doratha E. Drake, Stefan Hougardy. Information Processing Letters, 2002.
    .. [2] "Linear time local improvements for weighted matchings in graphs"
        Doratha E. Drake, Stefan Hougardy. Report.
    """
    assert not G.is_directed()
    G_adj = G.adj
    G_outedges  = lambda u: G[u]
    mx = max

    Matchings = ([],[])
    Weights   = [0.,0.]
    ind       = 0
    inspected = set()  #iff u in inspected, it has been already included in the matching.
    num_inspected_halfedges = 0
    num_edges = G.number_of_edges()

    nodes = G.nodes()
    try:
        #npr.shuffle(nodes)    #wishlist: why does it fail on npr.shuffle([u'903', 1]) ??
        node_mapping = dict(zip(nodes, sorted(nodes, key=lambda k: npr.random())))
        nodes = nx.relabel_nodes(G, node_mapping)
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

def drake_hougardy_flattened(G, maximize=True):
    assert G.is_directed()
    H = nx.reverse(G)
    GcomposeH = nx.compose(G, H).to_undirected()

    edge_data = {e: G.edges[e].get('weight', 1.0) + \
                    H.edges[e].get('weight', 1.0) for e in G.edges & H.edges}

    nx.set_edge_attributes(GcomposeH, edge_data, 'weight')

    return drake_hougardy(GcomposeH, maximize=maximize)

def graph_graph_delta(G, new_G, **kwargs):
    """
    lists the changes in the two graphs, and reports the Jaccard similarity coefficient for nodes and for edges
    """
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
        print('Warning: no nodes')
        ok = False
    elif G.has_node(None):
        print('Node with label "None" is in the graph.')
        ok = False
    elif G.number_of_edges() == 0:
        print('Warning: no edges')
        ok = False
    elif G.is_directed():
        print('Warning: the algorithm DOES NOT support directed graphs for now')
        ok = False

    if ok:
        selfloops = nx.selfloop_edges(G)
        if selfloops != []:
            print('Notice: self-loops detected - deleting')
            G.remove_edges_from(selfloops)
            ok = False

    return ok


def load_graph(path, params={}, list_types_and_exit=False):
    """
    reads graph from path, using automatic detection of graph type
       to attempt AUTODETECTION use params['graph_type'] = AUTODETECT
    """

    loaders = {
            'adjlist':nx.read_adjlist,
            'adjlist_implicit':read_adjlist_implicit,
            'adjlist_implicit_prefix':read_adjlist_implicit_prefix,
            'graph6':nx.read_graph6,
            #'shp':nx.read_shp,
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
            'yaml':yaml.load,
            'gpickle':cPickle.load,
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
        raise ValueError ('Path does not exist: %s'%path)

    if graph_type in loaders:
        if graph_type in ['edges', 'elist', 'edgelist']:
            print("Default weight is 1. To indicate weight, each line should use the format: node1 node2 {'weight':positive_wt}")
        if graph_type in ['adjlist']:
            print('Adjlist format: WARNING! assuming that the lines are: "u neighbor1 neighbor2 etc".  Implicit "u" is not allowed')
        try:
            G = loaders[graph_type](path=path, **read_params)
            if not sane_graph(G) and not skip_sanity:
                print('Warning: Sanity test failed!')
                print()
                graph_type = None
        except:
            print('Graph read error.')
            raise

    if G == None and graph_type != 'AUTODETECT':
        raise Exception('Unable to load graphs of type '+str(graph_type))

    extension_guess = os.path.splitext(path)[1][1:]
    if G == None and extension_guess in known_extensions:
        print('Attempting auto-detection of graph type.')

        if params.get('verbose', True):
            print('Warning: Trying to auto-detect graph type by extension')
        graph_type = known_extensions[extension_guess]
        if params.get('verbose', True):
            print('Guessing type: '+str(graph_type))
        try:
            G = loaders[graph_type](path=path)
            assert sane_graph(G) or skip_sanity
        except Exception as inst:
            print('Graph read error.  This might be caused by malformed edge data or unicode errors.')
            print(inst)

    if G == None and graph_type in raw_loaders:
        if params.get('verbose', True):
            print('Trying raw read...')
        try:
            f = open(path, 'rb')
            lines = f.readlines()
            G = raw_loaders[graph_type](lines=lines)
            del lines
            assert sane_graph(G) or skip_sanity
        except Exception as inst:
            print('Graph read error:')
            print(inst)
        finally:
            try:
                f.close()
            except:
                pass

    if G == None:
        if params.get('verbose', True):
            print('Warning: Trying to guess graph type iteratively: this often FAILS')
        for graph_type in loaders:
            try:
                if params.get('verbose', True):
                    sys.stdout.write(graph_type + '? ')
                G = loaders[graph_type](path=path)
                if sane_graph(G) or skip_sanity:
                    if params.get('verbose', True):
                        print(' Yes!')
                        print('Successfully detected type: '+str(graph_type))
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
        raise Exception( 'Could not load graph.  None of the available loaders succeeded.')

    #postprocessing
    if graph_type == 'dot':
        G.name = os.path.split(path)[1]  #otherwise the output is terrible
    try:
        if not hasattr(G, 'name') or G.name == '':
            G.name = os.path.split(path)[1]
    except:
        pass

    return G


def louvain_modularity(G, **kwargs):
    import networkx.algorithms.community as nx_comm
    best_q = -np.inf
    
    for trial in range(kwargs.get('num_trials', 5)):
        parttn = nx_comm.louvain_communities(G)
        best_q = max(best_q, nx_comm.modularity(G, parttn))

    return best_q
    
def powerlaw_mle(G, xmin=6.):
    """
    estimate the power law exponent based on Clauset et al., http://arxiv.org/abs/0706.1062,
    for simplicity, we avoid the MLE calculation of Eq. (3.5) and instead use the approximation of Eq. (3.7)
    the power law is only applied for nodes of degree > xmin, so it's not suitable for others
    """
    degseq = np.fromiter(G.degree().values(), float)

    #print np.array(degseq).transpose()

    if xmin < 6:
        print('Warning: the estimator uses an approximation which is not suitable for xmin < 6')

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
    """
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
    """

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
    except Exception as inst:
        if 'node_num' not in locals():
            raise
        raise IOError ('Parse error on line %d'%(node_num+1))

    expected_num_nodes = int(header_data[0])
    expected_num_edges = int(header_data[1])

    if G.number_of_nodes() != expected_num_nodes or G.number_of_edges() != expected_num_edges:
        raise IOError ('Failed integrity check to input. Expected nn=%d,ne=%d; Read nn=%d,ne=%d'%(expected_num_nodes,expected_num_edges,G.number_of_nodes(),G.number_of_edges()))

    return G

def safe_pickle(path, data, params=None):
    with open(path, 'wb') as f:
        cPickle.dump(data, f)
        if type(params) != type({}) or params.get('verbose', True):
            print('pickled to: '+str(path))


def write_dot_helper(G, path, encoding='utf-8'):
    """
    a simplified implementation of dot writer
    needed in the Windows platform where pygraphviz is not available
    loses label information
    """
    with open(path, mode='wb') as f:
        header = 'strict graph ' + getattr(G, 'name', 'replica') + ' {\n'.encode(encoding)
        f.write(header)
        for line in nx.generate_edgelist(G, ' -- ', False):
            line =' %s;\n'%line
            f.write(line.encode(encoding))
        f.write('}\n'.encode(encoding))

def write_adjlist_implicit_prefix(G,path):
    graph = nx.convert_node_labels_to_integers(G)  
    myfile = open(path, 'w')
    out = str(nx.number_of_nodes(graph)) + ' ' + str(nx.number_of_edges(graph)) + '\n'
    for node in sorted(graph.nodes()):
        for v in graph.neighbors(node):
            if node != v:
                out += str(v + 1) + ' '  # metis indexing starts from 1
        out += '\n'
    myfile.write(out)
    myfile.close()


def write_graph(G, path, params={}, list_types_and_exit=False):
    """
    write graph to path. format is automatically detected from the extension
    """

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
            'gpickle':cPickle.dump,
            'pajek':nx.write_pajek,
            'adjlistImpPre':write_adjlist_implicit_prefix,
            'yaml':yaml.dump}
    if os.name == 'nt': #simplified writers
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
        except Exception as inst:
            print('Graph write error:')
            print(inst)

            print('Attempting to write to DOT format')
            nx.drawing.nx_agraph.write_dot(G, path)
            print('Done.')
    else:
        raise Exception ('Unable to write graphs of type: '+str(graph_type))


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
default_metrics += [{'name':'modularity',          'weight':1, 'optional':0, 'runningtime': 2, 'function':louvain_modularity}]
default_metrics += [{'name':'powerlaw exp',          'weight':1, 'optional':0, 'runningtime': 3, 'function':powerlaw_mle}]
#'optional' runs from 0 (always used) to 5 (never)

