'''
Multiscale Entropic Network Generator 2 (MUSKETEER2)

Copyright (c) 2011-2015 by Alexander Gutfraind and Ilya Safro. 
All rights reserved.

Use and redistribution of this file is governed by the license terms in
the LICENSE file found in the project's top-level directory.

Advanced Test Scripts
- helpful for calibration of MUSKETEER2, see particularly parallel generation and evaluation
statistical_tests() and replica_vs_original()

- WARNING: some of this file was not thoroughly tested, and may no longer work

'''

import os, subprocess
import time
import numpy as np
import numpy.random as npr
import random, sys
import networkx as nx
import matplotlib
matplotlib.use('PDF')
#import matplotlib.pylab as pylab
import pylab
import pdb, traceback
import pickle
import joblib
Parallel = joblib.Parallel
delayed  = joblib.delayed

import algorithms
import benchmarks
import graphutils
import simpletesters

np.seterr(all='raise')

timeNow = lambda : time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + '_%d'%npr.randint(1000)

try:
    subprocess.check_call('latex -version > /dev/null')
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.usetex']=True
    latex_available = True
except:
    matplotlib.rc('text', usetex=False)
    matplotlib.rcParams['text.usetex']=False
    latex_available = False

def clean_path(fpath):
#make the path suitable for TeX
    fpath = fpath.replace('.', 'd')
    fpath = fpath.replace(',', 'c')
    fpath = fpath.replace(' ', '_')
    fpath = fpath.replace('*', '_')
    fpath = fpath.replace('(', '_')
    fpath = fpath.replace(')', '_')

    return fpath

 
def coarsening_test():
#visualizes coarsening
    import matplotlib as mpl
    def build_block_G(pin=0.1, pout=0.01, block_size=32, num_blocks=4):
        G = nx.Graph()
        nn = block_size*num_blocks
        G.add_nodes_from([i for i in range(nn)])

        for nodeA in G:
            for nodeB in G:
                if nodeA / block_size == nodeB / block_size:
                    if random.random() < pin:
                        G.add_edge(nodeA, nodeB)
                else:
                    if random.random() < pout:
                        G.add_edge(nodeA, nodeB)
        G.remove_edges_from(G.selfloop_edges())
        return G

    G = build_block_G(pin=0.1, pout=0.01)
    def visualize_coarsening(G, G_coarse, c_data):
        npr.seed(10)
        random.seed(10)
        pos = nx.fruchterman_reingold_layout(G)

        seeds = list(c_data['aggregates'].keys())
        for seed in seeds:
            trapped_nodes = c_data['aggregates'][seed][:]
            trapped_nodes.remove(seed)
            #rnd_color     = random.choice(['r', 'b', 'g', 'c', 'm', 'y', 'w']) #[npr.rand(), npr.rand(), npr.rand(), npr.rand()]
            #rnd_color     = mpl.colors.rgb2hex((npr.rand(), npr.rand(), npr.rand()))
            #rnd_color     = random.random()
            #rnd_color      = random.choice(mpl.colors.cnames.keys())
            rnd_color     = (npr.rand(), npr.rand(), npr.rand(), 1.)
            color_seed = np.ones((1,4))
            color_rest = np.ones((len(trapped_nodes),4))
            for i,val in enumerate(rnd_color):
                color_seed[:,i] *= val 
                color_rest[:,i] *= val 
            nx.draw_networkx_nodes(G, pos=pos, nodelist=[seed],        node_color=color_seed, cmap=pylab.hot, node_size=500, with_labels=True, node_shape='s')
            nx.draw_networkx_nodes(G, pos=pos, nodelist=trapped_nodes, node_color=color_rest, cmap=pylab.hot, node_size=200, with_labels=True, node_shape='o')
        nx.draw_networkx_edges(G, pos=pos, alpha=1.0)
        nx.draw_networkx_labels(G, pos=pos)
        pylab.show()

    params = {}
    params['do_coarsen_tester'] = visualize_coarsening
    params['edge_edit_rate']    = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    params['node_edit_rate']    = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    params['node_growth_rate']    = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]

    algorithms.generate_graph(G, params=params)

def coarsening_test2(seed=None):
#visualizes coarsening: stores the coarsening of the nodes, and then labels the original nodes based on their aggregates in the final level
    import matplotlib as mpl
    if seed==None:
        seed = npr.randint(1E6)
    print('rnd seed: %d'%seed)
    npr.seed(seed)
    random.seed(seed)

    G = graphutils.load_graph('data/mesh33.gml')
    #G = graphutils.load_graph('data-engineering/watts_strogatz98_power.elist')
    c_tree = []
    def store_aggregation_chain(G, G_coarse, c_data):
        store_aggregation_chain.static_c_tree.append(c_data['home_nodes'].copy())
        #print c_data['home_nodes']
        #print store_aggregation_chain.static_c_tree

    store_aggregation_chain.static_c_tree = c_tree

    params = {}
    params['do_coarsen_tester'] = store_aggregation_chain
    params['node_edit_rate']    = [0, 0, 0, 0]  #change to force coarsening

    dummy_replica = algorithms.generate_graph(G, params=params)

    node_colors = {}
    aggregate_colors = {seed:(npr.rand(), npr.rand(), npr.rand(), 1.) for seed in list(c_tree[-1].values())}
    for node in G:
        my_final_agg = node
        for c_set in c_tree:
            my_final_agg = c_set[my_final_agg]  #this could be faster with union-find structure
        node_colors[node] = aggregate_colors[my_final_agg]
        clr = aggregate_colors[my_final_agg]
        G.node[node]['color'] = '%.3f %.3f %.3f'%(clr[0],clr[1],clr[2])
        G.node[node]['label'] = ''

    all_nodes = G.nodes()
    color_array = np.ones((len(all_nodes),4))
    for i,node in enumerate(all_nodes):
        color_array[i,:] *= node_colors[node] 

    #pos = nx.fruchterman_reingold_layout(G)
    #nx.draw_networkx_nodes(G, pos=pos, nodelist=G.nodes(), node_color=color_array, cmap=pylab.hot, node_size=500, with_labels=True, node_shape='s')
    #nx.draw_networkx_edges(G, pos=pos, alpha=1.0)
    #nx.draw_networkx_labels(G, pos=pos)
    #pylab.show()
    
    gpath     = 'output/coarsening_test_'+timeNow()+'.dot'
    gpath_fig = gpath+'.pdf'
    graphutils.write_graph(G=G, path=gpath)
    print('Writing graph image: %s ..'%gpath_fig)
    visualizer_cmdl = 'sfdp -Nwidth=0.10 -Nheight=0.10 -Nfixedsize=true -Nstyle=filled -Tpdf %s > %s &'%(gpath,gpath_fig)
    #visualizer_cmdl = 'sfdp -Nwidth=0.03 -Nheight=0.03 -Nfixedsize=true -Nstyle=solid  -Tpdf %s > %s &'%(gpath,gpath_fig)
    retCode = os.system(visualizer_cmdl)
    time.sleep(1)
    subprocess.call(['xdg-open', gpath_fig])

def drake_hougardy_test():
    import new_algs, graphutils

    matching_weight = lambda G, mat: sum(G.edge[u][mat[u]].get('weight', 1.0) for u in mat)/2.0
    def is_matching(mat):
        G = nx.Graph()
        G.add_edges_from(list(mat.items()))
        for cc in nx.connected_components(G):
            if len(cc) not in [0,2]:
                return False
        return True
    def is_maximal(G, mat):
        for edge in G.edges():
            if (edge[0] not in mat) and (edge[1] not in mat):
                return False 
        return True

    path = nx.path_graph(11)
    for u,v,d in path.edges(data=True):
        d['weight'] = max(u,v)**2
    matching = graphutils.drake_hougardy_slow(path)
    print('Matching slow: ' + str(matching))
    print('      wt: ' + str(matching_weight(path,matching)))
    matching = graphutils.drake_hougardy(path)
    assert is_matching(matching)
    assert is_maximal(path,matching)
    print('Matching: ' + str(matching))
    print('      wt: ' + str(matching_weight(path,matching)))
    path_opt_m = nx.max_weight_matching(path)
    print(' Opt Mat: ' + str(path_opt_m))
    print('      wt: ' + str(matching_weight(path,path_opt_m)))

    Gr2 = graphutils.load_graph('data-cyber-small/gr2.gml')
    matching = graphutils.drake_hougardy_slow(Gr2)
    print('Matching slow: ' + str(matching))
    print('      wt: ' + str(matching_weight(Gr2,matching)))
    matching = graphutils.drake_hougardy(Gr2)
    assert is_matching(matching)
    assert is_maximal(Gr2, matching)
    print('Matching: ' + str(matching))
    print('      wt: ' + str(matching_weight(Gr2,matching)))
    gr2_opt_m = nx.max_weight_matching(Gr2)
    print(' Opt Mat: ' + str(gr2_opt_m))
    print('      wt: ' + str(matching_weight(Gr2, gr2_opt_m)))

    #matching = graphutils.drake_hougardy(nx.erdos_renyi_graph(1000, 0.02))
    num_test_graphs = 100
    num_nodes = 400
    edge_density = 0.02
    seed = 0
    for trial in range(num_test_graphs):
        seed += 1
        Gnp = nx.erdos_renyi_graph(num_nodes, edge_density, seed=seed)
        print('Seed: %d'%seed)
        matching = graphutils.drake_hougardy(Gnp)
        assert is_matching(matching)
        assert is_maximal(Gnp, matching)
        wtDH = matching_weight(Gnp,matching)
        print('      wt  DH: ' + str(wtDH))
        gnp_opt_m = nx.max_weight_matching(Gnp)
        wtOpt = matching_weight(Gnp, gnp_opt_m)
        print('      wt Opt: ' + str(wtOpt))
        assert wtOpt <= 2*wtDH



def edge_attachment_test(seed=None):
    import math
    if seed==None:
        seed = npr.randint(1E6)
    print('rnd seed: %d'%seed)
    npr.seed(seed)
    random.seed(seed)

    nn = 30
    G = nx.watts_strogatz_graph(n=nn, k=4, p=0.0)
    print('All new edges should lie close to the cycle')

    pos = {node:(math.cos(float(node)/nn * math.pi * 2),math.sin(float(node)/nn * math.pi * 2)) for node in G}

    def visualize_rewiring(G, added_edges_set, deled_edges_set, tpl_data):
        old_G = G.copy()
        old_G.remove_edges_from(added_edges_set)
        old_G.add_edges_from(deled_edges_set)
        print('added edges: ')
        print(added_edges_set)
        print('deled edges: ')
        print(deled_edges_set)
        benchmarks.editing_demo_draw(G=old_G, new_G=G, seed=1, pos=pos)
        print(tpl_data)
        pylab.show()

    params = {}
    params['edit_edges_tester'] = visualize_rewiring
    params['edge_edit_rate']    = [0.10]
    params['node_edit_rate']    = [0.]
    params['node_growth_rate']  = [0.]
    params['verbose'] = True

    algorithms.generate_graph(G, params=params)


def evaluate_metrics(graphs, metrics, n_jobs=-1):
#evaluate a set of metrics on a set of graphs.  typically the first graph is the original graph
    vals_of_graphs = [[] for i in range(len(metrics))]

    if n_jobs==1: #other values are meaningful for joblib
        print()
        for graph_idx, graph in enumerate(graphs):
            rets = safe_metrics(graph, metrics)
            sys.stdout.write('.')
            for met_num,metric in enumerate(metrics):
                vals_of_graphs[met_num].append(rets[met_num])
            sys.stdout.flush()
    else:
        #first parallelization: all the replications
        print('Running parallel MEASUREMENT ...')
        sys.stdout.flush()
        graph_data  = Parallel(n_jobs=n_jobs, verbose=True)(delayed(safe_metrics)(graph, metrics) for graph in graphs)
        for rets in graph_data:
            for met_num,metric in enumerate(metrics):
                vals_of_graphs[met_num].append(rets[met_num])
        sys.stdout.flush()

    return vals_of_graphs


def evaluate_similarity(base_graphs, graphs, sim_metrics=None, n_jobs=-1):
#evaluate a set of metrics on a set of graphs.  typically the first graph is the original graph
    if sim_metrics == None:
        sim_metrics = [{'name':'jacc_edges', 'function':graphutils.graph_graph_delta}  ]  
        #TODO: this might be too slow b/c all changes are listed
    if (type(base_graphs) is not list) and (type(base_graphs) is not tuple):
        base_graphs = [base_graphs]*len(graphs)

    vals_of_graphs = [[] for i in range(len(sim_metrics))]

    if n_jobs==1: #other values are meaningful for joblib
        print()
        for graph_idx, graph in enumerate(graphs):
            base = base_graphs[graph_idx]
            rets = safe_similarity(base, graph, sim_metrics)
            sys.stdout.write('.')
            for met_num,metric in enumerate(sim_metrics):
                vals_of_graphs[met_num].append(rets[met_num])
            sys.stdout.flush()
    else:
        print('Running parallel SIMILARITY MEASUREMENT ...')
        sys.stdout.flush()
        graph_data  = Parallel(n_jobs=n_jobs, verbose=True)(delayed(safe_similarity)(base_graphs[graph_i], graph, sim_metrics) for graph_i, graph in enumerate(graphs))
        for rets in graph_data:
            for met_num,metric in enumerate(sim_metrics):
                vals_of_graphs[met_num].append(rets[met_num])
        sys.stdout.flush()

    mean_jac = np.average([d['jaccard_edges'] for d in rets])

    return mean_jac


def param_set_generator(default_params=None, base_vectors=None, edit_amplitude=0.05, fixed_set=None):
    if fixed_set != None:
        yield fixed_set
        return

    import itertools
    if default_params == None:
        base_vectors = \
          ({'name':'node_edit_rate',           'value_options':([0], [1], [0, 1], [0, 0, 1], [0, 0, 0, 1], [0, 0, 0.5, 0.5])},
           {'name':'edge_edit_rate',           'value_options':([0], [1], [0, 1], [0, 0, 1], [0, 0, 0, 1], [0, 0, 0.5, 0.5])},
           {'name':'locality_bias_correction', 'value_options':([0], [0.5], [-0.5], [0.5, 0.5],)},
           {'name':'new_edge_horizon',         'value_options':(3, 10)},
          )
    if default_params == None:
        default_params = {'num_v_cycles':10, 'verbose':False, 'dont_cutoff_leafs':False, 'enforce_connected':True, 'accept_chance_edges':1.0, }

    all_option_sets = [p for p in itertools.product(*tuple(v['value_options'] for v in base_vectors))]
    print('Total # of parameter sets: %d'%len(all_option_sets))
    for option_set_idx, option_set in enumerate(all_option_sets):
        param_set = default_params.copy()
        for param_idx, param_val in enumerate(option_set):
            param_set[base_vectors[param_idx]['name']] = param_val
        param_set['node_edit_rate'] = (edit_amplitude*np.array(param_set['node_edit_rate'])).tolist()
        param_set['edge_edit_rate'] = (edit_amplitude*np.array(param_set['edge_edit_rate'])).tolist()
        
        yield param_set
        
    return


def plot_deviation(vals_of_replicas, vals_of_graph, metrics, figpath, jaccard_edges=None, title_infix='', seed=0, Gname=''):
#vals_of_graph could be a number (level 0) or a list (the same as the number of replicas)
    clean_names = {'num nodes': 'num nodes', 'num edges':'num edges', 'clustering':'clustering', 'average degree':'avg\ndegree', 'degree assortativity':'degree\nassortativity', 'degree connectivity':'degree\nconnectivity', 
            'total deg*deg':'total deg*deg\nassortativity', 
            's-metric':'s metric', 'mean ecc':'avg\neccentricity', 'num comps':'num comps', 'L eigenvalue sum':'L eigen-\nvalue sum', 
            'average shortest path':'avg\ndistance', 'harmonic mean path':'harmonic avg\ndistance', 'avg flow closeness':'avg flow\ncloseness', 
            'avg eigvec centrality':'avg eigenvec.\ncentrality', 'avg between. central.':'avg between.\ncentrality', 'modularity':'modularity'}

    multiple_models = type(vals_of_graph[0]) is list

    pylab.show(block=False)
    fig = pylab.figure()
    pylab.hold(True)
    num_of_metrics = len(metrics)
    med_vals = [np.median(vals_of_replicas[i]) for i in range(num_of_metrics)]
    avg_vals = [np.average(vals_of_replicas[i]) for i in range(num_of_metrics)]
    p25_vals = [np.percentile(vals_of_replicas[i],25) for i in range(num_of_metrics)]
    p75_vals = [np.percentile(vals_of_replicas[i],75) for i in range(num_of_metrics)]
    max_vals = [np.max(vals_of_replicas[i]) for i in range(num_of_metrics)]
    min_vals = [np.min(vals_of_replicas[i]) for i in range(num_of_metrics)]
    std_vals = [np.std(vals_of_replicas[i]) for i in range(num_of_metrics)]

    replica_stats = {'median_of_replicas':med_vals, 'avg_of_replicas':avg_vals, 'p25_of_replicas':p25_vals, 'p75_of_replicas':p75_vals, 'max_of_replicas':max_vals, 'min_of_replicas':min_vals, 'std_of_replicas':std_vals}
    
    normed_replica_vals = []
    avg_norms  = []
    print('Medians' + (' (average of model graphs)' if multiple_models else ''))
    print('-------')
    print('metric\t\tOriginalG\t\tReplicas')
    for met_num,metric in enumerate(metrics):
        try:
            model_val = np.average(vals_of_graph[met_num]) if multiple_models else vals_of_graph[met_num]
            print('%s\t\t%.5f\t\t%.5f'%(metric['name'],model_val,med_vals[met_num]))
        except:
            print('%\tserror'%metric['name'])
    for met_num,metric in enumerate(metrics):
        #handle error in original, 0 in original, error in one replica, error in all replicas          
        nor_vals = []
        if multiple_models:
            assert len(vals_of_graph[met_num]) == len(vals_of_replicas[met_num])
            pruned_model_vals = [v for v in vals_of_graph[met_num] if v!=graphutils.METRIC_ERROR]
            if len(pruned_model_vals) > 0:
                v_graph = np.average(pruned_model_vals)
            else:
                v_graph = graphutils.METRIC_ERROR
        else:
            v_graph     = vals_of_graph[met_num]
        
        v_reps      = vals_of_replicas[met_num]
        if v_graph != graphutils.METRIC_ERROR:
            if v_graph != 0.0:
                nor_vals = [float(v)/v_graph for v in v_reps if v != graphutils.METRIC_ERROR]
            else:
                if v_reps != [] and np.abs(v_reps).sum() == 0.:
                    nor_vals.append(len(v_reps)*[1.0])
            pylab.plot(1.0, met_num, 'o', color='k', linewidth=2., label=Gname)
            pylab.text(x=.0, y=(met_num-2./len(metrics)), s='%.2e'%v_graph)
            #if type(v_graph) is int:
            #    pylab.text(x=.0, y=(met_num-2./len(metrics)), s=str(v_graph))
            #else:
            #    pylab.text(x=.0, y=(met_num-2./len(metrics)), s='%.3f'%v_graph)
            nor_vals = np.array(nor_vals)
            normed_replica_vals.append(nor_vals)
            if len(nor_vals) >0:
                pylab.boxplot(nor_vals, positions=[met_num], vert=0, widths=0.5)
                if (nor_vals == graphutils.METRIC_ERROR).any():
                    val_str= r'undefined'
                    avg_norm = -np.inf
                elif np.abs(nor_vals).sum() < 1000:
                    avg_norm = np.average(nor_vals)
                    val_str= r'$%.2f$'%np.average(nor_vals) if latex_available else r'%.2f'%avg_norm
                else:
                    avg_norm = np.inf
                    val_str= r'$\gg0$'if latex_available else r'>>0'
                avg_norms.append(avg_norm)
            else:
                val_str = r'undefined'
                avg_norms.append(None)
        else:
            val_str = r'undefined'
            normed_replica_vals.append([None, None])
            avg_norms.append(None)
        pylab.text(x=1.74, y=(met_num-2./len(metrics)), s=val_str)
    try:
        pylab.yticks(list(range(num_of_metrics)), [clean_names.get(met['name'], met['name']) for met in metrics], rotation=0)
        if multiple_models:
            pylab.xlabel(r'Relative to mean of coarse networks', rotation=0, fontsize='20')#, x=0.1)
        else:
            pylab.xlabel(r'Relative to real network', rotation=0, fontsize='20')#, x=0.1)
        #pylab.title(G.name)
        #pylab.legend(loc='best')
        max_axis = 2
        pylab.xlim(-0.02,max_axis)
        pylab.ylim(-1.0,len(metrics))
        pylab.text(x=0.00, y=len(metrics)+0.05, s='Template\ngraph', va='bottom')
        pylab.text(x=1.650, y=-1.05, s='Median of\nreplicas', va='top')
        if jaccard_edges != None:
            pylab.text(x=0.30, y=len(metrics)+0.05, s='(Jaccard=%.3f)'%jaccard_edges, va='bottom')
            #pylab.text(x=-0.30, y=len(metrics)*(-0.15), s='E[EdgeJaccard]=%.3f'%jaccard_edges, ha='right', va='top')

        fig.subplots_adjust(left=0.17, right=0.95)

        if figpath == None:
            figpath = 'output/replica_vs_original_'+Gname+'_'+title_infix+'_'+str(seed)+'__'+timeNow()
            figpath = clean_path(figpath)
        save_figure_helper(figpath)
        pylab.hold(False)
    except Exception as inst:
        print('Warning: could not save stats figure '+figpath + ':\n'+str(inst))
        exc_traceback = sys.exc_info()[2]
        print(str(inst) + "\n" + str(traceback.format_tb(exc_traceback)).replace('\\n', '\n'))

    replica_stats['normed_replica_vals'] = normed_replica_vals
    replica_stats['avg_norm_of_replicas'] = avg_norms

    mean_rel_errors = []
    mean_relstd_errors = []
    for met_i in range(num_of_metrics):
        normed_vals = normed_replica_vals[met_i]
        if graphutils.METRIC_ERROR in normed_vals or len(normed_vals) == 1:
            mean_rel_errors.append(None)
            mean_relstd_errors.append(None)
            continue
        rel_error_ar = [v - 1.0 for v in normed_vals if v != None]
        if len(rel_error_ar) == 0:
            rel_error_ar = [graphutils.METRIC_ERROR,  graphutils.METRIC_ERROR]
        mean_rel_errors.append(np.average(rel_error_ar))
        mean_relstd_errors.append(np.average(rel_error_ar)/(1E-20 + np.std(rel_error_ar)))
        
    replica_stats['mean_rel_errors'] = mean_rel_errors
    replica_stats['mean_relstd_errors'] = mean_relstd_errors
    try:
        replica_stats['mean_mean_error']    = np.average(mean_rel_errors)     #the grand stat
        replica_stats['mean_mean_errorstd'] = np.average(mean_relstd_errors)  #the grand stat
    except:
        replica_stats['mean_mean_error']    = None
        replica_stats['mean_mean_errorstd'] = None

    return replica_stats, figpath


def replicate_graph(G, generator_func, num_replicas, params, title_infix='', n_jobs=-1):
    if n_jobs==1: #other values are meaningful for joblib
        print()
        print(getattr(G, 'name', '') + ' ' + title_infix)
        replicas = []
        for replica_idx in range(num_replicas):
            replica = generator_func(G, params=params)
            replicas.append(replica)
            sys.stdout.write('.')
            sys.stdout.flush()
    else:
        print('Running parallel GENERATION ...')
        replicas = Parallel(n_jobs=n_jobs, verbose=True)(delayed(generator_func)(G, params) for i in range(num_replicas))
        print('   %d replicas done'%len(replicas))
        sys.stdout.flush()

    return replicas


def replica_vs_original(seed=None, figpath=None, generator_func=None, G=None, params=None, num_replicas = 150, title_infix='', metrics=None, intermediates=False, n_jobs=-1):
#generate one or more replicas and compare them to the original graph
    if seed==None:
        seed = npr.randint(1E6)
    print('rand seed: %d'%seed)
    npr.seed(seed)
    random.seed(seed)

    if generator_func==None:
        generator_func=algorithms.generate_graph

    if G==None:
        G = graphutils.load_graph(path='data-social/potterat_Hiv250.elist')

    if metrics == None:
        metrics = graphutils.default_metrics[:]
    metrics = [m for m in metrics if m['optional'] < 2]
    if 'metric_runningtime_bound' in params:
        mrtb = params['metric_runningtime_bound']
        metrics = [m for m in metrics if m['runningtime'] <= mrtb]
    metrics = [m for m in metrics if m['name'] not in ['avg flow closeness']] #broken in NX 1.6
    metrics.reverse()

    if params == None:
        params  = {'verbose':False,  'node_edit_rate':[0.05, 0.04, 0.03, 0.02, 0.01], 
                'edge_edit_rate':[0.05, 0.04, 0.03, 0.02, 0.01], 'node_growth_rate':[0], 'locality_bias_correction':0., 'enforce_connected':True, 'accept_chance_edges':1.0,
                'retain_intermediates':intermediates}
    if intermediates:
        params['retain_intermediates'] = True
    print('Params:')
    print(params)
    print('Metrics:')
    print([metric['name'] for metric in metrics])

    replicas         = replicate_graph(G=G, generator_func=generator_func, num_replicas=num_replicas, params=params, title_infix=title_infix, n_jobs=n_jobs)
    jaccard_edges    = evaluate_similarity(base_graphs=G, graphs=replicas, n_jobs=n_jobs)  #this is actually a mean
    vals_of_all      = evaluate_metrics(graphs=[G]+replicas, metrics=metrics, n_jobs=n_jobs)
    vals_of_graph    = [metric_data[0]  for metric_data in vals_of_all]
    vals_of_replicas = [metric_data[1:] for metric_data in vals_of_all]
    replica_statistics, figpath = plot_deviation(vals_of_replicas, vals_of_graph, metrics, figpath, jaccard_edges, title_infix, seed, getattr(G, 'name', ''))
    #pylab.show()
    data = {'metrics':[met['name'] for met in metrics], 'name':getattr(G, 'name', ''), 'params':params, 'num_replicas':num_replicas, 'figpath':figpath}
    data[0] = replica_statistics
    data[0].update({'vals_of_replicas':vals_of_replicas, 'val_of_models':vals_of_graph, 'avg_jaccard_edges':jaccard_edges})

    if intermediates:
        current_replicas = replicas
        for level in range(1, max(len(params.get('node_edit_rate', [])), len(params.get('edge_edit_rate', [])), len(params.get('node_growth_rate', [])), len(params.get('edge_growth_rate', [])))):
            print('LEVEL: %d'%level)
            coarse_models   = [r.coarser_graph.model_graph  for r in current_replicas]
            coarse_replicas = [r.coarser_graph              for r in current_replicas]
            vals_of_models   = evaluate_metrics(graphs=coarse_models,   metrics=metrics, n_jobs=n_jobs)
            vals_of_replicas = evaluate_metrics(graphs=coarse_replicas, metrics=metrics, n_jobs=n_jobs)
            jaccard_edges    = evaluate_similarity(base_graphs=coarse_models, graphs=coarse_replicas, n_jobs=n_jobs)

            replica_statistics, dummy \
                 = plot_deviation(vals_of_replicas=vals_of_replicas, vals_of_graph=vals_of_models, 
                                  metrics=metrics, figpath=figpath + 'level%d'%level, jaccard_edges=jaccard_edges)
            current_replicas = coarse_replicas
            data[level] = replica_statistics
            data[level].update({'vals_of_replicas':vals_of_replicas, 'vals_of_models':vals_of_models, 'avg_jaccard_edges':jaccard_edges})
    graphutils.safe_pickle(path=figpath+'.pkl', data=data)
    save_param_set(params, seed, figpath)
    save_stats_csv(path=figpath+'.csv', seed=seed, data=data)

    return data


def safe_metrics(graph, metrics):
    rets = []
    for met_num,metric in enumerate(metrics):
        try:
            rets.append(metric['function'](graph))
        except Exception as inst:
            print('error in computing: '+metric['name'])
            print(inst)
            rets.append(graphutils.METRIC_ERROR)
    return rets

def safe_similarity(graph, new_graph, metrics):
    rets = []
    for met_num,metric in enumerate(metrics):
        try:
            rets.append(metric['function'](graph, new_graph))
        except Exception as inst:
            print('error in computing: '+metric['name'])
            print(inst)
            rets.append(graphutils.METRIC_ERROR)
    return rets


def save_figure_helper(fpath):
    if matplotlib.get_backend() == 'pdf':
        final_path = fpath + '.pdf'
        pylab.savefig(final_path)
        pylab.close()
        print('Written: %s'%final_path)
        print('Converting to eps...')
        os.system('pdftops -eps ' + final_path)
        os.rename(fpath+'.eps', fpath+'_.eps') #missing bounding box
        os.system('eps2eps ' + fpath+'_.eps' + '  ' + fpath+'.eps')
        os.remove(fpath+'_.eps')
    elif matplotlib.get_backend() == 'ps':
        final_path = fpath + '.eps'
        pylab.savefig(final_path)
        pylab.close()
        print('Written: %s'%final_path)
        print('Converting to pdf...')
        os.system('epstopdf ' + final_path)
    else:
        print('Trying to save to PDF.  Backend: %s'%matplotlib.get_backend())
        final_path = fpath + '.pdf'
        pylab.savefig(final_path)
        pylab.close()
        print('Written: %s'%final_path)
    subprocess.call(['xdg-open', final_path])

def save_param_set(param_set, seed, fpath):
    with open(fpath+'_params.txt', 'w') as f:
        f.write('parameters' + os.linesep)
        for p,v in list(param_set.items()):
            f.write('%s: %s'%(p,str(v)) + os.linesep)
        f.write('%s: %s'%('seed',str(seed)) + os.linesep)
        print('Written parameter set to: %s'%f.name)
    
def save_stats_csv(path, seed, data):
    header = []
    rets   = []

    header.append('date-time');   rets.append(timeNow())
    header.append('graph name');  rets.append(data['name'])
    header.append('figpath');     rets.append(data['figpath'])
    header.append('seed');        rets.append(seed)
    valid_param_names = list(simpletesters.valid_params.keys())
    valid_param_names.sort()
    for p in valid_param_names:
        myparams = data['params']
        header.append(p)
        if p in myparams:
            rets.append(myparams[p])
        else:
            rets.append('')

    metric_names = data['metrics']
    for level in range(100):
        if level not in data:
            break
        print('LEVEL: %d'%level)
        level_stats = data[level]
        header.append('level%d'%level); rets.append('--->')
        header.append('avg_jaccard_edges'); rets.append(level_stats['avg_jaccard_edges'])
        header.append('mean_mean_error');   rets.append(level_stats['mean_mean_error'])
        header.append('mean_mean_errorstd');   rets.append(level_stats['mean_mean_errorstd'])
        num_metrics = len(metric_names)
        distributional_stats = [k for k in level_stats if hasattr(level_stats[k], '__len__')] #e.g. median, avg, min, max ...
        distributional_stats = [k for k in distributional_stats if k not in ('avg_jaccard_edges',)]
        distributional_stats = [k for k in distributional_stats if len(level_stats[k]) == num_metrics]
        distributional_stats.sort()
        for met_idx, metric in enumerate(metric_names):
            for stat in distributional_stats:
                ret = level_stats[stat][met_idx]
                if hasattr(ret, '__len__'):
                    continue
                header.append(metric+' '+stat); rets.append(ret)

    separator = ','
    with open(path, 'w') as f:
        f.write(separator.join(['"' + str(v) + '"' for v in header]) + os.linesep)
        f.write(separator.join(['"' + str(v) + '"' for v in rets]) + os.linesep)
    print('Writen report: ' + path)
        

def statistical_tests(seed=8):
#systematic comparison of a collection of problems (graphs and parameters)
    if seed==None:
        seed = npr.randint(1E6)
    print('rand seed: %d'%seed)
    npr.seed(seed)
    random.seed(seed)

    default_num_replicas = 20
    
    params_default  = {'verbose':False, 'edge_edit_rate':[0.08, 0.07], 'node_edit_rate':[0.08, 0.07], 'node_growth_rate':[0], 
            'dont_cutoff_leafs':False,
            'new_edge_horizon':10, 'num_deletion_trials':20, 'locality_bias_correction':[0,], 'edit_method':'sequential',
            }
    #params_default['algorithm'] = algorithms.musketeer_on_subgraphs

    metrics_default = graphutils.default_metrics[:]
    #some metrics are removed because of long running time
    metrics_default  = [met for met in metrics_default if met['name'] not in ['avg flow closeness', 'avg eigvec centrality', 'degree connectivity', 'degree assortativity',  'average shortest path', 'mean ecc', 'powerlaw exp', ]]
    problems = [{'graph_data':nx.erdos_renyi_graph(n=300, p=0.04, seed=42), 'name':'ER300', 'num_replicas':20},
                {'graph_data':'data-samples/ftp3c.elist'},
                {'graph_data':'data-samples/mesh33.edges'},
                {'graph_data':'data-samples/newman06_netscience.gml', 'num_replicas':10},

                {'graph_data':'data-samples/watts_strogatz98_power.elist', 'num_replicas':10},
               ]

    for problem in problems:
        graph_data    = problem['graph_data']
        params        = problem.get('params', params_default)
        metrics       = problem.get('metrics', metrics_default)
        num_replicas  = problem.get('num_replicas', default_num_replicas)

        if type(graph_data) is str:
            base_graph = graphutils.load_graph(path=graph_data)
            base_graph.name = os.path.split(graph_data)[1]
        else:
            base_graph = graph_data
            if not hasattr(base_graph, 'name'):
                base_graph.name = problem.get('name', str(npr.randint(10000)))

        gpath     = 'output/'+os.path.split(base_graph.name)[1]+'_'+timeNow()+'.dot'
        gpath_fig = gpath[:-3]+'eps'
        graphutils.write_graph(G=base_graph, path=gpath)
        visualizer_cmdl = 'sfdp  -Nlabel="" -Nwidth=0.03 -Nfixedsize=true -Nheight=0.03 -Teps %s > %s &'%(gpath,gpath_fig)
        print('Writing graph image: %s ..'%gpath_fig)
        retCode = os.system(visualizer_cmdl)
        
        replica_vs_original(G=base_graph, num_replicas=num_replicas, seed=1, params=params, metrics=metrics, title_infix='musketeer')


if __name__ == '__main__': 
    pass
    #drake_hougardy_test()
    #coarsening_test()
    #coarsening_test2(1)
    #edge_attachment_test(seed=None)
    #print 'Statistical tests: this would take time ...'
    #statistical_tests()
    replica_vs_original(G=graphutils.load_graph('data-samples/mesh33.edges'), params={'edge_edit_rate':[0.01, 0.01]}, num_replicas=2, n_jobs=1)
