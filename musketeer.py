#! /usr/bin/python
'''
Multiscale Entropic Network Generator 2 (MUSKETEER2)

Copyright (c) 2011-2015 by Alexander Gutfraind and Ilya Safro. 
All rights reserved.

Use and redistribution of this file is governed by the license terms in
the LICENSE file found in the project's top-level directory.


Main user program

- processes user input and reads data
- calls algorithms
- returns replicas

'''

citation=[
"\\cite{musketeer}",
"",
"@inproceedings{musketeer2",
"  author = {Gutfraind, Alexander and Meyers, Lauren A. and Safro, Ilya},",
"  title  = {{Multiscale Network Generation}},",
"  booktitle= {Proceedings of the  International Conference on Information Fusion {FUSION}15 },",
"  number = {},",
"  issue  = {},",
"  year   = {2015},",
"  note   = {see also {arxiv.org/abs/1207.4266}},",
"  url    = {FIXME},",
"  doi    = {},",
"  address  = {Washington, {DC}},",
"}",]

import os
import time
import numpy as np
import numpy.random as npr
import random, sys
import networkx as nx
import matplotlib
matplotlib.use('PDF')
#import matplotlib.pylab as pylab
#import pylab
import getopt
import re
import pdb
import pickle
import algorithms
import graphutils
import simpletesters
import alternatives
try:
    import new_algs #for testing
except:
    pass

np.seterr(all='raise')
timeNow = lambda : time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())

version = '2015-04'

def initialize():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'M:chf:t:v:s:Tp:o:v:w:', ['help', 'citation', 'input_path=', 'output_path=', 'graph_type', 'seed=', 'test', 'params=', 'visualizer=', 'verbose=', 'metrics:'])
    except Exception as inst:
        print('Error parsing options:')
        print(inst)
        show_usage()
        sys.exit(1)

    input_path = None
    sgiven     = False
    graph_type = 'AUTODETECT'
    verbose    = True
    write_graph= True
    compare_replica = False
    input_path = None
    output_path= None
    visualizer = None
    params = {}

    if opts == []:
        try:
            return input_prompter()
        except:
            show_usage()
            sys.exit(1)

    for o, a in opts:
       if o in ('-c', '--citation'):
          print("Please cite:")
          print("\n".join(citation))
          sys.exit(0)
       if o in ('-h', '--help'):
          show_usage()
          sys.exit(0)
       elif o in ('-f', '--input_path'):
          input_path = a
       elif o in ('-M', '--metrics'):
          compare_replica = (a.lower() == 'true')
       elif o in ('-o', '--output_path'):
          output_path = a
       elif o in ('-p', '--params'):
          try:
              params.update(eval(a.strip()))
          except Exception as inst:
              print('Error parsing parameters!  Given:')
              print(a)
              raise
       elif o in ('-s', '--seed'):
          sgiven = True  
          random.seed(int(a))
          npr.seed(int(a))
          print('Warning: the random SEED was specified.')
       elif o in ('-t', '--graph_type'):
          graph_type = a
       elif o in ('-T', '--test'):
          simpletesters.smoke_test()
          sys.exit(0)
       elif o in ('-v', '--visualizer'):
          visualizer = a
       elif o in ('--verbose'):
          verbose = (a.lower() != 'false')
          params['verbose'] = verbose
       elif o in ('-w', '--write_graph'):
          write_graph = (a.lower() != 'false')
       else:
          print('Unrecognized option: %s'%o)
          show_usage()
          sys.exit(1)

    if input_path == None or not os.path.exists(input_path):
        raise IOError('Cannot open: '+str(input_path))

    if not sgiven:
        seed = npr.randint(1E6)
        if verbose:
            print('random number generator seed: %d'%seed)
        random.seed(seed)
        npr.seed(seed)

    ret = {'params':params, 'input_path':input_path, 'graph_type':graph_type, 'visualizer':output_path, 'output_path':output_path, 'visualizer':visualizer, 'verbose':verbose, 'compare_replica':compare_replica, 'write_graph':write_graph}
    return ret


def input_default(prompt, default_value, pythonize=True):
    try:
        input_str = input(prompt)
        if input_str != '':
            if pythonize:
                return eval(input_str.strip())
            else:
                return input_str.strip()
        else:
            return default_value
    except:
        return default_value

def input_prompter():
    show_banner()
    input_path       = 'data-samples/arenas_email.edges'
    graph_type       = 'AUTODETECT'
    edge_edit_rate   = [0.01, 0.01, 0.01, 0.01]
    node_edit_rate   = [0.01, 0.01, 0.01, 0.01]
    node_growth_rate = [0.05, 0.01]
    edge_growth_rate = [0.05, 0.01]

    print('Please enter the following information:')
    input_path       = input_default('File path to input graph (default: %s): '%str(input_path), input_path, pythonize=False)
    if (not input_path) or not os.path.exists(input_path):
        print('Cannot read: '+str(input_path))
        print()
        raise IOError('Cannot read: '+str(input_path))
    else:
        print('found.')

    graph_type       = input_default('     file format (default: %s): '%str(graph_type), graph_type)
    edge_edit_rate   = input_default('Edge edit rate (default: %s): '%str(edge_edit_rate), edge_edit_rate) 
    node_edit_rate   = input_default('Node edit rate (default: %s): '%str(node_edit_rate), node_edit_rate) 
    node_growth_rate = input_default('Node growth rate (default: %s): '%str(node_growth_rate), node_growth_rate) 
    edge_growth_rate = input_default('Edge growth rate (default: %s): '%str(edge_growth_rate), edge_growth_rate) 
    compare_replica  = input_default('Compare output to original (default: %s): '%str(False), False) 
    try:
        [float(x) for x in edge_edit_rate]
        [float(x) for x in node_edit_rate]
        [float(x) for x in node_growth_rate]
        [float(x) for x in edge_growth_rate]
    except:
        print('Error parsing input!')
        print() 
        sys.exit(1)


    params = {'edge_edit_rate':edge_edit_rate, 'node_edit_rate':node_edit_rate, 'node_growth_rate':node_growth_rate, 'edge_growth_rate':edge_growth_rate}

    ret = {'params':params, 'compare_replica':compare_replica, 'input_path':input_path, 'graph_type':graph_type, 'output_path':None, 'visualizer':None, 'verbose':True, 'write_graph':True}
    return ret


def open_in_unix(image_path, verbose, ext='pdf'):
    cmdls = ['xdg-open %s'%image_path,
             'acroread %s'%image_path,
             'xpdf     %s'%image_path,]

    for cmdl in cmdls:
        if verbose:
            print(cmdl)
        ret=os.system(cmdl)
        if ret == 0: 
            return
    if verbose:
        print('Unable to open image file')


def show_banner():
    print('########################################################################')
    print('######### Multiscale Entropic Network Generator 2 (MUSKETEER2) #########')
    print('########################################################################')
    print('version '+str(version))
    print('for a list of options run with -h flag')
    print()

def show_usage():
    print('Multiscale Entropic Network Generator 2 (MUSKETEER2)')
    print('Allowed options are:')
    print('-c, --citation    Citation information for MUSKETEER 2')
    print('-f, --input_path  Input graph file path')
    print('-h, --help        Shows these options')
    print('-M, --metrics     Compare the replica to the original.  Computing intensive. (Default: -M False).')
    print('-o, --output_path Path to the output file for the graph.')
    print('                  Output format is chosen automatically based on the extension.')
    print('-p, --params      Input paremeters.  Surround the argument with double quotes:')
    print('                  e.g. -p "{\'p1_name\':p1_value, \'p2_name\':p2_value}"')
    print('                  Key parameters: edge_edit_rate, node_edit_rate, node_growth_rate, edge_growth_rate (all are lists of values e.g. [0.01, 0.02])')
    print('-s, --seed        Random seed (integer)')
    print('-T, --test        Run a quick self-test')
    print('-t, --graph_type  Specify the format of the input graph (Default: -t AUTODETECT)')
    print('-v, --visualizer  Visualization command to call after the replica has been prepared (Default: -v None). Try -v sfdp or -v sfdp3d')
    print('--verbose         Verbose output (Default: --verbose True)')
    print('-w, --write_graph Write replica to disc (Default: -w True).')
    print('                  For interactive Python make False to speed up generation (disables visualization).')
    print() 
    print('For reading graphs with -t, the supported graph types are: \n%s'%graphutils.load_graph(path=None, list_types_and_exit=True))
    print() 
    print('For writing graphs with -o, the supported graph extensions are: \n%s'%graphutils.write_graph(G=None, path=None, list_types_and_exit=True))
    print() 
    print() 
    print('Example call format:')
    print(graphutils.MUSKETEER_EXAMPLE_CMD)




if __name__ == '__main__': 
    init_options = initialize()
    input_path   = init_options['input_path']
    params       = init_options['params']
    graph_type   = init_options['graph_type']
    output_path  = init_options['output_path']
    visualizer   = init_options['visualizer']
    verbose      = init_options['verbose']
    write_graph  = init_options['write_graph']

    if verbose:
        print('Loading: %s'%input_path)
    G = graphutils.load_graph(path=input_path, params={'graph_type':graph_type, 'verbose':verbose})

    if verbose:
        print('Generating ...')
    new_G = algorithms.generate_graph(G, params=params)

    #optional
    #print graphutils.graph_graph_delta(G=G, new_G=new_G)
    #new_G = nx.convert_node_labels_to_integers(new_G, 1, 'default', True)

    #TODO: too many reports
    if params.get('stats_report_on_all_levels', False):
        model_Gs = [new_G.model_graph]
        Gs       = [new_G]
        current_G = new_G.coarser_graph
        while current_G.coarser_graph != None:
            Gs.      append(current_G)
            model_Gs.append(current_G.model_graph)
            current_G = current_G.coarser_graph
        model_Gs.reverse()
        Gs.      reverse()
        for model_G, new_graph in zip(model_Gs, Gs):
            graphutils.compare_nets(model_G, new_graph, params={'verbose':True, 'normalize':True})

    new_G = params.get('post_processor', lambda G,original,params:G)(G=new_G, original=G, params=params)

    if output_path == None:
        t_str = timeNow()
        if not os.path.exists('output'):
            os.mkdir('output')
        if not os.path.isdir('output'):
            raise ValueError('Cannot write to directory "output"')
        output_base = 'output/'+os.path.splitext(os.path.basename(input_path))[0]
        output_path     = output_base + '__' + t_str + '.dot'
        output_path_adj = output_base + '__' + t_str + '.adjlist'
        if write_graph: 
            if verbose:
                print('Saving graph: %s'%output_path_adj)
            sys.stdout.flush()
            nx.write_adjlist(new_G, output_path_adj)
    if write_graph: 
        if verbose:
            print('Saving graph: %s'%output_path)
        sys.stdout.flush()
        graphutils.write_graph(new_G, output_path)
    image_path  = output_path + '.pdf'
    stderr_path = output_path + '.err.txt'

    if init_options['compare_replica']:
        if verbose:
            print('Generator Report')
        sys.stdout.flush()
        graphutils.compare_nets(G, new_G, params=params)

    #0.03 is too small for Linux
    #sfdp_default_cmd = 'sfdp -Goverlap="prism100" -Goverlap_scaling=-100 -Nlabel="" -Nwidth=0.01 -Nfixedsize=true -Nheight=0.01'
    sfdp_default_cmd = 'sfdp -Nlabel="" -Nwidth=0.06 -Nfixedsize=true -Nheight=0.06 -Nstyle=filled'
    if write_graph and visualizer == 'sfdp' and output_path[-3:] == 'dot':
        visualizer_cmdl = sfdp_default_cmd +' -Tpdf %s > %s 2> %s '%(output_path,image_path,stderr_path)
        if verbose:
            print('Writing graph image: %s ..'%image_path)
        sys.stdout.flush()
        retCode = os.system(visualizer_cmdl)

        if verbose:
            print(visualizer_cmdl)
        if os.name == 'nt':
            pdf_cmdl = 'start %s'%image_path
            if verbose:
                print(pdf_cmdl)
            os.system(pdf_cmdl)
        elif os.name == 'posix':
            #aside: file --mime-type -b my.pdf
            open_in_unix(image_path, verbose=verbose, ext='pdf')
    elif write_graph and visualizer == 'sfdp3d':
        tmp_path = output_path+'_tmp'
        visualizer_cmdl = sfdp_default_cmd +' -Gdimen=3 -Txdot %s > %s 2> %s '%(output_path,tmp_path,stderr_path)
        if verbose:
            print('Writing graph with coordinates: %s ..'%tmp_path)
        sys.stdout.flush()
        retCode = os.system(visualizer_cmdl)

        replica_name = new_G.name
        new_G = graphutils.color_by_3d_distances(nx.read_dot(tmp_path), verbose)
        new_G.name = replica_name
        if verbose:
            print('Saving graph with updated layout and color: %s'%output_path)
        sys.stdout.flush()
        graphutils.write_graph(new_G, output_path)

        visualizer_cmdl = sfdp_default_cmd +' -Tpdf %s > %s 2> %s '%(output_path,image_path,stderr_path)
        if verbose:
            print('Writing graph image: %s ..'%image_path)
        sys.stdout.flush()
        retCode = os.system(visualizer_cmdl)

        if verbose:
            print(visualizer_cmdl)
        if os.name == 'nt':
            pdf_cmld = 'start %s'%image_path
            if verbose:
                print(pdf_cmld)
            os.system(pdf_cmld)
        elif os.name == 'posix':
            #aside: file --mime-type -b my.pdf
            open_in_unix(image_path, verbose=verbose, ext='pdf')
    elif write_graph and visualizer != None:
        if verbose:
            print('Running visualizer: ' + str(visualizer))
        sys.stdout.flush()
        visualizer_cmdl = visualizer + ' ' + output_path
        retCode = os.system(visualizer_cmdl)

        if verbose:
            print(visualizer_cmdl)

    graphutils.safe_pickle(path=output_path+'.pkl', data=new_G, params=params)
    if verbose:
        print('Replica is referenced by variable: new_G')
