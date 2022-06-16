Multiscale Entropic Network Generator 2 (MUSKETEER2)

Copyright (c) 2011-2018 by Alexander Gutfraind and Ilya Safro. 
All rights reserved.

Use and redistribution of this file is governed by the license terms in
the LICENSE file found in the project's top-level directory.

Setup and Installation
----------------------
Make sure your system has 
1. Python 2.6 or 2.7
2. NetworkX library (version >= 1.10)
3. numpy library (version >=1.5)

You can check this by running 

python

From Python:

import sys, numpy, networkx
sys.version
numpy.__version__
networkx.__version__

Also,
graphviz is recommended for visualization.



Usage example
-------------
python musketeer.py -f data-samples/arenas_email.edges -t edgelist -p "{'node_growth_rate':[0.005, 0.001], 'edge_edit_rate':[0.05, 0.04, 0.03], 'node_edit_rate':[0.07, 0.06, 0.05]}" -o output/test.dot

explanation:
1. Loads data-social/arenas_email.edges
2. Increases the number of nodes by 0.9% at level 1 deep, and by 0.1% at level 0 (finest level)
3. Edits 3% of the edges and 5% of the nodes at level 2 deep, 
   4% of edges at level 1 deep and 6% of the nodes,
   5% of edges at level 0 deep and 7% of the nodes
4. The graph is outputed to output/test.dot (DOT format used by graphviz)

#note that step 2 changes the network statistics, and is often skipped in practice

#a direct call from python script:
import algorithms
new_G = algorithms.generate_graph(G, params)



Special usage modes
--------------------
node_growth_rate:
This parameter accepts values in (-1,infinity) at every level.  
When set to values 1 or greater it will establish number_of_current_nodes*floor(node_growth_rate) new nodes,
and then add number_of_current_nodes*(node_growth_rate%1) in expectation.

edge_growth_rate:
As above, but for edges

algorithm:
Optional replication with one of several alternative generators (not MUSKETEER), that are implemented in algernatives.py
For example, to use the Expected Degrees Model, use:
-p "{'algorithm':alternatives.expected_degree_replicate}"


'algorithm':algorithms.musketeer_snapshots:
Parameter causes editing sequentially: G->R1->R2->...->Rn and return a sequence of snapshots.
The edit rate parameter is applied at every generation step.
Use
'num_snapshots':num_snapshots    (any positive integer)
The output returns Rn.  The attribute .snapshots contains a list of all the graphs from G to Rn.


Algorithm Tuning
----------------
accept_chance_edges:
Chance edges are those edges that inserted regardless of the distance between u and v.
The rate of such edges is determined when computing topological/structural data
but is often spuriously elevated because of computational limitation (see edge horizon)
Graphs like the Watts-Strogatz graph have a few but very significant long-distance edges.
Decreasing accept_chance_edges from 1.0 (default) closer to 0.0 would reduce spurious chance edges.

'algorithm':algorithms.musketeer_iterated_cycles
Alternating editing: calls the main algorithm multiple times passing the output to the input.
The edit and growth rates are split over the cycles 
in order to gradually change the graph, increasing realism.
(e.g. two cycles of [0.05, 0.01] + [0.05, 0.01] instead of one [0.1, 0.02])
Use the specification:
'num_v_cycles':num_v_cycles    (any positive integer)


deep_copying:
When True (default), new coarse nodes (aggregate) precisely match the structure of an existing aggregate,
and this is true at all levels of refinement.
Previously, at each level of refinement, random resampling was applied to interpolate the aggregate.

deferential_detachment_factor:
Default 1.  
Changes the probability of deleting an edge (u,v) to approximately 1/(degree(u) * degree(v) )  
Reducing this closer to 0 would disable deferential detachment mechanism.

dont_cutoff_leafs:
When true, prevents edge editing from removing connection to a leaf node (boolean)
Not recommended, since it skews the degree distribution.

edge_welfare_fraction:
Increases the likelihood that nodes which lost an edge will be given a replacement.  Ranges 0..1
Helps recover from bias in clustering.

enforce_connected:
This parameter (boolean) will insert new edges from small components to the giant component.
The default behavior is to enforce connectedness if the original graph was connected.

An alternative handling of disconnected graphs is to apply musketeer on each of the components of the graph.
Use the parameter 
'algorithm':algorithms.musketeer_on_subgraphs
This keeps their sizes the same.

fine_clustering:
When True (default: False), the addition of edges considers the number of triangles that would be created.
This slightly increases fidelity at somewhat high computational cost.

locality_bias_correction:
Increases the number of triangles generated during edge insertion.  
Ranges -1..1 at each level, similar to edge_edit_rate: [0.6, 0.7]
Positive values help recover from negative bias in clustering.

maintain_edge_attributes:
When True (default False), preserves attributes of edges (weight, strengths, friendship type). 
Attributes of new edges are copied from attributes of existing edges.

maintain_node_attributes
When True (default False), preserves attributes of nodes (such as color, body mass, electrical output etc) 
Attributes of new nodes are copied from attributes of existing nodes.
 
memoriless_interpolation:
When True (default False), interpolation is applied even to nodes and edges which were not edited.
Data on aggregates that was created during coarsening is not used.
This option provides randomization of the network and may be useful for anonymization (no guarantees are given).

minorizing_node_deletion:
When True, deleted nodes are replaced by a tree that connects all their old neighbors.

new_edge_horizon:
This parameter (positive integer) determines the depth of scanning when inserting new edges.
Increasing this value increases fidelity but slows the algorithm, especially when the graph is dense.
The default value is determined automatically based on the density of the graph.

post_processor:
Runs the replica through an algorithm with the pattern f(new_G, original, params) before saving the replica
try 'post_processor':graphutils.color_new_nodes_and_edges

preserve_clustering_on_deletion:
When True (default), significantly helps increase the fidelity of clustering in replicas at a modest overhead and decrease in entropy.
It decreases the probability that an edge (u,v) is deleted when u and v have many mutual neighbors.

retain_intermediates:
When True (default: False), intermediate graphs are retained and accessible from new_G.
  new_G.coarser_graph = the final result of the previous level
  new_G.model_graph       = the input graph to the previous level retain_intermediates (i.e. the graph G_i, which is horizontally on the V cycle)
coarser_graph is a linked list (new_G.coarser_graph.coarser_graph...) which terminates with None at the coarsest level.


stats_report_on_all_levels:
When True (default: False), compares the coarsen graphs at each level, showing how the original and edited versions differ.

suppress_warnings:
When True (default: False), warnings are not displayed and the algorithm is almost completely silent.

verbose:
Enables detailed output (boolean).


Troubleshooting
---------------
Accelerating the computation
* remove the computation of the metrics with "-M False" argument (all metrics);  or in graphutils.py, make some complex metrics optional.
* convert the original graph into a simple format like edgelist
* reduce the new_edge_horizon (reduces fidelity)
* set deferential_detachment_factor close to 0 (reduces fidelity)
* give the authors funding to develop a version in the C language

Clustering is diminished in the replicas
* make sure that the pattern is consistent - there would be and need to be some variance
* reduce the edge editing rate at the finest level: it would improve preservation of all fine-level properties of the graph
* increase locality_bias_correction
* make fine_clustering True
* make preserve_clustering_on_deletion True

Path lengths and related global metrics are diminished in the replicas
* make accept_chance_edges closer to 0.  As a compensation to maintain fidelity, increase new_edge_horizon to 10, 20 or larger.

Support for weighted edges
* currently only used in coarsening
* we are planning to include it in future releases

Support for node and edge attributes
* use the parameters 'maintain_node_attributes':True, 'maintain_edge_attributes':True

Error writing DOT file or pygraphviz error:
* pygraphviz is not currently functional in the Windows platform
* specify an alternative output file such as "-o my_output.elist"
* or, try install/ing pydot package

