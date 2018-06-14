import os, sys
import networkx as nx
import operator
import matplotlib.pyplot as plt
import glob

def get_max_degree_node(x):
	sorted_x = sorted(x.items(), key=operator.itemgetter(1))
	return sorted_x[-1][0]


if len(sys.argv) != 3:
	sys.exit("Usage: coarsegrain_dependency_network.py [results_folder] [output.txt]")

results_folder = sys.argv[1]
outfname = sys.argv[2]

G = nx.Graph()

print "Reading..."
for fname in glob.glob(results_folder + "/ID_*"):
	with open(fname, "r") as infile:
		for line in infile.readlines():
			cell1, cell2, dist = line.split()
			cell1 = int(cell1)
			cell2 = int(cell2)
			dist = float(dist)

			G.add_edge(cell1, cell2, weight=1.0/dist)
num_nodes_remaining = len(G)
mapping = [None] * num_nodes_remaining
iterations = 0

print "Coarsegraining..."
while num_nodes_remaining > 0:

	# get max degree node
	max_deg_node = get_max_degree_node(G.degree())
#	print "Max degree node = ", max_deg_node

	# Map this node to use its own MD results
	mapping[max_deg_node] = max_deg_node

	neighbours = [max_deg_node]
	for neighbour in nx.all_neighbors(G, max_deg_node):
		neighbours.append(neighbour)
		mapping[neighbour] = max_deg_node

#	print "Neighbours:", neighbours

#	print "Removing", len(neighbours)
	G.remove_nodes_from(neighbours)

	num_nodes_remaining = len(G)
#	print "Num nodes remaining", len(G)

	iterations += 1

print "Converged in", iterations, "iterations"
print "Mapping:"

i = 0
with open(outfname, "w") as outfile:
	for mapp in mapping:
		outfile.write(str(i) + " -> " + str(mapp) + "\n");
		i+=1
print len(set(mapping)), "simulations required"
