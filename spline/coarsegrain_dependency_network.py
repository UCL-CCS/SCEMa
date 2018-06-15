import os, sys
import networkx as nx
import operator
import matplotlib.pyplot as plt
import glob

def get_max_degree_node(x):
	sorted_x = sorted(x.items(), key=operator.itemgetter(1))
	return sorted_x[-1][0]


if len(sys.argv) != 3:
	sys.exit("Usage: coarsegrain_dependency_network.py [input_folder] [out_mapping.csv]")

input_folder = sys.argv[1]
out_mapping_fname = sys.argv[2]

G = nx.Graph()

print("Reading...")
max_cell_ID = 0;
for fname in glob.glob(input_folder + "/last.*.similar_hist"):
	with open(fname, "r") as infile:
		for line in infile.readlines():
			cell1, cell2, dist = line.split()
			cell1 = int(cell1)
			cell2 = int(cell2)
			dist = float(dist)

			if cell1 > max_cell_ID:
				max_cell_ID = cell1

			G.add_edge(cell1, cell2, weight=1.0/dist)

num_nodes_remaining = len(G)
mapping = ["Not to be updated"] * max_cell_ID
iterations = 0

print("Coarsegraining...")
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

print("Converged in", iterations, "iterations")
print("Mapping:")

i = 0
with open(out_mapping_fname, "w") as outfile_map:
	for mapp in mapping:
		outfile_map.write(str(i) + "," + str(mapp) + "\n");
		i+=1
print(len(set(mapping)), "simulations required")
