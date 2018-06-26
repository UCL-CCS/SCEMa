import os, sys
import networkx as nx
import operator
import glob

print("           ...entering coarsegrain_dependency_network.py...", end='')

def get_max_degree_node(x):
	sorted_x = sorted(x.items(), key=operator.itemgetter(1))
	return sorted_x[-1][0]


if len(sys.argv) != 4:
	sys.exit("Usage: coarsegrain_dependency_network.py [input_folder] [out_mapping.csv] [number_of_qps]")

input_folder = sys.argv[1]
out_mapping_fname = sys.argv[2]
num_qps = int(sys.argv[3])

num_qps_tbu = 0

G = nx.Graph()

print(" reading...", end='')
for fname in glob.glob(input_folder + "/last.*.similar_hist"):
	with open(fname, "r") as infile:
		num_qps_tbu+=1
		for line in infile.readlines():
			qcid1, qcid1, dist = line.split()
			dist = float(dist)

			G.add_edge(qcid1, qcid2, weight=1.0/dist)

num_nodes_remaining = len(G)
mapping = [i for i in range(num_qps)]
iterations = 0
neighbour_removed = 0

print(" coarsegraining...", end='')

while num_nodes_remaining > 0:

	# get max degree node
	max_deg_node = get_max_degree_node(dict(G.degree()))
#	print "Max degree node = ", max_deg_node

	# Map this node to use its own MD results
	mapping[max_deg_node] = max_deg_node

	neighbours = [max_deg_node]
	for neighbour in nx.all_neighbors(G, max_deg_node):
		neighbours.append(neighbour)
		mapping[neighbour] = max_deg_node
		neighbour_removed += 1

#	print "Neighbours:", neighbours

#	print "Removing", len(neighbours)
	G.remove_nodes_from(neighbours)

	num_nodes_remaining = len(G)
#	print "Num nodes remaining", len(G)

	iterations += 1

print(" converged in", iterations, "iterations")

i = 0
with open(out_mapping_fname, "w") as outfile_map:
	for mapp in mapping:
		outfile_map.write(str(i) + " " + str(mapp) + "\n");
		i+=1
print("           ...number of quadrature points to be udpated:",num_qps_tbu, " - number of simulations required: ",num_qps_tbu-neighbour_removed)

sys.exit(0)
