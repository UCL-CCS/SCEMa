import os
import networkx as nx
import operator
import matplotlib.pyplot as plt

def get_max_degree_node(x):
	sorted_x = sorted(x.items(), key=operator.itemgetter(1))
	return sorted_x[-1][0]


os.system("cat __results/ID_* > network.txt\n")

G = nx.Graph()

labels = {}
with open("network.txt") as infile:
	for line in infile.readlines():
		cell1, cell2, dist = line.split()
		cell1 = int(cell1)
		cell2 = int(cell2)
		dist = float(dist)
#		print cell1, cell2, dist

#		G.add_edge(cell1, cell2)
		G.add_edge(cell1, cell2, weight=1.0/dist)
		labels[cell1] = str(cell1)
		labels[cell2] = str(cell2)

#print(G.degree())
#rofl = sorted(set(G.degree().values()))
#rofl = sorted(G.degree, key=lambda x: x[1], reverse=True)

num_nodes_remaining = len(G)

while num_nodes_remaining > 0:
	node = get_max_degree_node(G.degree())
	print "Max degree node = ", node

	neighbours = [node]
	for neighbour in nx.all_neighbors(G, node):
		neighbours.append(neighbour)
	print "Neighbours:", neighbours

	print "Removing", len(neighbours)
	G.remove_nodes_from(neighbours)

	num_nodes_remaining = len(G)
	print "Num nodes remaining", len(G)

pos=nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, alpha=0.5, node_size=1)
nx.draw_networkx_edges(G, pos)
#nx.draw_networkx_labels(G, pos, labels, font_size=10)
plt.show()
