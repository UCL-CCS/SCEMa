import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

labels = {}
with open("network.txt") as infile:
	for line in infile.readlines():
		cell1, cell2, dist = line.split()
		cell1 = int(cell1)
		cell2 = int(cell2)
		dist = float(dist)
		print cell1, cell2, dist

		G.add_edge(cell1, cell2, weight=dist)
		labels[cell1] = str(cell1)
		labels[cell2] = str(cell2)

pos=nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, alpha=0.5)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels, font_size=10)
plt.show()
