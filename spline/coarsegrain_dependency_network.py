import os, sys
import networkx as nx
import operator
import glob

def get_max_degree_node(x):
    """
    Returns the node with the highest degree (i.e. the most highly connected node)

    Parameters
    ----------
    x : dict
        A dictionary of all node IDs and their corresponding degree of connectivity.

    Returns
    -------
        The ID of the node with the highest degree of connectivity.
    """

    sorted_x = sorted(x.items(), key=operator.itemgetter(1))
    return sorted_x[-1][0]


def coarsegrain_dependency_network(input_folder, out_mapping_fname, num_gps):
    """
    Constructs a dependency graph from the similarity files in input_folder, and calculates
    the mapping that corresponds to the least MD simulations needing to be run.

    Parameters
    ----------
    input_folder : str
        Path of the directory containing similarity input files.
        Files are assumed to be named in the pattern "last.*.similar_hist".
    out_mapping_fname : str
        File name of output mapping file.
    num_gps:
        Total number of gauss points in the finite element mesh.

    Returns
    -------
    """

    # Keep track of how many gauss points will need to be updated
    num_gp_tbu = 0

    # Build a networkx graph based on the similarities
    G = nx.Graph()
    for fname in glob.glob(input_folder + "/last.*.similar_hist"):
        with open(fname, "r") as infile:
            num_gp_tbu+=1
            for line in infile.readlines():
                cell1, cell2, dist = line.split()
                cell1 = int(cell1)
                cell2 = int(cell2)
                dist = float(dist)

                G.add_edge(cell1, cell2, weight=1.0/dist)

    num_nodes_remaining = len(G)
    mapping = [i for i in range(num_gp)]
    iterations = 0
    neighbour_removed = 0

    # Recursively remove the highest degree node (and all neighbours) from the network until all nodes
    # are gone. When a node is removed, it and all its neighbours are mapped to obtain their results
    # from that node's MD simulation in the next MD step.
    while num_nodes_remaining > 0:

        # get max degree node
        max_deg_node = get_max_degree_node(dict(G.degree()))

        # Map this node to use its own MD results
        mapping[max_deg_node] = max_deg_node

        # Map all neighbours to the ID of the max. degree node, then remove them all from the graph.
        neighbours = [max_deg_node]
        for neighbour in nx.all_neighbors(G, max_deg_node):
            neighbours.append(neighbour)
            mapping[neighbour] = max_deg_node
            neighbour_removed += 1

        G.remove_nodes_from(neighbours)

        num_nodes_remaining = len(G)
        iterations += 1

    # Output mapping to file
    with open(out_mapping_fname, "w") as outfile_map:
        for i, mapp in enumerate(mapping):
            outfile_map.write(str(i) + " " + str(mapp) + "\n");

    print("Converged in", iterations, "iterations")
    print("Number of gauss points to be udpated: ", num_gp_tbu)
    print("Number of simulations required: ", num_gp_tbu-neighbour_removed)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: coarsegrain_dependency_network.py [input_folder] [out_mapping.csv] [number_of_gps]")

    input_folder = sys.argv[1]
    out_mapping_fname = sys.argv[2]
    num_gps = int(sys.argv[3])


