from classes import Cluster, State
from matplotlib import gridspec
from globals import BLUE, BLACK, GEOSPATIAL_BOUND_NEW
from itertools import cycle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_state(state: State):
    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))
    fig.tight_layout(pad=1.0)

    # add subplot to figure
    spec = gridspec.GridSpec(ncols=1, nrows=1)
    ax1 = fig.add_subplot(spec[0])
    ax1.axis("off")

    oslo = plt.imread("test_data/kart_oslo.png")
    ax1.imshow(
        oslo, zorder=0, extent=(0, 1, 0, 1), aspect="auto", alpha=0.6,
    )

    # constructs the networkx graph
    graph, labels, node_border, node_color = make_graph(state.clusters)
    pos = nx.get_node_attributes(graph, "pos")

    # add number of scooters and battery label to nodes
    for i, cluster in enumerate(state.clusters):
        node_info = f"S = {cluster.number_of_scooters()} \nB = {round(cluster.get_current_state(),1)}"
        x, y = pos[i]
        ax1.annotate(
            node_info, xy=(x, y + 0.03), horizontalalignment="center", fontsize=12
        )

    # set edge color for solution
    edges = graph.edges()
    e_colors = [graph[u][v]["color"] for u, v in edges]
    e_weights = [graph[u][v]["width"] for u, v in edges]

    # draw solution graph
    nx.draw(
        graph,
        pos,
        node_color=node_color,
        edgecolors=node_border,
        edge_color=e_colors,
        width=e_weights,
        node_size=1000,
        alpha=0.7,
        with_labels=False,
        ax=ax1,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        labels,
        font_size=16,
        font_color="white",
        font_weight="bold",
        ax=ax1,
    )

    plt.tight_layout()
    plt.show()


def make_graph(clusters: [Cluster]):
    cartesian_clusters = convert_geographic_to_cart(clusters, GEOSPATIAL_BOUND_NEW)

    colors = cycle("bgrcmyk")

    # make graph object
    graph = nx.DiGraph()
    graph.add_nodes_from([c for c in np.arange(len(cartesian_clusters))])

    # set node label and position in graph
    labels = {}
    node_color = []
    node_border = []
    for i, cartesian_cluster_coordinates in enumerate(cartesian_clusters):
        cluster_color = next(colors)
        label = i + 1
        labels[i] = label
        node_color.append(cluster_color)
        node_border.append(BLACK)
        graph.nodes[i]["pos"] = cartesian_cluster_coordinates

    return graph, labels, node_border, node_color


def convert_geographic_to_cart(clusters: [Cluster], bound: [float]) -> [(float, float)]:
    lat_min, lat_max, lon_min, lon_max = bound
    delta_lat = lat_max - lat_min
    delta_lon = lon_max - lon_min
    zero_lat = lat_min / delta_lat
    zero_lon = lon_min / delta_lon

    output = []

    for i, cluster in enumerate(clusters):
        lat, lon = cluster.center

        y = lat / delta_lat - zero_lat
        x = lon / delta_lon - zero_lon

        output.append((x, y))

    return output
