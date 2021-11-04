import os
import shutil
import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
import imageio



def greedyOQC(G: nx.Graph, objective: callable, alpha: float):
    best = G.copy()
    best_objective = objective(best, alpha)

    smallest = []

    for _ in range(len(G.nodes)):
        # find the node with the smallest degree
        degrees = dict(G.degree)
        v = min(degrees, key=degrees.get)
        smallest.append(v)

        # remove the node with the smallest degree
        G.remove_node(v)

        # evaluate the new Graph and
        current_objective = objective(G, alpha)

        # save the Graph if the objective function improves
        if current_objective > best_objective:
            best = G.copy()
            best_objective = current_objective

    return best, smallest


def localSearchOQC(G: nx.Graph, t_max: int, objective: callable, alpha: float):
    v = random.choice(list(G.nodes))
    S: nx.Graph = G.subgraph([v])
    b_1 = True
    t = 1
    history = [list(S.nodes)]
    while b_1 and t <= t_max:
        b_2 = True
        remainingGraph = nx.subgraph_view(G, lambda n: n not in S.nodes)
        while b_2:
            best_objective = objective(S, alpha)
            found = False

            nodes_in_S = list(S.nodes)
            for node in remainingGraph.nodes:
                candidate = nx.subgraph_view(G, lambda n: n in nodes_in_S + [node])
                current_objective = objective(candidate, alpha)
                if current_objective >= best_objective:
                    S = G.subgraph(list(S.nodes) + [node])
                    found = True
                    history.append(list(S.nodes))
                    break

            if not found:
                b_2 = False

        best_objective = objective(S, alpha)
        found = False

        for node in S.nodes:
            nodes_in_S = list(S.nodes)
            nodes_in_S.remove(node)
            candidate = nx.subgraph_view(G, lambda n: n not in nodes_in_S)
            current_objective = objective(candidate, alpha)
            if current_objective >= best_objective:
                new_view = list(S.nodes)
                new_view.remove(node)
                S = G.subgraph(new_view)
                found = True
                history.append(list(S.nodes))
                break

        if not found:
            b_1 = False

        t += 1

    # Todo: hier moet nog iets bij volgens mij
    return S, history


def densest_subgraph(G: nx.Graph, alpha: float) -> float:
    if len(G.nodes) <= 1:
        return 0
    else:
        return np.log(len(G.edges)) - alpha * np.log(len(G.nodes))


def edge_density(G: nx.Graph, alpha: float) -> float:
    if len(G.nodes) <= 1:
        return 0
    else:
        n_nodes, n_edges = len(G.nodes), len(G.edges)
        return np.log(n_edges) - alpha * np.log(n_nodes * (n_nodes - 1) / 2)


def plot_greedy_oqc(G: nx.Graph, objective: callable, alpha: float):
    shutil.rmtree('images')
    os.makedirs('images')
    B, history = greedyOQC(G.copy(), objective, alpha)
    layout = nx.spring_layout(G)
    visited = []
    plots = []
    for i,v in enumerate(history):
        coloring = []
        for node in G.nodes:
            if node == v:
                coloring.append('blue')
            elif node in visited:
                coloring.append('gray')
            else:
                coloring.append('green')
        visited.append(v)
        nx.draw(G, pos=layout, node_color=coloring)
        image = 'images/' + str(i) + '.png'
        plt.savefig(image)
        plots.append(image)

    coloring = []
    for node in G.nodes:
        if node in B.nodes:
            coloring.append('red')
        else:
            coloring.append('gray')

    nx.draw(G, pos=layout, node_color=coloring)
    image = 'images/full.png'
    plt.savefig(image)
    plots.append(image)
    plots.append(image)
    plots.append(image)
    return plots

def plot_local_search(G: nx.Graph, t_max: int, objective: callable, alpha: float):
    shutil.rmtree('images')
    os.makedirs('images')
    B, history = localSearchOQC(G.copy(), t_max, objective, alpha)
    layout = nx.spring_layout(G)

    plots = []
    for i, current_subgraph in enumerate(history):
        coloring = []
        for node in G.nodes:
            if node in current_subgraph:
                coloring.append('blue')
            else:
                coloring.append('gray')
        nx.draw(G, pos=layout, node_color=coloring)
        image = 'images/' + str(i) + '.png'
        plt.savefig(image)
        plots.append(image)

    coloring = []
    for node in G.nodes:
        if node in B.nodes:
            coloring.append('red')
        else:
            coloring.append('gray')

    nx.draw(G, pos=layout, node_color=coloring)
    image = 'images/full.png'
    plt.savefig(image)
    plots.append(image)
    plots.append(image)
    plots.append(image)
    return plots


if __name__ == '__main__':
    G = nx.erdos_renyi_graph(10, .6)
    # B, history = localSearchOQC(G.copy(), 100, edge_density, 2/3)
    # plots = plot_local_search(G, 10, edge_density, 1/3)
    # with imageio.get_writer('test.gif', mode='I', duration=.5) as writer:
    #     for filename in plots:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

    plots = plot_greedy_oqc(G, edge_density, 1)
    with imageio.get_writer('test.gif', mode='I', duration=.5) as writer:
        for filename in plots:
            image = imageio.imread(filename)
            writer.append_data(image)


