import argparse
import os
import shutil
import random

import networkx as nx
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

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


def paper(G: nx.Graph, alpha: float) -> float:
    if len(G.nodes) <= 1:
        return 0
    else:
        n_nodes, n_edges = len(G.nodes), len(G.edges)
        return n_edges - alpha * (n_nodes * (n_nodes - 1) / 2)


def greedyOQC(G: nx.Graph, objective: callable, alpha: float):
    best = G.copy()
    best_objective = objective(best, alpha)

    for _ in tqdm.tqdm(range(len(G.nodes))):
        # find the node with the smallest degree
        degrees = dict(G.degree)
        v = min(degrees, key=degrees.get)

        # remove the node with the smallest degree
        G.remove_node(v)

        # evaluate the new Graph and
        current_objective = objective(G, alpha)

        # save the Graph if the objective function improves
        if current_objective > best_objective:
            best = G.copy()
            best_objective = current_objective

    return best


def localSearchOQC(G: nx.Graph, t_max: int, objective: callable, alpha: float):
    v = random.choice(list(G.nodes))
    S: nx.Graph = G.subgraph([v])
    b_1 = True
    t = 1
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
                    break

            if not found:
                b_2 = False

        best_objective = objective(S, alpha)
        found = False

        for node in S.nodes:
            nodes_in_S = list(S.nodes)
            nodes_in_S.remove(node)
            candidate = nx.subgraph_view(G, lambda n: n in nodes_in_S)
            current_objective = objective(candidate, alpha)
            if current_objective >= best_objective:
                new_view = list(S.nodes)
                new_view.remove(node)
                S = G.subgraph(new_view)
                found = True
                break

        if not found:
            b_1 = False

        t += 1

    # Todo: hier moet nog iets bij volgens mij
    return S


'''
Data sources:
https://networks.skewed.de/net/football
https://networks.skewed.de/net/dolphins
https://networks.skewed.de/net/as_skitter
http://snap.stanford.edu/data/wiki-topcats.html
http://snap.stanford.edu/data/web-Google.html
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1, help='number of runs')

    args = parser.parse_args()

    n = args.runs
    alpha = 1 / 3
    t_max = 50

    os.makedirs('./results', exist_ok=True)
    random.seed(42)

    # graphs: as_skitter, dolphins, football, web-Google and wiki-topcats
    graphs = [
        {"name": 'as-skitter', "path": './networks/as-skitter.gml',
         'description': 'An aggregate snapshot of the Internet Protocol (IP) graph, as measured by the traceroute tool on CAIDA\'s skitter infrastructure, in 2005'},
        {"name": 'dolphins', "path": './networks/dolphins.gml',
         'description': 'An undirected social network of frequent associations observed among 62 dolphins (Tursiops) in a community living off Doubtful Sound, New Zealand, from 1994-2001'},
        {"name": 'football', "path": './networks/football.gml',
         'description': 'A network of American football games between Division IA colleges during regular season Fall 2000'},
        {"name": 'web-Google', "path": './networks/web-Google.gml',
         'description': 'Webgraph from the Google programming contest, 2002'},
        {"name": 'wiki-topcats', "path": './networks/wiki-topcats.gml',
         'description': 'Hyperlink network of the top catagories of Wikipedia in 2011'}
    ]

    # collect meta data
    meta_data = []
    for graph in tqdm.tqdm(graphs):
        G = nx.read_gml(graph['path'])
        meta_data.append({
            'name': graph['name'],
            'descriptoin': graph['description'],
            'nodes': len(G.nodes),
            'edges': len(G.edges),
        })

    # save meta data
    meta_data_df = pd.DataFrame(meta_data)
    meta_data_df.to_csv('./results/meta_data.csv', index=False)

    data = []

    # collect diameter and edge density for all graphs
    for graph in graphs:
        # run experiment n times
        # load graph
        G = nx.read_gml(graph['path'])
        for i in range(n):
            # find the densest subgraph
            B = greedyOQC(G, paper, alpha)
            B_diameter = nx.diameter(B)
            B_edge_density = edge_density(B, alpha)

            data.append([i, graph['name'], B_diameter, B_edge_density])

    # save data
    df = pd.DataFrame(data, columns=['run', 'graph', 'diameter', 'edge_density'])
    df.to_csv('./results/greedyOQC.csv', index=False)

    data = []

    # collect diameter and edge density for all graphs
    for graph in tqdm.tqdm(graphs):
        # run experiment n times
        # load graph
        G = nx.read_gml(graph['path'])
        for i in range(n):
            # find the densest subgraph
            B = localSearchOQC(G, t_max, paper, alpha)
            B_diameter = nx.diameter(B)
            B_edge_density = edge_density(B, alpha)

            data.append([i, graph['name'], B_diameter, B_edge_density])

    # save data
    df = pd.DataFrame(data, columns=['run', 'graph', 'diameter', 'edge_density'])
    df.to_csv('./results/localSearchOQC.csv', index=False)
