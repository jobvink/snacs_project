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

def ours(G: nx.Graph, alpha: float) -> float:
    if len(G.nodes) <= 1:
        return 0
    else:
        n_nodes, n_edges = len(G.nodes), len(G.edges)
        max_edges = (n_nodes * (n_nodes - 1) / 2)
        return n_edges - alpha * max_edges + nx.is_connected(G) * max_edges


def greedyOQC(G: nx.Graph, objective: callable, alpha: float):
    original = G.copy()
    best = original.nodes
    best_objective = objective(original, alpha)

    # sort the degree of the graph and put it into a stack
    sorted_nodes = sorted(G.degree, key=lambda x: x[1])

    for _ in tqdm.tqdm(range(len(G.nodes))):
        # find the node with the smallest degree
        v = sorted_nodes.pop()[0]

        # remove the node with the smallest degree
        G.remove_node(v)

        # evaluate the new Graph and
        current_objective = objective(G, alpha)

        # save the Graph if the objective function improves
        if current_objective > best_objective:
            best = list(G.nodes)
            best_objective = current_objective

    B = original.subgraph(best)

    return B


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



# Genetic algorithm
def geneticAlgorithm(G: nx.Graph, objective: callable, alpha: float, population_size: int, t_max: int, mutation_rate: float):
    # Initialize population
    population = []
    for _ in range(population_size):
        population.append(random.sample(list(G.nodes), len(G.nodes)))

    # Sort population by fitness
    population = [(objective(G.subgraph(individual), alpha), individual) for individual in population]
    population.sort(key=lambda x: x[0], reverse=True)

    # Iterate until convergence
    for _ in range(t_max):
        # Select parents
        parents = random.sample(population, 2)

        # Crossover
        children = [crossover(parents[0][1], parents[1][1]) for _ in range(2)]

        # Mutate an individual with mutation rate
        for _, individual in population:
            if random.random() <= mutation_rate:
                individual = ranndom_mutate(G, individual)
                children.append(individual)

        # Evaluate and sort children
        children = [(objective(G.subgraph(individual), alpha), individual) for individual in children]
        children.sort(key=lambda x: x[0], reverse=True)

        # Replace worst
        population[-1] = children[0]

        # sort population by fitness
        population.sort(key=lambda x: x[0], reverse=True)

        print(objective(G.subgraph(population[0][1]), alpha))

    # return the best child
    return G.subgraph(population[0][1])

def crossover(parent_1: list, parent_2: list):
    children = []
    for i in range(min(len(parent_1), len(parent_2))):
        if random.random() <= 0.5:
            children.append(parent_1[i])
        else:
            children.append(parent_2[i])
    return children


def ranndom_mutate(G: nx.Graph, child: list):
    """
    :param G: Graph
    :param child: list of nodes
    :return: a mutated child

    The mutation operator can remove a node or add a node.
    """
    if random.random() <= 0.5:
        # remove a node
        child.remove(random.choice(child))
    else:
        # add a node
        child.append(random.choice(list(set(G.nodes) - set(child))))

    return child

def neighborhood_mutate(G: nx.Graph, child: list):
    """
    :param G: Graph
    :param child: list of nodes
    :return: a mutated child

    The mutation operator can remove a node or add a node from the neighborhood of the child.
    """
    if random.random() <= 0.5:
        # remove a node
        child.remove(random.choice(child))
    else:
        # add a node
        child.append(random.choice(list(G.neighbors(random.choice(child)))))

    return child


'''
Data sources:
https://snap.stanford.edu/data/email-Eu-core.html
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
    t_max = 1000

    os.makedirs('./results', exist_ok=True)
    random.seed(42)

    # graphs: as_skitter, dolphins, football, web-Google and wiki-topcats
    graphs = [
        {"name": 'football', "path": './networks/football.gml',
         'description': 'A network of American football games between Division IA colleges during regular season Fall 2000'},
        {"name": "email-EU-core", "path": "./networks/email-Eu-core.gml",
         "description": "The network was generated using email data from a large European research institution."},
        {"name": 'dolphins', "path": './networks/dolphins.gml',
         'description': 'An undirected social network of frequent associations observed among 62 dolphins (Tursiops) in a community living off Doubtful Sound, New Zealand, from 1994-2001'},
        # {"name": 'as-skitter', "path": './networks/as-skitter.gml',
        #  'description': 'An aggregate snapshot of the Internet Protocol (IP) graph, as measured by the traceroute tool on CAIDA\'s skitter infrastructure, in 2005'},
        # {"name": 'web-Google', "path": './networks/web-Google.gml',
        #  'description': 'Webgraph from the Google programming contest, 2002'},
        # {"name": 'wiki-topcats', "path": './networks/wiki-topcats.gml',
        #  'description': 'Hyperlink network of the top catagories of Wikipedia in 2011'}
    ]

    # evaluate the genetic algorithm on each graph
    data = []

    # collect diameter and edge density for all graphs
    for graph in graphs:
        G = nx.read_gml(graph['path'])
        print(f'{graph["name"]}: {graph["description"]}')
        for _ in range(n):
            print(f'Run {_ + 1}')
            B = geneticAlgorithm(G, paper, alpha, population_size=10, t_max=t_max, mutation_rate=0.1)

            # if the graph is not connected retrun 0
            if nx.is_connected(B):
                B_diameter = nx.algorithms.approximation.distance_measures.diameter(B)
            else:
                B_diameter = 0

            B_edge_density = edge_density(B, alpha)

            data.append({'graph': graph['name'], 'diameter': B_diameter, 'edge_density': B_edge_density})

    # save results with pandas
    df = pd.DataFrame(data)
    df.to_csv('./results/genetic_algorithm.csv', index=False)


    # collect meta data
    # meta_data = []
    # for graph in tqdm.tqdm(graphs):
    #     G = nx.read_gml(graph['path'])
    #     meta_data.append({
    #         'name': graph['name'],
    #         'description': graph['description'],
    #         'nodes': len(G.nodes),
    #         'edges': len(G.edges),
    #     })
    #
    # # save meta data
    # meta_data_df = pd.DataFrame(meta_data)
    # meta_data_df.to_csv('./results/meta_data.csv', index=False)
    #
    # data = []
    #
    # # collect diameter and edge density for all graphs
    # for graph in graphs:
    #     # run experiment n times
    #     # load graph
    #     G = nx.read_gml(graph['path'])
    #     for i in range(n):
    #         # find the densest subgraph
    #         B = greedyOQC(G, paper, alpha)
    #
    #         # if the graph is not connected retrun 0
    #         if nx.is_connected(B):
    #             B_diameter = nx.algorithms.approximation.distance_measures.diameter(B)
    #         else:
    #             B_diameter = 0
    #
    #         B_edge_density = edge_density(B, alpha)
    #
    #         data.append([i, graph['name'], B_diameter, B_edge_density])
    #
    # # save data
    # df = pd.DataFrame(data, columns=['run', 'graph', 'diameter', 'edge_density'])
    # df.to_csv('./results/greedyOQC.csv', index=False)
    #
    # data = []

    # collect diameter and edge density for all graphs
    # for graph in tqdm.tqdm(graphs):
    #     # run experiment n times
    #     # load graph
    #     G = nx.read_gml(graph['path'])
    #     for i in range(n):
    #         # find the densest subgraph
    #         B = localSearchOQC(G, t_max, paper, alpha)
    #         B_diameter = nx.diameter(B)
    #         B_edge_density = edge_density(B, alpha)
    #
    #         data.append([i, graph['name'], B_diameter, B_edge_density])
    #
    # # save data
    # df = pd.DataFrame(data, columns=['run', 'graph', 'diameter', 'edge_density'])
    # df.to_csv('./results/localSearchOQC.csv', index=False)