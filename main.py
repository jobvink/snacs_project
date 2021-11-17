import argparse
import os
import shutil
import random

import networkx as nx
import numpy as np
import pandas as pd
import tqdm
import snap
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

def edge_density_for_snap(G: snap.PUNGraph, alpha: float) -> float:
    if G.GetNodes() <= 1:
        return 0
    else:
        n_nodes, n_edges = G.GetNodes(), G.GetEdges()
        return np.log(n_edges) - alpha * np.log(n_nodes * (n_nodes - 1) / 2)


def paper(G: nx.Graph, alpha: float) -> float:
    if len(G.nodes) <= 1:
        return 0
    else:
        n_nodes, n_edges = len(G.nodes), len(G.edges)
        return n_edges - alpha * (n_nodes * (n_nodes - 1) / 2)

def paper_for_snap(G: snap.PUNGraph, alpha: float) -> float:
    if G.GetNodes() <= 1:
        return 0
    else:
        n_nodes, n_edges = G.GetNodes(), G.GetEdges()
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
    degrees = list(G.degree)
    random.shuffle(degrees)
    sorted_nodes = sorted(degrees, key=lambda x: x[1])

    for _ in tqdm.tqdm(range(len(G.nodes))):
        # find the node with the smallest degree
        v = sorted_nodes.pop(0)[0]

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

def neighborhood(G: nx.Graph, nodes: list):
    neighbors = []
    for node in nodes:
        neighbors += list(G.neighbors(node))
    return set(neighbors)


def localSearchOQC(G: nx.Graph, objective: callable, alpha: float, t_max: int):
    v = random.choice(list(G.nodes))
    S = [v]
    for _ in tqdm.tqdm(range(t_max)):
        b_2 = True
        while b_2:
            best_objective = objective(G.subgraph(S), alpha)
            b_2 = False

            remainingNodes = neighborhood(G, S) - set(S)
            for node in remainingNodes:
                candidate = G.subgraph(S + [node])
                current_objective = objective(candidate, alpha)
                if current_objective >= best_objective:
                    S += [node]
                    b_2 = True
                    break

        best_objective = objective(G.subgraph(S), alpha)

        b_1 = False
        for node in S:
            candidate = G.subgraph(list(set(S) - set(node)))
            current_objective = objective(candidate, alpha)
            if current_objective >= best_objective:
                S.remove(node)
                b_1 = True
                break

        if not b_1:
            break

    return G.subgraph(S)

def localSearchOQCSnap(G: snap.PUNGraph, objective: callable, alpha: float, t_max: int):
    S = [G.GetRndNId()]
    best_objective = objective(G.GetSubGraph(S), alpha)
    b_1 = True
    t = 1

    for _ in tqdm.tqdm(range(t_max)):
        b_2 = True
        while b_2:
            b_2 = False
            # search the remaining nodes to improve the objective function
            for node in G.Nodes():
                if node.GetId() in S:
                    continue
                candidate = G.GetSubGraph(S + [node.GetId()])
                current_objective = objective(candidate, alpha)
                if current_objective >= best_objective:
                    S += [node.GetId()]
                    best_objective = current_objective
                    b_2 = True
                    break

        # search a node that can be removed to improve the objective function
        b_1 = False
        for node in S:
            candidate_nodes = S.copy()
            candidate_nodes.remove(node)
            candidate = G.GetSubGraph(candidate_nodes)
            current_objective = objective(candidate, alpha)
            if current_objective >= best_objective:
                S.remove(node)
                best_objective = current_objective
                b_1 = True
                break

        if not b_1:
            break

    return G.GetSubGraph(S)


# Genetic algorithm
def geneticAlgorithm(G: nx.Graph, objective: callable, alpha: float, population_size: int, t_max: int,
                     mutation_rate: float):
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
                new_individual = ranndom_mutate(G, individual)
                children.append(new_individual)

        # Evaluate and sort children
        children = [(objective(G.subgraph(individual), alpha), individual) for individual in children]
        children.sort(key=lambda x: x[0], reverse=True)

        # Replace worst
        population[-1] = children[0]

        # sort population by fitness
        population.sort(key=lambda x: x[0], reverse=True)

    # return the best child
    return G.subgraph(population[0][1])


def crossover(parent_1: list, parent_2: list):
    child = []
    for i in range(min(len(parent_1), len(parent_2))):
        if random.random() < 0.5:
            child.append(parent_1[i])
        else:
            child.append(parent_2[i])
    return list(set(child))


def ranndom_mutate(G: nx.Graph, child: list):
    """
    :param G: Graph
    :param child: list of nodes
    :return: a mutated child

    The mutation operator can remove a node or add a node.
    """
    child = child.copy()
    if random.random() <= 0.5:
        # remove a node
        if len(child) > 1:
            child.remove(random.choice(child))
    else:
        # add a node
        candidates = list(set(G.nodes) - set(child))
        if len(candidates) > 0:
            child.append(random.choice(candidates))

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


def benchmark_function(graphs, function, objective, n, alpha, name, config):
    """
    This function benchmarks a function n times
    and saves the diameter and edge_density of the resulting graph with pandas and a given name
    :return:
    """
    data = []

    for graph in graphs:
        # read graph from edge list
        G = nx.read_edgelist(graph['path'])

        print(f'Benchmarking {graph["name"]}')

        for run in tqdm.tqdm(range(n)):
            S = function(G.copy(), objective, alpha, **config)

            # if the graph is not connected
            if nx.is_connected(S):
                S_diameter = nx.algorithms.approximation.distance_measures.diameter(S)
            else:
                S_diameter = np.nan

            S_edge_density = edge_density(S, alpha)
            S_nodes = len(S.nodes)
            S_edges = len(S.edges)

            data.append({
                'graph': graph['name'],
                'run': run,
                'diameter': S_diameter,
                'edge_density': S_edge_density,
                'nodes': S_nodes,
                'edges': S_edges
            })

    # save results with pandas
    df = pd.DataFrame(data)
    df.to_csv('./results/' + name + '.csv')


def benchmark_function_with_snap(graphs, function, objective, n, alpha, name, config):
    """
    This function benchmarks a function n times
    and saves the diameter and edge_density of the resulting graph with pandas and a given name
    :return:
    """
    data = []

    for graph in graphs:
        G = snap.LoadEdgeListStr(snap.PUNGraph, graph['path'])

        print(f'Benchmarking {graph["name"]}')

        for run in tqdm.tqdm(range(n)):
            S = function(G, objective, alpha, **config)

            # if the graph is not connected
            S_diameter = S.GetBfsFullDiam(100, False)
            S_edge_density = edge_density_for_snap(S, alpha)
            S_nodes = S.GetNodes()
            S_edges = S.GetEdges()

            data.append({
                'graph': graph['name'],
                'run': run,
                'diameter': S_diameter,
                'edge_density': S_edge_density,
                'nodes': S_nodes,
                'edges': S_edges
            })

    # save results with pandas
    df = pd.DataFrame(data)
    df.to_csv('./results/' + name + '.csv')

'''
Data sources:
https://snap.stanford.edu/data/email-Eu-core.html
https://networks.skewed.de/net/football
https://networks.skewed.de/net/dolphins
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1, help='number of runs')

    args = parser.parse_args()

    n = args.runs
    alpha = 1 / 3

    os.makedirs('./results', exist_ok=True)
    random.seed(42)

    # load graphs: dolp, football, dolphins, C-elegans-frontal, amazon0302
    graphs = [
        {"name": 'dolphins', "path": './networks/dolphins.edgelist',
         'description': 'An undirected social network of frequent associations observed among 62 dolphins (Tursiops) in a community living off Doubtful Sound, New Zealand, from 1994-2001'},
        {"name": 'football', "path": './networks/football.edgelist',
         'description': 'A network of American football games between Division IA colleges during regular season Fall 2000'},
        {"name": "email-EU-core", "path": "./networks/email-Eu-core.edgelist",
         "description": "The network was generated using email data from a large European research institution."},
        {"name": "C-elegans-frontal", "path": "./networks/C-elegans-frontal.edgelist",
         "description": "..."},
        {"name": "amazon0302", "path": "./networks/amazon0302.edgelist",
         "description": "Network was collected by crawling Amazon website."}
    ]

    # collect meta data
    print('Collecting meta data...')
    meta_data = []
    for graph in tqdm.tqdm(graphs):
        G = nx.read_edgelist(graph['path'])
        meta_data.append({
            'name': graph['name'],
            'description': graph['description'],
            'nodes': len(G.nodes),
            'edges': len(G.edges),
        })

    # save meta data
    meta_data_df = pd.DataFrame(meta_data)
    meta_data_df.to_csv('./results/meta_data.csv', index=False)

    # benchmark the genetic algorithm
    print('Benchmarking geneticAlgorithm...')
    benchmark_function(graphs, geneticAlgorithm, paper, n, alpha, 'geneticAlgorithm',
                       {'population_size': 10, 't_max': 1000, 'mutation_rate': 0.1})

    # benchmark the greedyOQC algorithm
    print('Benchmarking greedyOQC...')
    benchmark_function(graphs, greedyOQC, paper, n, alpha, 'greedyOQC', {})

    # benchmark the localSearchOQCSnap algorithm
    print('Benchmarking localSearchOQCSnap...')
    benchmark_function_with_snap(graphs, localSearchOQCSnap, paper_for_snap, n, alpha, 'localSearchOQCSnap', {'t_max': 50})




