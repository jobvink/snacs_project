import os
import random

import math
import networkx as nx
import pandas as pd
import tqdm
import numpy as np
import argparse


def edge_surplus(G: nx.Graph, alpha: float) -> float:
    if len(G.nodes) <= 1:
        return 0
    else:
        n_nodes, n_edges = len(G.nodes), len(G.edges)
        return n_edges - alpha * (n_nodes * (n_nodes - 1) / 2)


def modified_edge_surplus(G: nx.Graph, alpha: float) -> float:
    if len(G.nodes) <= 1:
        return 0
    else:
        n_nodes, n_edges = len(G.nodes), len(G.edges)
        max_edges = (n_nodes * (n_nodes - 1) / 2)
        return n_edges - alpha * max_edges - (not nx.is_connected(G)) * max_edges


def edge_density(G: nx.Graph, alpha: float) -> float:
    if len(G.nodes) <= 1:
        return 0
    else:
        n_nodes, n_edges = len(G.nodes), len(G.edges)
        return n_edges / math.comb(n_nodes, 2)


def triangle_density(G: nx.Graph):
    """
    The triangle density is defined as the number of triangles divided by the maximum possible number of triangles.
    :param G: Graph
    :return: the triangle density of the graph
    """
    number_of_triangles = sum(nx.triangles(G).values()) / 3
    return number_of_triangles / math.comb(len(G.nodes), 3)


# Genetic algorithm
def geneticAlgorithm(G: nx.Graph, objective: callable, alpha: float, population_size: int, t_max: int, mutation_function: callable):
    # Initialize population
    individual_size = np.floor(np.log(len(G.nodes)) * np.log(len(G.edges)))
    population = [random.sample(list(G.nodes), int(individual_size)) for _ in range(population_size)]

    # Sort population by fitness
    population = [(objective(G.subgraph(individual), alpha), individual) for individual in population]
    population.sort(key=lambda x: x[0], reverse=True)

    # Iterate until convergence
    for _ in range(t_max):
        # Select parents
        parents = random.sample(population, 2)

        # Crossover
        children = [crossover(G, parents[0][1], parents[1][1]) for _ in range(2)]

        # Mutate an individual with mutation rate
        for _, individual in population:
            new_individual = mutation_function(G, individual)
            children.append(new_individual)

        # Evaluate and sort children
        children = [(objective(G.subgraph(individual), alpha), individual) for individual in children]
        children.sort(key=lambda x: x[0], reverse=True)

        # append the children to the population
        population.extend(children)

        # sort population by fitness
        population.sort(key=lambda x: x[0], reverse=True)

        # remove the worst individuals
        population = population[:population_size]

    # return the best child
    return G.subgraph(population[0][1])

def crossover(G: nx.Graph, parent_1: list, parent_2: list):
    """
    parten_1 = [1 ,2, 3]
    :param parent_1:
    :param parent_2:
    :return:
    """
    nodes = list(G.nodes)
    # split the nodes in two parts
    split_point = random.randint(0, len(nodes) - 1)
    nodes_left = nodes[:split_point]
    nodes_right = nodes[split_point:]

    # intersection of parent 1 and nodes left
    child_1 = list(set(parent_1) & set(nodes_left))
    # intersection of parent 2 and nodes right
    child_2 = list(set(parent_2) & set(nodes_right))

    # concatenate the two lists and return the result
    return child_1 + child_2


def random_mutate(G: nx.Graph, child: list):
    """
    :param G: Graph
    :param child: list of nodes
    :return: a mutated child

    The mutation operator can remove a node or add a node.
    """
    child = child.copy()

    # choose a random node
    node = random.choice(list(G.nodes))

    # if the node is in the child, remove it
    if node in child and len(child) > 1:
        child.remove(node)

    # if the node is not in the child, add it
    else:
        child.append(node)

    return child


def neighborhood_mutate(G: nx.Graph, child: list):
    """
    :param G: Graph
    :param child: list of nodes
    :return: a mutated child

    The mutation operator can remove a node or add a node from the neighborhood of the child.
    """
    child = child.copy()

    if random.random() < 0.5 and len(child) > 0:
        child.remove(random.choice(child))
    else:
        # neighbours
        neighbours = list(nx.neighbors(G, random.choice(child)))

        # difference between child and neighbours
        neighbours = list(set(neighbours) - set(child))

        # if there are no neighbours, skip
        if len(neighbours) > 0:
            child.append(random.choice(neighbours))

    return child


def benchmark_function(run, graphs, function, objective, alpha, name, config):
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

        S = function(G.copy(), objective, alpha, **config)

        # if the graph is not connected
        if nx.is_connected(S):
            S_diameter = nx.algorithms.approximation.distance_measures.diameter(S)
        else:
            S_diameter = np.nan

        # triangle density
        S_triangle_density = triangle_density(S)

        S_edge_density = edge_density(S, alpha)
        S_nodes = len(S.nodes)
        S_edges = len(S.edges)

        data.append({
            'run': run,
            'graph': graph['name'],
            'diameter': S_diameter,
            'edge_density': S_edge_density,
            'triangle_density': S_triangle_density,
            'nodes': S_nodes,
            'edges': S_edges
        })

    # create directory if it does not exist
    os.makedirs(f'benchmark/{name}/{run}', exist_ok=True)

    # save results with pandas
    df = pd.DataFrame(data)
    df.to_csv(f'benchmark/{name}/{run}/benchmark.csv', index=False)


if __name__ == '__main__':
    # load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=0, help='run number')

    args = parser.parse_args()

    n = args.run

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

    alpha = 1 / 3

    # benchmarking different mutation functions
    config = {
        'population_size': 250,
        't_max': 1000,
        "mutation_function": random_mutate,
    }

    # benchmarking different objective functions
    for objective in [edge_surplus, modified_edge_surplus]:
        benchmark_function(n, [graphs[2]], geneticAlgorithm, objective, alpha, objective.__name__, config)