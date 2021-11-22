import random
import networkx as nx
import pandas as pd
import tqdm
import numpy as np
import argparse

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

def edge_density(G: nx.Graph, alpha: float) -> float:
    if len(G.nodes) <= 1:
        return 0
    else:
        n_nodes, n_edges = len(G.nodes), len(G.edges)
        return np.log(n_edges) - alpha * np.log(n_nodes * (n_nodes - 1) / 2)


# Genetic algorithm
def geneticAlgorithm(G: nx.Graph, objective: callable, alpha: float, population_size: int, t_max: int,
                     mutation_rate: float, mutation_function: callable):
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
                new_individual = mutation_function(G, individual)
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

if __name__ == '__main__':
    # load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10, help='number of runs')

    args = parser.parse_args()

    n = args.runs


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
    
    alpha = 1/3

    # benchmarking Gentic Algorithm with different population sizes 10, 50, 100, 200, 500, 1000
    # config = {
    #     't_max': 1000,
    #     'mutation_rate': 0.1,
    #     'mutation_function': ranndom_mutate,
    # }
    #
    # for population_size in [10, 50, 100, 200, 500, 1000]:
    #     config['population_size'] = population_size
    #     benchmark_function(graphs, geneticAlgorithm, paper, n, alpha, f'ga_{population_size}', config)

    # benchmarking different mutation functions
    config = {
        'population_size': 100,
        't_max': 1000,
        'mutation_rate': 0.1,
    }

    for mutation_function in [ranndom_mutate, neighborhood_mutate]:
        config['mutation_function'] = mutation_function
        benchmark_function(graphs, geneticAlgorithm, paper, n, alpha, f'ga_{mutation_function.__name__}', config)

    # benchmarking different objective functions
    config = {
        'population_size': 100,
        't_max': 1000,
        'mutation_rate': 0.1,
        'mutation_function': ranndom_mutate,
    }
    for objective in [paper, ours]:
        benchmark_function(graphs, geneticAlgorithm, objective, n, alpha, f'ga_{objective.__name__}', config)

    
