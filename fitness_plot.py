import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    edge_surplus = pd.read_csv('./results/edge_surplus/benchmark.csv', index_col=0)
    modified_edge_surplus = pd.read_csv('./results/modified_edge_surplus/benchmark.csv', index_col=0)

    fitness_functions = ['neighborhood', 'random']

    edge_surplus = edge_surplus[edge_surplus['graph'] == 'football']
    modified_edge_surplus = modified_edge_surplus[modified_edge_surplus['graph'] == 'football']

    # Make a scatter plot of the edge density vs. the number of nodes in the graph
    plt.scatter(edge_surplus['nodes'], edge_surplus['edge_density'], marker='v', label='Edge surplus')
    plt.scatter(modified_edge_surplus['nodes'], modified_edge_surplus['edge_density'], marker='o', label='Modified Edge surplus')
    plt.xlabel('Number of nodes')
    plt.ylabel('Edge density')
    plt.title('football')
    plt.legend()
    # save the figure as pdf
    plt.savefig('./images/fitness_football_edge_density.pdf')

    plt.close()

    # Make a scatter plot of the diameter vs. the number of nodes in the graph
    plt.scatter(edge_surplus['nodes'], edge_surplus['diameter'], marker='v', label='Edge surplus')
    plt.scatter(modified_edge_surplus['nodes'], modified_edge_surplus['diameter'], marker='o', label='Modified Edge surplus')
    plt.xlabel('Number of nodes')
    plt.ylabel('Diameter')
    plt.title('football')
    plt.legend()
    # save the figure as pdf
    plt.savefig('./images/fitness_football_diameter.pdf')
