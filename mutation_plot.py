import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    neighborhood_mutation = pd.read_csv('./results/neighborhood_mutate/benchmark.csv', index_col=0)
    random_mutation = pd.read_csv('./results/random_mutate/benchmark.csv', index_col=0)

    mutation_functions = ['neighborhood', 'random']

    neighborhood_mutation = neighborhood_mutation[neighborhood_mutation['graph'] == 'email-EU-core']
    random_mutation = random_mutation[random_mutation['graph'] == 'email-EU-core']

    neighborhood_mutation[neighborhood_mutation == 'email-EU-core']['mutation'] = mutation_functions[0]
    random_mutation[random_mutation == 'email-EU-core']['mutation'] = mutation_functions[1]

    # Make a scatter plot of the edge density vs. the number of nodes in the graph
    plt.scatter(neighborhood_mutation['nodes'], neighborhood_mutation['edge_density'], marker='v', c='red', label=mutation_functions[0])
    plt.scatter(random_mutation['nodes'], random_mutation['edge_density'], marker='o', c='purple', label=mutation_functions[1])
    plt.xlabel('Number of nodes')
    plt.ylabel('Edge density')
    plt.title('email-EU-core')
    plt.legend()

    # save the figure as pdf
    plt.savefig('./images/mutation_email-EU-core_mutation_plot.pdf')

    plt.close()

    # Make a scatter plot of the diameter vs. the number of nodes in the graph
    plt.scatter(neighborhood_mutation['nodes'], neighborhood_mutation['diameter'], marker='v', c='red', label=mutation_functions[0])
    plt.scatter(random_mutation['nodes'], random_mutation['diameter'], marker='o', c='purple', label=mutation_functions[1])
    plt.xlabel('Number of nodes')
    plt.ylabel('Diameter')
    plt.title('email-EU-core')
    plt.legend()

    # save the figure as pdf
    plt.savefig('./images/mutation_email-EU-core_diameter_plot.pdf')



