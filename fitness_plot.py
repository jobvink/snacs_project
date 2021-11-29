import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    paper = pd.read_csv('./results/ga_paper.csv', index_col=0)
    ours = pd.read_csv('./results/ga_ours.csv', index_col=0)

    fitness_functions = ['neighborhood', 'random']

    paper = paper[paper['graph'] == 'email-EU-core']
    ours = ours[ours['graph'] == 'email-EU-core']

    # Make a scatter plot of the edge density vs. the number of nodes in the graph
    plt.scatter(paper['nodes'], paper['edge_density'], marker='v', label='Edge surplus')
    plt.scatter(ours['nodes'], ours['edge_density'], marker='o', label='Modified Edge surplus')
    plt.xlabel('Number of nodes')
    plt.ylabel('Edge density')
    plt.title('email-EU-core')
    plt.legend()
    # save the figure as pdf
    plt.savefig('./images/fitness_email-EU-core_edge_density.pdf')

    plt.close()

    # Make a scatter plot of the diameter vs. the number of nodes in the graph
    plt.scatter(paper['nodes'], paper['diameter'], marker='v', label='Edge surplus')
    plt.scatter(ours['nodes'], ours['diameter'], marker='o', label='Modified Edge surplus')
    plt.xlabel('Number of nodes')
    plt.ylabel('Diameter')
    plt.title('email-EU-core')
    plt.legend()
    # save the figure as pdf
    plt.savefig('./images/fitness_email-EU-core_diameter.pdf')

