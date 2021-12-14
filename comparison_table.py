import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load the data with run, graph, diameter, edge_density, nodes, edges, objective, algorithm
    df = pd.read_csv('results/all.csv', index_col=0)

    # separate the data by algorithm
    edge_surplus = df[df['objective'] == 'edge_surplus']
    modified_edge_surplus = df[df['objective'] == 'modified_edge_surplus']

    # group the data by algorithm and graph
    edge_surplus_grouped = edge_surplus.groupby(['graph', 'algorithm'])
    modified_edge_surplus_grouped = modified_edge_surplus.groupby(['graph', 'algorithm'])

    # crate a table with the mean diameter, edge_density, nodes, edges
    edge_surplus_table = edge_surplus_grouped[['diameter', 'edge_density', 'nodes', 'edges']].mean()
    modified_edge_surplus_table = modified_edge_surplus_grouped[['diameter', 'edge_density', 'nodes', 'edges']].mean()

    # print the table with the graph on the x-axis and the algorithm for every metric on the y-axis
    print(edge_surplus_table.to_latex(index=True, header=True, float_format='%.2f'))
    print(modified_edge_surplus_table.to_latex(index=True, header=True, float_format='%.2f'))

    # count the number of missing values where algorithm is geneticAlgorithm and graph is wiki-Vote
    missing_values = edge_surplus[(edge_surplus['algorithm'] == 'geneticAlgorithm') & (edge_surplus['graph'] == 'wiki-Vote')].isnull().sum()
    print('Missing values:', missing_values['diameter'])
    missing_values = modified_edge_surplus[(modified_edge_surplus['algorithm'] == 'geneticAlgorithm') & (modified_edge_surplus['graph'] == 'wiki-Vote')].isnull().sum()
    print('Missing values:', missing_values['diameter'])

    print(edge_surplus[(edge_surplus['algorithm'] == 'geneticAlgorithm') & (edge_surplus['graph'] == 'wiki-Vote')])
    print(modified_edge_surplus[(modified_edge_surplus['algorithm'] == 'geneticAlgorithm') & (modified_edge_surplus['graph'] == 'wiki-Vote')])

    # for diameter  edge_density  nodes  edges
    for graph in ['dolphins', 'football', 'C-elegans-frontal', 'email-EU-core', 'wiki-Vote']:
        print(graph, end=' & ')
        for metric in ['nodes', 'diameter', 'edge_density']:
            print(' & '.join(map(lambda x: f'{x:.2f}', edge_surplus_table.loc[graph, metric].to_list())), end=' & ')
        print()

    print()
    # for diameter  edge_density  nodes  edges
    for graph in ['dolphins', 'football', 'C-elegans-frontal', 'email-EU-core', 'wiki-Vote']:
        print(graph, end=' & ')
        for metric in ['nodes', 'diameter', 'edge_density']:
            print(' & '.join(map(lambda x: f'{x:.2f}', modified_edge_surplus_table.loc[graph, metric].to_list())), end=' & ')
        print()
