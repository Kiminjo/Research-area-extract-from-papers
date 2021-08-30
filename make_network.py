import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def network_draw(network_name, threshold) :
    print('edge threshold : {}'.format(threshold))

    network = pd.read_csv(network_name, index_col=0)
    data = network.where(network>threshold, 0)

    G1 = nx.convert_matrix.from_pandas_adjacency(network)
    G2 = nx.convert_matrix.from_pandas_adjacency(data)
    G2.remove_nodes_from(list(nx.isolates(G2)))

    print(G1)
    print(G2)

    fig, ax = plt.subplots(figsize=(16,10))
    pos = nx.kamada_kawai_layout(G2)
    nx.draw(G2, pos)
    plt.show()

