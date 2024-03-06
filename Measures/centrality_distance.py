import networkx as nx
import numpy as np

def centrality_distance(G1, G2, centrality_measure = 'betweenness'):
    '''
    ## Parameters
    G1: Graph object at time t
    G2: Graph object at time t+n
    centrality_measure: A centrality measure. Default is 'betweenness'. We can take only 'katz', 'betweenness' and 'load'.

    ## Return 
    Centrality-based distance metric
    '''

    if centrality_measure == 'betweenness':
        G1_centrality = np.array(list(nx.betweenness_centrality(G1).values()))
        G2_centrality = np.array(list(nx.betweenness_centrality(G2).values()))
    elif centrality_measure == 'katz':
        G1_centrality = np.array(list(nx.katz_centrality_numpy(G1).values()))
        G2_centrality = np.array(list(nx.katz_centrality_numpy(G2).values()))
    elif centrality_measure == 'load':
        G1_centrality = np.array(list(nx.load_centrality(G1).values()))
        G2_centrality = np.array(list(nx.load_centrality(G2).values()))
    else:
        raise(ValueError)

    diff = G2_centrality - G1_centrality

    distance = np.sqrt(np.sum(diff**2))

    return distance 

centrality_distance(G_bali_t3, G_bali_t4, centrality_measure='katz')
