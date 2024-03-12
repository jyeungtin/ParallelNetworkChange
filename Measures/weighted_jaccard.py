## libraries
import numpy as np 
import networkx as nx

## function
def weighted_jaccard_distance(G1, G2):
    
    '''
    Function to compute the weighted version of the Jaccard distance between two graphs
    using their adjacency matrices
    
    - Input: a pair of network graph objects
    - Output: Jaccard distance coefficient
    '''
    
    # getting adjacency matrices of the graphs
    A1 = nx.linalg.graphmatrix.adjacency_matrix (G1, weight='weight').todense()
    A2 = nx.linalg.graphmatrix.adjacency_matrix (G2, weight='weight').todense()

    # computing numerator and denominator of the weighted Jaccard distance
    numerator = np.sum (np.abs (A1 - A2))
    denominator = np.sum (np.maximum (A1, A2))

    # computing the weighted Jaccard distance
    jaccard_distance = numerator / denominator
    return jaccard_distance
