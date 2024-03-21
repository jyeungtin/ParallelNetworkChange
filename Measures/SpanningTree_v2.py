import networkx as nx
import numpy as np
import math

def spanning_tree_num(G):
    """
    Helper function to calculate number of spanning trees of a graph
    ## Parameters:
    G - singular networkx graph

    ## Return
    Number of spanning trees
    """
    eigenvalues = nx.normalized_laplacian_spectrum(G)
    nonzero_eig = eigenvalues[np.nonzero(eigenvalues)]
    N = len(nonzero_eig)

    product = 1
    for e in range(1,len(nonzero_eig)):
        product = product * nonzero_eig[e]
    
    return (1/N)*product

def st_distance_v2(G1, G2):
    """
    Calculates spanning tree distance using difference in number of spanning trees
    ## Parameters:
    G1 - networkx graph at time t
    G2 - networkx graph at time t+n

    ## Return
    Spanning tree distance
    """
    st_1 = spanning_tree_num(G1)
    st_2 = spanning_tree_num(G2)

    return abs(math.log(st_1)-math.log(st_2))
