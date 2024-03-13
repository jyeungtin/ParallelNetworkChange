import networkx as nx
import numpy as np

def spanning_tree_similarity(G1, G2):
    '''
    ## Parameters
    G1: Graph object at time t
    G2: Graph object at time t+n

    ## Return 
    Spanning tree similarity metric
    '''

    # create empty list of eigenvalues 
    eigens = []

    for G in [G1, G2]:
        L = nx.normalized_laplacian_matrix(G) # generate the Laplacian

        L_eigens = np.linalg.eigvals(L.toarray()) # obtain the eigenvalues of L

        L_eigens = L_eigens[L_eigens>0] # get eigenvalues that are larger than 0 

        eigens.append(sorted(L_eigens, reverse=False)) # sort from small to big 

    # calculate the product of eigen values
        
    ST_vals = []
    
    for eigen in eigens:
        prod = np.prod(eigen)

        ST = prod / len(eigen)

        ST_vals.append(ST)
    
    # calculate the difference between the two values 
    distance = np.abs(np.log(ST_vals[1]) - np.log(ST_vals[0]))

    return distance
