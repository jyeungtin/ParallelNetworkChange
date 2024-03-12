## libraries
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.integrate import quad

## polynomial distance function
def polynomial_distance (G1, G2, k = 5, alpha = 1): 
    
    '''
    Function to compute the polynomial spectral distance between two graphs
    using their polynomial transformation of the eigenvalues of the
    of the adjacency matrix in combination with the eigenvectors of the
    adjacency matrix.
    
    - Input(s): 
            G1, G2 -> a pair of network graph objects
            k -> maximum degree of the polynomial used in 
                 the polynomial dissimilarity distance calculation
            alpha -> parameter controlling the influence of the 
                 polynomial transformation on the similarity score calculation
    - Output: Polynomial distance coefficient
    '''
    
    # getting adjacency matrices of the graphs
    A1 = nx.linalg.graphmatrix.adjacency_matrix (G1, weight = 'weight').todense()
    A2 = nx.linalg.graphmatrix.adjacency_matrix (G2, weight = 'weight').todense()
    
    # similarity 
    def similarity(A, k, alpha): 
        
        # eigen-decomposition
        eigVals, eigVec = np.linalg.eig(A)
        
        # shape of adjMatrix -> number of nodes
        n = np.shape(A)[0]
        
        # defining polynomial
        def polynomial(degree):
            
            # replicating formula
            return eigVals**degree / (n - 1) ** (alpha * (degree - 1))
        
        # diagonal matrix constructed from the sum of the polynomial transformations
        W = np.diag (sum([polynomial(k) for k in range (1, k + 1)]))
        
        # similarity score matrix 
        similarityScore = np.dot (np.dot (eigVec, W), eigVec.T)
        return similarityScore
    
    # computing similarityScore for each adjMatrix
    simi_A1 = similarity(A1, k, alpha)
    simi_A2 = similarity(A2, k, alpha)
    
    # polynomial distance
    polyDist = np.linalg.norm (simi_A1 - simi_A2, ord = "fro") / A1.shape[0] ** 2
    
    return polyDist
