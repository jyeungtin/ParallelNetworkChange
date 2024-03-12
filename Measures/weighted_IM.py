## libraries
import numpy as np 
import networkx as nx
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.integrate import quad

## IM distance function
def weighted_ipsenMikhailov_distance (G1, G2, hwhm = 0.08): 
    
    '''
    Function to compute the polynomial spectral distance between two graphs
    using their polynomial transformation of the eigenvalues of the
    of the adjacency matrix in combination with the eigenvectors of the
    adjacency matrix.
    
    - Input(s): 
            G1, G2 -> a pair of network graph objects
            hwhm -> half with at half maximum of the lorentzian kernel
    - Output: Polynomial distance coefficient
    '''
    
    # adjacency matrices
    A1 = nx.linalg.graphmatrix.adjacency_matrix (G1, weight = 'weight').todense()
    A2 = nx.linalg.graphmatrix.adjacency_matrix (G2, weight = 'weight').todense()
    
    # baseline function for IM distances
    def IMdistance (A1, A2, hwhm): 
            
        # number of nodes
        n = len(A1)
            
        # Laplacians
        L1 = laplacian (A1, normed = True)
        L2 = laplacian (A2, normed = True)
            
        # modes for the positive-semidefinite laplacian
        w1 = np.sqrt (np.abs (eigh (L1)[0][1:]))
        w2 = np.sqrt (np.abs (eigh (L2)[0][1:]))
            
        # norm of both spectra
        norm1 = (n - 1) * np.pi / 2 - np.sum (np.arctan (-w1 / hwhm))
        norm2 = (n - 1) * np.pi / 2 - np.sum (np.arctan (-w2 / hwhm))
            
        # spectral densitites
        density1 = lambda w: np.sum (hwhm / ((w - w1) ** 2 + hwhm**2)) / norm1
        density2 = lambda w: np.sum (hwhm / ((w - w2) ** 2 + hwhm**2)) / norm2
            
        # IM distance
        func = lambda w: (density1(w) - density2(w)) ** 2
        return np.sqrt (quad (func, 0, np.inf, limit = 100)[0])

    # computing distance
    distance = IMdistance (A1, A2, hwhm)
    return distance
