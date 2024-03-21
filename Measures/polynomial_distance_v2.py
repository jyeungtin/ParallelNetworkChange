import numpy as np
import networkx as nx

def calculate_w(eigenvalues, k, alpha):
    """
    Helper function that calculate sum of polynomial function

    ## Parameters:
    eigenvalues - eigenvalues of graph laplacian
    k - number of hops to get from node i to node j
    alpha - weighting factor

    ## Returns
    sum of polynomial function
    """
    w = eigenvalues[0]
    n = len(eigenvalues)
    for i in range(2,k+1):
        coef = ((1/(n-1))**(alpha*(i-1)))
        w += coef*(eigenvalues[i-1]**(i))

    return w

def calculate_p(w, q):
    """
    Helper function that calculates dot product between w and q

    ##Parameters:
    w - sum of polynomial function
    q - eigenvectors

    ## Returns
    Dot product
    """
    q_w = np.dot(q,w)
    return np.dot(q_w, q.T)

def polynomial_distance(G1, G2):
    """
    Calculates polynomial distance between two graphs
    ## Parameters:
    G1 - networkx graph at time t
    G2 - networkx graph at time t+n

    ## Return
    Polynomial distance
    """
    n = len(G1.nodes())

    t_1 = nx.to_numpy_array(G1)
    t_2 = nx.to_numpy_array(G2)

    eigvec_1 = np.linalg.eig(t_1)[1]
    eigenvals_1 = nx.laplacian_spectrum(G1)

    eigvec_2 = np.linalg.eig(t_2)[1]
    eigenvals_2 = nx.laplacian_spectrum(G2)

    t1_w = calculate_w(eigenvals_1, 3, 0.5)
    t2_w = calculate_w(eigenvals_2, 3, 0.5)

    t1_p = calculate_p(t1_w, eigvec_1)
    t2_p = calculate_p(t2_w, eigvec_2)

    return (1/n**2)*np.linalg.norm(t1_p-t2_p)