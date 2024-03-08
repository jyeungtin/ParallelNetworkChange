import networkx as nx

def jaccard_distance(G1, G2):
    '''
    Calculates jaccard distance between two graphs (G1 and G2)

    ## Parameters:
    G1: Networkx graph object 1
    G2: Networkx graph object 2

    ## Return 
    Jaccard distance value
    '''
    t_1 = nx.to_numpy_array(G1)
    t_2 = nx.to_numpy_array(G2)

    N = len(t_1)

    sum = 0
    absolute = 0
    maximum = 0
    for row_1, row_2 in zip(t_1, t_2):
        for value_1, value_2 in zip(row_1, row_2):
            absolute += abs(value_1-value_2)
            maximum += max(value_1,value_2)
        
    sum += absolute/maximum

    return sum