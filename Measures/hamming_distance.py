import networkx as nx

def hamming_distance(G1, G2):
  '''
  Calculates hamming distance between two graphs
  ## Parameters
  G1: graph at time t
  G2: graph at time t+n

  ## Return 
  Hamming distance value
  '''

  t_1 = nx.to_numpy_array(G1)
  t_2 = nx.to_numpy_array(G2)

  N = len(t_1)

  sum = 0
  for row_1, row_2 in zip(t_1, t_2):
      for value_1, value_2 in zip(row_1, row_2):
          sum += abs(value_1-value_2)/(N*(N-1))
  
  return sum
