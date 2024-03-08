import networkx as nx
import numpy as np

def hamming_distance(G1, G2):
  '''
  Calculates hamming distance between each pair of neighboring graphs at time t and t+1
  ## Parameters
  G1: graph at time t
  G2: graph at time t+1

  ## Return 
  Hamming distance value
  '''

  t_1 = nx.to_numpy_array(G1)
  t_2 = nx.to_numpy_array(G2)

  N = len(t_1)

  sum = 0
  for row_1, row_2 in zip(t_1, t_2):
      for value_1, value_2 in zip(row_1, row_2):
          sum += np.linalg.norm(value_1-value_2)/(N*(N-1))
  
  return sum
