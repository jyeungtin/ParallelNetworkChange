def hamming_distance(adj_arr):
  '''
  Calculates hamming distance between each pair of neighboring graphs at time t and t+1
  ## Parameters
  adj_arr: array of adjacency matrices at each time t

  ## Return 
  Array of hamming distances
  '''
  
  h_distances = []
  for graph in range(len(adj_matrices)-1):
    t_1 = adj_matrices[graph]
    t_2 = adj_matrices[graph+1]

    N = len(t_1)
  
    sum = 0
    for row_1, row_2 in zip(t_1, t_2):
      for value_1, value_2 in zip(row_1, row_2):
        sum += abs(value_1-value_2)/(N*(N-1))

    h_distances.append(sum)

  return h_distances
