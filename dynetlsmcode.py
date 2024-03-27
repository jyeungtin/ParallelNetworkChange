import numpy as np
from dynetlsm import DynamicNetworkLSM
from sklearn.metrics import accuracy_score

# Create a list of adjacency matrices from your graphs
adj_mig = [nx.to_numpy_array(g) for g in mig_graphs]
adj_trade = [nx.to_numpy_array(g) for g in trade_graphs]
adj_terrorism = [nx.to_numpy_array(g) for g in terrorism_graphs]

# Combine the adjacency matrices
adj_combined = adj_mig + adj_trade

# Create a DynamicNetworkLSM object from your adjacency matrices
model = DynamicNetworkLSM(adj_combined)

# Fit the model to your data
model.fit()

# Predict the edges in the terrorism network
preds = model.predict()

# The last element in preds is the prediction for the next time step
pred_terrorism = preds[-1]

# Flatten the adjacency matrices and predictions for accuracy calculation
true_labels = adj_terrorism[-1].flatten()
pred_labels = (pred_terrorism > 0.5).flatten()  # Threshold can be adjusted

# Calculate accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print(f'Accuracy: {accuracy}')
