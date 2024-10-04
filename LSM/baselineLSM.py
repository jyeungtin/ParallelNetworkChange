
## libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from tqdm import tqdm

## loading in graphs from pkl objects
## REPLACE FILE PATHS W/ YOUR LOCAL PATHS FOR THE PICKLES
with open('../Data/MTT_graphs/mig_graphs.pkl', 'rb') as f:
    mig_graphs = pickle.load(f)

with open('../Data/MTT_graphs/terrorism_graphs.pkl', 'rb') as f:
    terrorism_graphs = pickle.load(f)

with open('../Data/MTT_graphs/trade_graphs.pkl', 'rb') as f:
    trade_graphs = pickle.load(f)

def extract_network_features(graph):

    # extracting relevant network features
    n = len(graph.nodes())
    features = np.zeros((n, 4))
    
    for i, node in enumerate(tqdm(graph.nodes(), desc="Extracting network features")):
        features[i, 0] = nx.degree_centrality(graph)[node]
        features[i, 1] = nx.betweenness_centrality(graph)[node]
        features[i, 2] = nx.closeness_centrality(graph)[node]
        features[i, 3] = nx.eigenvector_centrality(graph, max_iter=1000)[node]
    
    return features

def create_edge_features(graph1, graph2, target_graph):

    # creatng edge features from two predictor networks
    n = len(graph1.nodes())
    edge_features = []
    edge_labels = []
    
    nodes = list(graph1.nodes())
    for i in tqdm(range(n), desc="Creating edge features"):
        for j in range(i+1, n):

            # features from graph1
            f1 = extract_network_features(graph1)
            edge_f1 = np.concatenate([f1[i], f1[j]])
            
            # features from graph2
            f2 = extract_network_features(graph2)
            edge_f2 = np.concatenate([f2[i], f2[j]])
            
            # combining features yay
            edge_feature = np.concatenate([edge_f1, edge_f2])
            edge_features.append(edge_feature)
            
            # label: 1 if edge exists in target graph, 0 otherwise
            label = 1 if target_graph.has_edge(nodes[i], nodes[j]) else 0
            edge_labels.append(label)
    
    return np.array(edge_features), np.array(edge_labels)

def latent_space_model(mig_graphs, trade_graphs, terrorism_graphs):
    results = []
    
    for year in tqdm(range(len(terrorism_graphs)), desc="Running latent space model"):

        # creating edge features and labels
        X, y = create_edge_features(mig_graphs[year], trade_graphs[year], 
                                   terrorism_graphs[year])
        
        # standardising features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # adding constant for intercept
        X_scaled = sm.add_constant(X_scaled)
        
        # fitting logistic regression
        model = sm.Logit(y, X_scaled)
        result = model.fit(disp=0)
        
        results.append(result)
    
    return results

# running model <3
model_results = latent_space_model(mig_graphs, trade_graphs, terrorism_graphs)

# analysing results to get coefs and pvals
def analyse_results(results):
    migration_coefs = []
    trade_coefs = []
    migration_pvals = []
    trade_pvals = []
    
    for result in tqdm (results, desc = "Analysing results"):
        params = result.params
        pvalues = result.pvalues
        
        # we're assuming the first 4 coefficients (after intercept) are for migration
        # and next 4 are for trade features
        migration_coefs.append(np.mean(params[1:5]))
        trade_coefs.append(np.mean(params[5:9]))
        migration_pvals.append(np.mean(pvalues[1:5]))
        trade_pvals.append(np.mean(pvalues[5:9]))
    
    return migration_coefs, trade_coefs, migration_pvals, trade_pvals

mig_coefs, trade_coefs, mig_pvals, trade_pvals = analyse_results (model_results)

# printing mig_coefs, trade_coefs, mig_pvals, trade_pvals
print (f'Coefficient on Migration: ', {mig_coefs}, "\nP-value: ", {mig_pvals})
print (f'Coefficient on Trade: ', {trade_coefs}, "\nP-value: ", {trade_pvals})
