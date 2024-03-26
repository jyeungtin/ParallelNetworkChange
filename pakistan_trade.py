import pandas as pd
import zipfile
import networkx as nx

def main():
    zf = zipfile.ZipFile('Data/trade_network_data.csv.zip') 
    df = pd.read_csv(zf.open('trade_network_data.csv'), index_col=0)

    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

    graphs = []

    for i in sorted(df['year'].unique()):

        year_graph = df[(df['year']==i)]
        edges = pd.DataFrame(
            {
                "source": list(df['location_name_1']),
                "target": list(df['location_name_2']),
                "weight": list(df['edge_value'])
            }
        )
        G = nx.from_pandas_edgelist(edges, edge_attr=True)
        G.graph['year'] = int(i)

        target_node = 'Pakistan'

        # Get the one-hop neighbors of the target node
        one_hop_neighbors = list(G.neighbors(target_node))
        one_hop_neighbors.append(target_node)  # Adding the target node itself

        # Create a subgraph with only the specified node and its one-hop neighbors
        subgraph = G.subgraph(one_hop_neighbors)

        graphs.append(G)

    return graphs

if __name__ == "__main__":
    main()