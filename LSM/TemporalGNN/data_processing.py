## libraries
import pandas as pd

## cleaning function
def load_and_clean_data (filepath, layer):

    """
    Function to load in data and clean it to only include a single layer from the multi-layer network
    """

    # loading in data
    data = pd.read_csv (filepath)

    # ensuring we only have single layer edges
    same_layer = data[data['ori_layer'] == data['des_layer']]
    same_layer['node_type'] = same_layer['ori_layer']

    # isolating a single layer
    data_layer = same_layer[same_layer['node_type'] == layer]

    return data_layer

