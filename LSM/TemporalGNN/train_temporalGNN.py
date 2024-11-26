## imports
import pandas as pd
from temporal_gnn import train_temporal_gnn, set_seed
from data_processing import load_and_clean_data

def main():
    """
    Main function to load data, clean it, and train single-layer Temporal GNN.
    """
    # set seed 
    seed = 13 
    set_seed(seed)

    # file path to original dataset
    filepath = 'Data/clean_edge.csv'
    
    # data preprocessing
    print("🧹 Loading and cleaning data...")
    df = load_and_clean_data (filepath, layer = 4)
    
    # training Temporal GNN
    print("🏋️ Training Temporal GNN...")
    max_auc = train_temporal_gnn(df)
    
    print(f"🎉 Max AUC achieved: {max_auc:.4f}")

if __name__ == "__main__":
    main()