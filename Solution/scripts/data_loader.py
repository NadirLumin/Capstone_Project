import pandas as pd
from sklearn.model_selection import KFold

def load_datasets():
    training_data = pd.read_csv("data/final_training_data.csv")
    outdated_terms = pd.read_csv("data/cleaned_synonyms_data.csv")
    return training_data, outdated_terms

def get_kfolds(data, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(kf.split(data))
