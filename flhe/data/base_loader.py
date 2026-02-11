
import pandas as pd
from sklearn.model_selection import train_test_split

class BaseLoader:
    def __init__(self, path):
        self.path = path

    def split(self, df, target_column, test_size=0.2, seed=42):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=seed)
