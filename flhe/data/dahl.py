
import os
import pandas as pd
from src.flhe.data.base_loader import BaseLoader

class DahlLoader(BaseLoader):

    def load_dataset(self):
        files = [f for f in os.listdir(self.path) if f.endswith(".csv")]

        dfs = [pd.read_csv(os.path.join(self.path, f)) for f in files]
        return pd.concat(dfs, ignore_index=True)

