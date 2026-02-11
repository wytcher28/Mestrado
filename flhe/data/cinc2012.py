
import os
import pandas as pd
from src.flhe.data.base_loader import BaseLoader

class CinC2012Loader(BaseLoader):

    def load_dataset(self):
        files = [f for f in os.listdir(self.path) if f.endswith(".csv")]

        dataframes = []
        for file in files:
            df = pd.read_csv(os.path.join(self.path, file))
            dataframes.append(df)

        full_df = pd.concat(dataframes, ignore_index=True)
        return full_df

