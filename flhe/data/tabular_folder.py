import os
import pandas as pd

def list_files_summary(folder: str):
    exts = {}
    for dirpath, _, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            exts[ext] = exts.get(ext, 0) + 1
    return dict(sorted(exts.items(), key=lambda x: (-x[1], x[0])))

def load_all_csvs(folder: str):
    csvs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(root, f))
    csvs = sorted(csvs)

    if not csvs:
        summary = list_files_summary(folder)
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em: {folder}. Extensões encontradas: {summary}"
        )

    dfs = [pd.read_csv(p) for p in csvs]
    df = pd.concat(dfs, ignore_index=True)
    return df

def basic_clean(df: pd.DataFrame):
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("UNK")
    return df
