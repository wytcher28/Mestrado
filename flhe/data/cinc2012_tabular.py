import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def _walk_files(root: str):
    for dirpath, dirnames, files in os.walk(root):
        # ignora duplicatas e lixo do zip
        if "/sources/" in dirpath.replace("\\", "/"):
            continue
        if "__MACOSX" in dirpath:
            continue
        for f in files:
            yield dirpath, f

def _find_outcomes_file(root: str, prefer="a"):
    candidates = []
    for dirpath, f in _walk_files(root):
        fl = f.lower()
        if fl.startswith("outcomes") and fl.endswith(".txt"):
            candidates.append(os.path.join(dirpath, f))

    if not candidates:
        return None

    # prioriza Outcomes-a.txt na raiz (menor profundidade) e depois por nome
    def score(p):
        depth = p.replace("\\", "/").count("/")
        name = os.path.basename(p).lower()
        prefer_hit = 0 if f"outcomes-{prefer}.txt" in name else 1
        return (prefer_hit, depth, name)

    candidates.sort(key=score)
    return candidates[0]

def _find_patient_files(root: str):
    patient_files = []
    for dirpath, f in _walk_files(root):
        fl = f.lower()
        # pacientes são *.txt numéricos; exclui outcomes
        if fl.endswith(".txt") and not fl.startswith("outcomes"):
            base = os.path.splitext(f)[0]
            if base.isdigit():
                patient_files.append(os.path.join(dirpath, f))
    patient_files.sort()
    return patient_files

def _read_patient_timeseries(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Parameter" not in df.columns or "Value" not in df.columns:
        raise ValueError(f"Formato inesperado em: {path}. Colunas: {df.columns.tolist()}")

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    g = df.groupby("Parameter")["Value"]
    feats = {
        "mean": g.mean(),
        "min": g.min(),
        "max": g.max(),
        "last": g.apply(lambda s: s.dropna().iloc[-1] if len(s.dropna()) else np.nan),
    }

    out = {}
    for stat, series in feats.items():
        for param, val in series.items():
            out[f"{param}_{stat}"] = val
    return out

def load_cinc2012_as_tabular(root: str, outcome_col: str = "InHospitalDeath", prefer_set="a"):
    outcomes_path = _find_outcomes_file(root, prefer=prefer_set)
    if outcomes_path is None:
        raise FileNotFoundError(f"[CinC2012] Não encontrei Outcomes*.txt dentro de: {root}")

    print(f"[CinC2012] Outcomes escolhido: {outcomes_path}")

    outcomes = pd.read_csv(outcomes_path)
    if "RecordID" not in outcomes.columns:
        raise ValueError(f"[CinC2012] Outcomes sem RecordID. Colunas: {outcomes.columns.tolist()}")
    if outcome_col not in outcomes.columns:
        raise ValueError(f"[CinC2012] Outcomes sem {outcome_col}. Colunas: {outcomes.columns.tolist()}")

    outcomes = outcomes[["RecordID", outcome_col]].copy()
    outcomes["RecordID_int"] = outcomes["RecordID"].astype(int)

    patient_files = _find_patient_files(root)
    if not patient_files:
        raise FileNotFoundError(f"[CinC2012] Não encontrei arquivos de pacientes *.txt em: {root}")

    print(f"[CinC2012] Arquivos de pacientes encontrados: {len(patient_files)}")

    rec_to_file = {int(os.path.splitext(os.path.basename(p))[0]): p for p in patient_files}

    outcomes = outcomes[outcomes["RecordID_int"].isin(rec_to_file.keys())].reset_index(drop=True)
    print(f"[CinC2012] Pacientes a processar (com arquivo): {len(outcomes)}")

    rows = []
    for _, row in tqdm(outcomes.iterrows(), total=len(outcomes), desc="Processando pacientes CinC2012"):
        rid = int(row["RecordID_int"])
        feats = _read_patient_timeseries(rec_to_file[rid])
        feats[outcome_col] = int(row[outcome_col])
        rows.append(feats)

    df = pd.DataFrame(rows)
    df = df.dropna(axis=1, how="all")

    for c in df.columns:
        if c == outcome_col:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    return df
