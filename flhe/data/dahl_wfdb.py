import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb

def _stats_features(x: np.ndarray, prefix: str):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}

    mean = float(np.mean(x))
    std = float(np.std(x))
    mn = float(np.min(x))
    mx = float(np.max(x))
    med = float(np.median(x))
    q25 = float(np.quantile(x, 0.25))
    q75 = float(np.quantile(x, 0.75))
    iqr = float(q75 - q25)
    rms = float(np.sqrt(np.mean(x**2)))
    energy = float(np.sum(x**2))
    zc = int(np.sum(np.diff(np.signbit(x)) != 0))

    if std > 0:
        skew = float(np.mean(((x - mean) / std) ** 3))
        kurt = float(np.mean(((x - mean) / std) ** 4) - 3.0)
    else:
        skew = 0.0
        kurt = 0.0

    t = np.arange(x.size, dtype=np.float64)
    if x.size > 1:
        cov = np.cov(t, x, bias=True)[0, 1]
        var = np.var(t)
        slope = float(cov / var) if var > 0 else 0.0
    else:
        slope = 0.0

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": mn,
        f"{prefix}_max": mx,
        f"{prefix}_median": med,
        f"{prefix}_iqr": iqr,
        f"{prefix}_rms": rms,
        f"{prefix}_energy": energy,
        f"{prefix}_zero_cross": zc,
        f"{prefix}_skew": skew,
        f"{prefix}_kurt": kurt,
        f"{prefix}_slope": slope,
    }

def _label_from_record_name(rec: str):
    r = rec.lower()
    # HS = 1, LS = 0
    if r.startswith("sshs") or r.startswith("ssbn13hs"):
        return 1
    if r.startswith("ssls") or r.startswith("ssbn13ls"):
        return 0
    return None

def load_dahl_as_tabular(root: str, target_col: str = "label"):
    records_path = os.path.join(root, "RECORDS")
    if not os.path.exists(records_path):
        raise FileNotFoundError(f"[DAHL] Não encontrei RECORDS em: {root}")

    with open(records_path, "r") as f:
        records = [line.strip() for line in f if line.strip()]

    print(f"[DAHL] Total de registros em RECORDS: {len(records)}")

    rows = []
    skipped_no_label = 0
    skipped_read_error = 0

    for rec in tqdm(records, desc="Extraindo features (DAHL/WFDB)"):
        y = _label_from_record_name(rec)
        if y is None:
            skipped_no_label += 1
            continue

        try:
            r = wfdb.rdrecord(os.path.join(root, rec))
            sig = r.p_signal
        except Exception:
            skipped_read_error += 1
            continue

        if sig is None or sig.ndim != 2:
            skipped_read_error += 1
            continue

        feats = {target_col: int(y), "record": rec}

        for ch in range(sig.shape[1]):
            feats.update(_stats_features(sig[:, ch], prefix=f"ch{ch}"))

        try:
            feats["fs"] = float(r.fs) if r.fs is not None else np.nan
        except Exception:
            feats["fs"] = np.nan

        rows.append(feats)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"[DAHL] DataFrame vazio. "
            f"Sem label: {skipped_no_label}, erros leitura: {skipped_read_error}. "
            f"Exemplo RECORDS: {records[:10]}"
        )

    for c in df.columns:
        if c in ("record", target_col):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    print(f"[DAHL] Usados: {len(df)} | Sem label: {skipped_no_label} | Erro leitura: {skipped_read_error}")
    return df

