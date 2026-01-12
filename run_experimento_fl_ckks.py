
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

import tenseal as ts

# Caminhos base
PROJECT_DIR = "/content/drive/MyDrive/FL_CKKS_PROJETO"
DATA_DIR = "/content/drive/MyDrive/synthea/csv"  # AJUSTE SE SEUS CSVs ESTIVEREM EM OUTRO LUGAR

NUM_CLIENTS = 5
NUM_ROUNDS = 5
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = os.path.join(PROJECT_DIR, "resultados")
FIG_DIR = os.path.join(PROJECT_DIR, "graficos")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------- DADOS Synthea ---------------- #

def carregar_synthea(data_dir: str):
    print(f"Carregando dados da Synthea a partir de: {data_dir}")
    patients = pd.read_csv(os.path.join(data_dir, "patients.csv"))
    conditions = pd.read_csv(os.path.join(data_dir, "conditions.csv"))

    print("patients:", patients.shape)
    print("conditions:", conditions.shape)

    target_condition = "Hypertension"
    cond_target = conditions[
        conditions["DESCRIPTION"].str.contains(target_condition, case=False, na=False)
    ]
    pos_ids = set(cond_target["PATIENT"].unique())

    labels = pd.DataFrame({
        "Id": patients["Id"],
        "LABEL": patients["Id"].isin(pos_ids).astype(int),
    })

    feat = patients[["Id", "BIRTHDATE", "RACE", "ETHNICITY", "GENDER", "STATE"]].copy()
    feat["BIRTHDATE"] = pd.to_datetime(feat["BIRTHDATE"], errors="coerce")
    ref_year = 2025
    feat["AGE"] = ref_year - feat["BIRTHDATE"].dt.year

    df = feat.merge(labels, on="Id")
    df = df[["Id", "AGE", "RACE", "ETHNICITY", "GENDER", "STATE", "LABEL"]]

    df = pd.get_dummies(
        df,
        columns=["RACE", "ETHNICITY", "GENDER", "STATE"],
        drop_first=False
    )

    X = df.drop(columns=["Id", "LABEL"]).values.astype("float32")
    y = df["LABEL"].values.astype("int64")

    print("X shape:", X.shape, "dtype:", X.dtype)
    print("y shape:", y.shape, "dtype:", y.dtype)
    print("positivos:", int(y.sum()))

    return X, y, df

# ---------------- DATASET & MODELO ---------------- #

class SyntheaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, d_in, d_out=2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 128)
        self.fc2 = nn.Linear(128, d_out)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def get_params(model):
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]

def set_params(model, params):
    sd = model.state_dict()
    for (k, v), p in zip(sd.items(), params):
        sd[k] = torch.tensor(p, dtype=v.dtype)
    model.load_state_dict(sd)

def clone_model(model, d_in):
    m = MLP(d_in)
    m.load_state_dict(model.state_dict())
    return m

def train_one_epoch(model, loader, lr=LR, device=DEVICE):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(Xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

def eval_metrics(model, loader, device=DEVICE):
    model.to(device)
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return acc, f1, auc

# -------------- CKKS HELPERS ---------------- #

def flatten_params(params):
    return np.concatenate([p.reshape(-1) for p in params]).astype(np.float64)

def unflatten_params(flat, template_params):
    new_params = []
    idx = 0
    for t in template_params:
        size = t.size
        new_params.append(flat[idx:idx+size].reshape(t.shape).astype(t.dtype))
        idx += size
    return new_params

# -------------- FL SEM HE ---------------- #

def fl_sem_he(X, y, d_in):
    print("\n=== INICIANDO FL SEM HE ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    train_ds = SyntheaDataset(X_train, y_train)
    test_ds = SyntheaDataset(X_test, y_test)

    indices = np.arange(len(train_ds))
    np.random.shuffle(indices)
    splits = np.array_split(indices, NUM_CLIENTS)

    client_datasets = [torch.utils.data.Subset(train_ds, idxs) for idxs in splits]
    client_loaders = [
        DataLoader(cd, batch_size=BATCH_SIZE, shuffle=True)
        for cd in client_datasets
    ]
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print("Tamanho de cada cliente:", [len(cd) for cd in client_datasets])

    global_plain = MLP(d_in)
    results_plain = []

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n[FL SEM HE] Rodada {rnd}")
        start_round = time.time()

        global_params = get_params(global_plain)
        agg_delta = [np.zeros_like(p) for p in global_params]
        total_examples = 0
        total_bytes = 0

        for cid, loader in enumerate(client_loaders):
            n_i = len(loader.dataset)
            print(f"  Cliente {cid} - exemplos: {n_i}")
            local_model = clone_model(global_plain, d_in)
            train_one_epoch(local_model, loader)
            local_params = get_params(local_model)
            delta_i = [lp - gp for lp, gp in zip(local_params, global_params)]

            flat_delta = flatten_params(delta_i)
            total_bytes += flat_delta.nbytes

            agg_delta = [ad + d * n_i for ad, d in zip(agg_delta, delta_i)]
            total_examples += n_i

        avg_delta = [ad / total_examples for ad in agg_delta]
        new_params = [gp + d for gp, d in zip(global_params, avg_delta)]
        set_params(global_plain, new_params)

        acc, f1, auc = eval_metrics(global_plain, test_loader)
        tempo_round = time.time() - start_round

        results_plain.append({
            "cenario": "FL_sem_HE",
            "round": rnd,
            "acc": acc,
            "f1": f1,
            "auc": auc,
            "tempo_seg": tempo_round,
            "bytes_transmitidos": total_bytes,
        })

        print(f"  Acurácia = {acc:.4f} | F1 = {f1:.4f} | AUC = {auc:.4f} | tempo = {tempo_round:.3f}s | bytes = {total_bytes}")

    return results_plain, X_train, X_test, y_train, y_test, client_datasets, test_ds

# -------------- FL COM CKKS (Arquitetura A/B) ---------------- #

def fl_com_ckks(d_in, X_train, X_test, y_train, y_test, client_datasets, test_ds):
    print("\n=== INICIANDO FL COM CKKS (Arquitetura A/B) ===")

    client_loaders = [
        DataLoader(cd, batch_size=BATCH_SIZE, shuffle=True)
        for cd in client_datasets
    ]
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    ctx_A = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx_A.global_scale = 2**40
    ctx_A.generate_galois_keys()
    ctx_A.generate_relin_keys()

    ctx_pub = ts.context_from(ctx_A.serialize())
    ctx_pub.make_context_public()

    global_he = MLP(d_in)
    template_params = get_params(global_he)
    results_he = []

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n[FL COM HE (CKKS)] Rodada {rnd}")
        start_round = time.time()

        global_params = get_params(global_he)

        enc_deltas = []
        total_examples = 0
        total_bytes = 0

        for cid, loader in enumerate(client_loaders):
            local_model = clone_model(global_he, d_in)
            n_i = len(loader.dataset)
            print(f"  Cliente {cid} - exemplos: {n_i}")
            train_one_epoch(local_model, loader)
            local_params = get_params(local_model)
            delta_i = [lp - gp for lp, gp in zip(local_params, global_params)]

            flat_delta = flatten_params(delta_i) * n_i
            total_examples += n_i

            enc_vec = ts.ckks_vector(ctx_pub, flat_delta)
            enc_deltas.append(enc_vec)

            enc_bytes = enc_vec.serialize()
            total_bytes += len(enc_bytes)

        enc_sum = None
        for enc_vec in enc_deltas:
            if enc_sum is None:
                enc_sum = enc_vec
            else:
                enc_sum += enc_vec

        enc_sum_A = ts.ckks_vector_from(ctx_A, enc_sum.serialize())
        flat_sum = np.array(enc_sum_A.decrypt())
        flat_avg = flat_sum / total_examples

        avg_delta_params = unflatten_params(flat_avg, template_params)
        new_params_he = [gp + d for gp, d in zip(global_params, avg_delta_params)]
        set_params(global_he, new_params_he)

        acc, f1, auc = eval_metrics(global_he, test_loader)
        tempo_round = time.time() - start_round

        results_he.append({
            "cenario": "FL_com_CKKS",
            "round": rnd,
            "acc": acc,
            "f1": f1,
            "auc": auc,
            "tempo_seg": tempo_round,
            "bytes_transmitidos": total_bytes,
        })

        print(f"  Acurácia = {acc:.4f} | F1 = {f1:.4f} | AUC = {auc:.4f} | tempo = {tempo_round:.3f}s | bytes = {total_bytes}")

    return results_he

# -------------- SALVAR RESULTADOS + GRÁFICOS ---------------- #

def salvar_resultados_e_graficos(results_plain, results_he):
    df_plain = pd.DataFrame(results_plain)
    df_he = pd.DataFrame(results_he)

    df_all = pd.merge(
        df_plain,
        df_he,
        on="round",
        suffixes=("_sem_he", "_com_ckks"),
    )

    df_all["overhead_tempo_seg"] = df_all["tempo_seg_com_ckks"] - df_all["tempo_seg_sem_he"]
    df_all["overhead_bytes"] = df_all["bytes_transmitidos_com_ckks"] - df_all["bytes_transmitidos_sem_he"]

    csv_path = os.path.join(RESULTS_DIR, "resultados_mestrado_synthea_fl_ckks.csv")
    df_all.to_csv(csv_path, index=False)
    print("\nResultados consolidados salvos em:", csv_path)

    rounds = df_all["round"]

    # 1) Acurácia
    plt.figure(figsize=(6,4))
    plt.plot(rounds, df_all["acc_sem_he"], marker="o", label="FL sem HE")
    plt.plot(rounds, df_all["acc_com_ckks"], marker="s", label="FL com CKKS")
    plt.xlabel("Rodada federada")
    plt.ylabel("Acurácia")
    plt.title("Acurácia por rodada - FL sem HE vs FL com CKKS")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_acuracia_por_rodada.png"), dpi=300)
    plt.close()

    # 2) AUC
    plt.figure(figsize=(6,4))
    plt.plot(rounds, df_all["auc_sem_he"], marker="o", label="FL sem HE")
    plt.plot(rounds, df_all["auc_com_ckks"], marker="s", label="FL com CKKS")
    plt.xlabel("Rodada federada")
    plt.ylabel("AUC")
    plt.title("AUC por rodada - FL sem HE vs FL com CKKS")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_auc_por_rodada.png"), dpi=300)
    plt.close()

    # 3) F1-score
    plt.figure(figsize=(6,4))
    plt.plot(rounds, df_all["f1_sem_he"], marker="o", label="FL sem HE")
    plt.plot(rounds, df_all["f1_com_ckks"], marker="s", label="FL com CKKS")
    plt.xlabel("Rodada federada")
    plt.ylabel("F1-score")
    plt.title("F1-score por rodada - FL sem HE vs FL com CKKS")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_f1_por_rodada.png"), dpi=300)
    plt.close()

    # 4) Tempo
    plt.figure(figsize=(6,4))
    plt.plot(rounds, df_all["tempo_seg_sem_he"]*1000, marker="o", label="FL sem HE")
    plt.plot(rounds, df_all["tempo_seg_com_ckks"]*1000, marker="s", label="FL com CKKS")
    plt.xlabel("Rodada federada")
    plt.ylabel("Tempo médio por rodada (ms)")
    plt.title("Tempo por rodada - FL sem HE vs FL com CKKS")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_tempo_por_rodada.png"), dpi=300)
    plt.close()

    # 5) Bytes
    plt.figure(figsize=(6,4))
    plt.plot(rounds, df_all["bytes_transmitidos_sem_he"]/1024, marker="o", label="FL sem HE")
    plt.plot(rounds, df_all["bytes_transmitidos_com_ckks"]/1024, marker="s", label="FL com CKKS")
    plt.xlabel("Rodada federada")
    plt.ylabel("Bytes transmitidos (KB)")
    plt.title("Comunicação por rodada - FL sem HE vs FL com CKKS")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_bytes_por_rodada.png"), dpi=300)
    plt.close()

    # 6a) Overhead tempo
    plt.figure(figsize=(6,4))
    plt.plot(rounds, df_all["overhead_tempo_seg"]*1000, marker="o")
    plt.xlabel("Rodada federada")
    plt.ylabel("Overhead de tempo (ms)")
    plt.title("Overhead de tempo por rodada (CKKS - sem HE)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06a_overhead_tempo.png"), dpi=300)
    plt.close()

    # 6b) Overhead bytes
    plt.figure(figsize=(6,4))
    plt.plot(rounds, df_all["overhead_bytes"]/1024, marker="o")
    plt.xlabel("Rodada federada")
    plt.ylabel("Overhead de comunicação (KB)")
    plt.title("Overhead de bytes por rodada (CKKS - sem HE)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06b_overhead_bytes.png"), dpi=300)
    plt.close()

    print("\nGráficos salvos em:", FIG_DIR)

# -------------- MAIN ---------------- #

def main():
    print("Dispositivo:", DEVICE)
    X, y, df_all = carregar_synthea(DATA_DIR)
    d_in = X.shape[1]
    results_plain, X_train, X_test, y_train, y_test, client_datasets, test_ds = fl_sem_he(X, y, d_in)
    results_he = fl_com_ckks(d_in, X_train, X_test, y_train, y_test, client_datasets, test_ds)
    salvar_resultados_e_graficos(results_plain, results_he)
    print("\nExperimento concluído com sucesso.")

if __name__ == "__main__":
    main()
