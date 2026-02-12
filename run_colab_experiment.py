import os
import time
import pandas as pd

from flhe.data.tabular_folder import basic_clean
from flhe.data.cinc2012_tabular import load_cinc2012_as_tabular
from flhe.data.dahl_wfdb import load_dahl_as_tabular

from flhe.logging.experiment_logger import ExperimentLogger
from flhe.fl.federated_runner import run_federated_training
from flhe.metrics.plots import plot_metric_curve, plot_confusion_matrix
from flhe.metrics.evaluation import evaluate_binary, safe_predict_proba

def prepare_tabular(df, target_column):
    df = basic_clean(df)
    # remove não-numéricas exceto target
    non_num = [c for c in df.columns if df[c].dtype == "object" and c != target_column]
    if non_num:
        df = df.drop(columns=non_num)
    return df

def align_features(train_df: pd.DataFrame, test_df: pd.DataFrame, target_train: str, target_test: str):
    """
    Alinha o espaço de features do test ao do train:
    - define features do treino (todas exceto target_train)
    - reindexa test nessas features, preenchendo faltantes com 0
    - remove features extras do test
    """
    train_feats = [c for c in train_df.columns if c != target_train]
    test_feats = [c for c in test_df.columns if c != target_test]

    X_train_df = train_df[train_feats].copy()
    y_train = train_df[target_train].astype(int).values

    X_test_df = test_df[test_feats].copy()
    y_test = test_df[target_test].astype(int).values

    # reindex no espaço do treino
    X_test_aligned = X_test_df.reindex(columns=train_feats, fill_value=0.0)

    return X_train_df, y_train, X_test_aligned, y_test, train_feats

def standardize_like_train(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame):
    # scaling consistente: fit no treino, transform nos dois
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_df.values)
    X_te = scaler.transform(X_test_df.values)
    return X_tr, X_te

def main(DATASETS, targets, out_root="results", rounds=10, num_clients=5, seed=42):
    os.makedirs(out_root, exist_ok=True)

    # ===== TREINO (CinC 2012) =====
    print("\n[INFO] Carregando TREINO (CinC 2012) ...")
    t0 = time.time()
    df_train = load_cinc2012_as_tabular(DATASETS["treino"], outcome_col=targets["treino"], prefer_set="a")
    df_train = prepare_tabular(df_train, targets["treino"])
    print("[INFO] TREINO carregado:", df_train.shape, "em", round(time.time()-t0,2), "s")

    # ===== TESTE (DAHL/WFDB) =====
    print("\n[INFO] Preparando TESTE (DAHL/WFDB) ...")
    t1 = time.time()
    df_test = load_dahl_as_tabular(DATASETS["teste"], target_col=targets["teste"])
    df_test = prepare_tabular(df_test, targets["teste"])
    print("[INFO] TESTE carregado:", df_test.shape, "em", round(time.time()-t1,2), "s")

    # ===== ALINHAMENTO DE FEATURES =====
    print("\n[INFO] Alinhando features TREINO->TESTE ...")
    X_train_df, y_train, X_test_df, y_test, feat_space = align_features(
        df_train, df_test, targets["treino"], targets["teste"]
    )
    print("[INFO] Feature space:", len(feat_space))
    print("[INFO] Features presentes no TESTE (antes):", df_test.shape[1]-1)
    print("[INFO] Features após alinhamento TESTE:", X_test_df.shape[1])

    # padronização consistente
    X_tr, X_te = standardize_like_train(X_train_df, X_test_df)

    # split treino/val do TREINO (no mesmo espaço)
    from sklearn.model_selection import train_test_split
    X_tr_train, X_tr_val, y_tr_train, y_tr_val = train_test_split(
        X_tr, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )

    for scenario in ["baseline", "he", "hybrid"]:
        print(f"\n[INFO] Iniciando cenário: {scenario}")
        exp_dir = os.path.join(out_root, f"{scenario}_{time.strftime('%Y%m%d_%H%M%S')}")
        logger = ExperimentLogger(exp_dir)
        logger.log_meta(
            scenario=scenario,
            seed=seed,
            rounds=rounds,
            num_clients=num_clients,
            datasets=DATASETS,
            targets=targets,
            feature_space_size=len(feat_space),
        )

        model = run_federated_training(
            scenario=scenario,
            X_train=X_tr_train, y_train=y_tr_train,
            X_val=X_tr_val, y_val=y_tr_val,
            rounds=rounds,
            num_clients=num_clients,
            seed=seed,
            logger=logger
        )

        # ===== AVALIAÇÃO TESTE (ALINHADO) =====
        print("[INFO] Avaliando em TESTE (alinhado) ...")
        y_pred = model.predict(X_te)
        y_proba = safe_predict_proba(model, X_te)
        test_metrics = evaluate_binary(y_test, y_pred, y_proba)

        # ===== OFICIAL (IMAGEM) =====
        official_metrics = None
        print("[INFO] OFICIAL é imagem (pulado neste pipeline tabular).")

        logger.log_meta(final_test=test_metrics, final_official=official_metrics)
        logger.flush()

        # Gráficos padrão
        df_r = pd.read_csv(os.path.join(exp_dir, "rounds.csv"))
        xs = df_r["round"].tolist()

        plot_metric_curve(xs, df_r["val_accuracy"].tolist(), f"{scenario} - Val Accuracy", "Round", "Accuracy",
                          os.path.join(exp_dir, "plots", "val_accuracy.png"))
        plot_metric_curve(xs, df_r["val_f1"].tolist(), f"{scenario} - Val F1", "Round", "F1",
                          os.path.join(exp_dir, "plots", "val_f1.png"))
        plot_metric_curve(xs, df_r["latency_round_ms"].tolist(), f"{scenario} - Latency/Round", "Round", "ms",
                          os.path.join(exp_dir, "plots", "latency_round_ms.png"))
        plot_metric_curve(xs, df_r["bytes_client_to_B"].tolist(), f"{scenario} - Bytes Client->B", "Round", "bytes",
                          os.path.join(exp_dir, "plots", "bytes_client_to_B.png"))
        plot_metric_curve(xs, df_r["bytes_B_to_A"].tolist(), f"{scenario} - Bytes B->A", "Round", "bytes",
                          os.path.join(exp_dir, "plots", "bytes_B_to_A.png"))

        plot_confusion_matrix(test_metrics["cm"], f"{scenario} - Confusion Matrix (TESTE)",
                              os.path.join(exp_dir, "plots", "cm_test.png"))

        print("\n=== RESULTADOS FINAIS ===")
        print("Scenario:", scenario)
        print("TEST:", test_metrics)

