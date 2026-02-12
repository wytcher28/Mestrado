import time
import numpy as np

from flhe.models.mlp import init_model, get_shapes, get_flat_weights, set_flat_weights
from flhe.fl.utils import split_clients
from flhe.crypto.he_ckks import HEContext
from flhe.crypto.sym_aesgcm import pack_update, unpack_update, aes_encrypt_bytes, aes_decrypt_bytes

def _client_train_one_round(model, Xc, yc, seed=42):
    # 1 epoch/iter por round (mantém seu desenho)
    model.partial_fit(Xc, yc, classes=np.array([0,1]))
    return model

def run_federated_training(
    scenario: str,
    X_train, y_train,
    X_val, y_val,
    rounds=5,
    num_clients=5,
    seed=42,
    logger=None
):
    # init model
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1, warm_start=True, random_state=seed)
    model = init_model(model, X_train[:200], y_train[:200])

    shapes = get_shapes(model)
    clients = split_clients(X_train, y_train, num_clients=num_clients, seed=seed)

    # ===== CRYPTO SETUP =====
    he = None
    if scenario in ("he", "hybrid"):
        he = HEContext()  # usa seu setup atual

    # chave simétrica (apenas hybrid)
    sym_key = None
    if scenario == "hybrid":
        sym_key = HEContext.random_bytes(32)  # helper (ou os.urandom(32))

    # ===== TRAINING LOOP =====
    for r in range(1, rounds+1):
        t_round0 = time.time()

        flat_global = get_flat_weights(model)
        updates = []

        bytes_client_to_B = 0
        bytes_B_to_A = 0
        t_serverB_ms = 0.0
        t_serverA_ms = 0.0

        # SERVER A “prepara” (simulado)
        if scenario in ("he", "hybrid"):
            tA0 = time.time()
            # no seu design: A prepara CKKS / chaves públicas p/ B
            _ = he.public_context_blob()
            t_serverA_ms += (time.time() - tA0) * 1000.0

        # ===== CLIENTS =====
        for (Xc, yc) in clients:
            # treina local
            local_model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1, warm_start=True, random_state=seed)
            local_model = init_model(local_model, X_train[:200], y_train[:200])
            set_flat_weights(local_model, flat_global, shapes)

            local_model = _client_train_one_round(local_model, Xc, yc, seed=seed)
            flat_local = get_flat_weights(local_model)
            update = (flat_local - flat_global).astype(np.float32)

            if scenario == "baseline":
                # envia update "em claro"
                payload = update.tobytes()
                bytes_client_to_B += len(payload)
                updates.append(update.astype(np.float64))

            elif scenario == "he":
                # cliente cifra direto em CKKS (grande)
                ct = he.encrypt_vector(update.tolist())
                blob = he.serialize_ciphertext(ct)
                bytes_client_to_B += len(blob)
                updates.append(blob)  # ciphertexts

            elif scenario == "hybrid":
                # cliente: compacta + AES
                packed, shp = pack_update(update, compress=True)
                nonce, ct = aes_encrypt_bytes(sym_key, packed, aad=b"upd")
                # envia nonce+ct + shape (shape é pequeno)
                bytes_client_to_B += (len(nonce) + len(ct) + 16)  # +16 aprox para shape/metadata
                updates.append((nonce, ct, shp))

        # ===== SERVER B AGGREGATION =====
        tB0 = time.time()

        if scenario == "baseline":
            agg = np.mean(np.stack(updates, axis=0), axis=0).astype(np.float64)

        elif scenario == "he":
            # soma homomórfica
            agg_ct = None
            for blob in updates:
                ct = he.deserialize_ciphertext(blob)
                agg_ct = ct if agg_ct is None else (agg_ct + ct)
            agg_ct = agg_ct * (1.0/len(updates))
            # envia para A
            blob_agg = he.serialize_ciphertext(agg_ct)
            bytes_B_to_A += len(blob_agg)

        elif scenario == "hybrid":
            # transciphering: AES->plain->CKKS (custo no servidor B)
            agg_ct = None
            for (nonce, ct, shp) in updates:
                packed = aes_decrypt_bytes(sym_key, nonce, ct, aad=b"upd")
                vec = unpack_update(packed, shp, compressed=True).astype(np.float32)
                ct_he = he.encrypt_vector(vec.tolist())
                agg_ct = ct_he if agg_ct is None else (agg_ct + ct_he)
            agg_ct = agg_ct * (1.0/len(updates))
            blob_agg = he.serialize_ciphertext(agg_ct)
            bytes_B_to_A += len(blob_agg)

        t_serverB_ms += (time.time() - tB0) * 1000.0

        # ===== SERVER A UPDATE =====
        if scenario == "baseline":
            # aplica update direto
            new_flat = flat_global + agg.astype(np.float64)
            set_flat_weights(model, new_flat, shapes)

        else:
            # A “decripta” agregado e atualiza (simulado pela API atual)
            tA1 = time.time()
            agg_ct = he.deserialize_ciphertext(blob_agg)
            agg_vec = np.array(he.decrypt_vector(agg_ct), dtype=np.float64)
            new_flat = flat_global + agg_vec
            set_flat_weights(model, new_flat, shapes)
            t_serverA_ms += (time.time() - tA1) * 1000.0

        # ===== VALIDATION =====
        val_acc = model.score(X_val, y_val) if len(X_val) else None

        # F1
        from flhe.metrics.evaluation import evaluate_binary, safe_predict_proba
        if len(X_val):
            y_pred = model.predict(X_val)
            y_proba = safe_predict_proba(model, X_val)
            m = evaluate_binary(y_val, y_pred, y_proba)
            val_f1 = m["f1"]
        else:
            val_f1 = None

        latency_round_ms = (time.time() - t_round0) * 1000.0

        if logger:
            logger.log_round(
                round=r,
                val_accuracy=val_acc,
                val_f1=val_f1,
                latency_round_ms=latency_round_ms,
                bytes_client_to_B=bytes_client_to_B,
                bytes_B_to_A=bytes_B_to_A,
                serverB_ms=t_serverB_ms,
                serverA_ms=t_serverA_ms,
            )

    return model
