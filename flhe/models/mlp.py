import numpy as np
from sklearn.neural_network import MLPClassifier

def make_mlp(seed: int, input_dim: int):
    # MLP simples e estável para CPU
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=1,            # treinamos por "epochs" manualmente (rounds)
        warm_start=True,       # mantém pesos entre chamadas fit()
        random_state=seed,
        verbose=False,
    )

def get_weights(model) -> np.ndarray:
    params = []
    for w in model.coefs_:
        params.append(w.ravel())
    for b in model.intercepts_:
        params.append(b.ravel())
    return np.concatenate(params).astype(np.float64)

def set_weights(model, flat: np.ndarray, shapes):
    idx = 0
    new_coefs = []
    for s in shapes["coefs"]:
        size = int(np.prod(s))
        new_coefs.append(flat[idx:idx+size].reshape(s))
        idx += size
    new_intercepts = []
    for s in shapes["intercepts"]:
        size = int(np.prod(s))
        new_intercepts.append(flat[idx:idx+size].reshape(s))
        idx += size
    model.coefs_ = new_coefs
    model.intercepts_ = new_intercepts

def get_shapes(model):
    return {
        "coefs": [w.shape for w in model.coefs_],
        "intercepts": [b.shape for b in model.intercepts_],
    }

def init_model(model, X_sample, y_sample):
    # primeira chamada para inicializar estruturas internas
    model.fit(X_sample, y_sample)
    return model
