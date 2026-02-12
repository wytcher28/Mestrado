import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def make_splits(df, target_column, seed=42, test_size=0.2):
    X = df.drop(columns=[target_column]).values.astype(np.float64)
    y = df[target_column].values.astype(int)

    # permite "usar tudo" sem split
    if test_size is None or test_size == 0 or test_size == 0.0:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_empty = X[:0].copy()
        y_empty = y[:0].copy()
        return X, X_empty, y, y_empty

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def split_clients(X, y, num_clients=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    parts = np.array_split(idx, num_clients)
    clients = [(X[p], y[p]) for p in parts]
    return clients
