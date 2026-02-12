import tenseal as ts
import numpy as np

def make_context():
    # parâmetros iniciais (ajustáveis)
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2**40
    return ctx

def public_context(ctx):
    pub = ctx.copy()
    pub.make_context_public()
    return pub

def encrypt_vector(ctx_public, vec: np.ndarray):
    return ts.ckks_vector(ctx_public, vec.tolist())

def decrypt_vector(ctx_secret, enc_vec):
    return np.array(enc_vec.decrypt(), dtype=np.float64)

def add_enc(a, b):
    return a + b

def scale_enc(enc, factor: float):
    return enc * factor
