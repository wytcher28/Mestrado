import os, zlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def aes_encrypt_bytes(key: bytes, plaintext: bytes, aad: bytes = b""):
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, aad)
    return nonce, ct

def aes_decrypt_bytes(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes = b""):
    aesgcm = AESGCM(key)
    pt = aesgcm.decrypt(nonce, ciphertext, aad)
    return pt

def pack_update(update_vec, compress: bool = True):
    """
    update_vec: np.ndarray (float64/float32)
    Retorna bytes (possivelmente comprimidos).
    """
    import numpy as np
    arr = np.asarray(update_vec, dtype=np.float32)  # reduz tamanho
    raw = arr.tobytes(order="C")
    if compress:
        raw = zlib.compress(raw, level=6)
    return raw, arr.shape

def unpack_update(payload: bytes, shape, compressed: bool = True):
    import numpy as np
    raw = payload
    if compressed:
        raw = zlib.decompress(raw)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    return arr

