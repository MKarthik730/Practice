import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def attention_forward(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        S = np.where(mask, S, -1e9)
    P = softmax(S, axis=-1)
    O = P @ V
    return O, (Q, K, V, P)

def attention_backward(dO, cache):
    Q, K, V, P = cache
    d_k = Q.shape[-1]
    dV = P.T @ dO
    dP = dO @ V.T
    dS = P * (dP - np.sum(dP * P, axis=-1, keepdims=True))
    dS = dS / np.sqrt(d_k)
    dQ = dS @ K
    dK = dS.T @ Q
    return dQ, dK, dV

np.random.seed(0)
n, d = 4, 8
Q, K, V = (np.random.randn(n, d) for _ in range(3))
O, cache = attention_forward(Q, K, V)
dO = np.random.randn(*O.shape)
dQ, dK, dV = attention_backward(dO, cache)

def numerical_grad(f, x, dO, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        O_plus, _ = attention_forward(Q, K, V)
        x[idx] = orig - eps
        O_minus, _ = attention_forward(Q, K, V)
        x[idx] = orig
        grad[idx] = np.sum((O_plus - O_minus) * dO) / (2 * eps)
    return grad

print(np.allclose(dQ, numerical_grad(lambda: None, Q, dO), atol=1e-4))