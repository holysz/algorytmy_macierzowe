from matrix import Matrix, zeros
from tree import Node
import numpy as np

def power_iteration(M, max_iter=1000, eps=1e-8):
    v = np.random.rand(M.shape[1])
    v /= np.linalg.norm(v)
    prev = np.empty(M.shape[1])
    for _ in range(max_iter):
        prev[:] = v
        v = np.dot(M, v)
        v /= np.linalg.norm(v)
        if np.allclose(v, prev, atol=eps):
            break
    eigval = np.dot(v, np.dot(M, v)) / np.dot(v, v)
    return v, eigval

def truncated_svd(M, r, eps):
    U = np.zeros([M.shape[1], r])
    D = np.zeros(r)
    V = np.zeros([M.shape[1], r])

    A = M.T @ M
    for rank in range(r):
        eigvec, eigval = power_iteration(A)
        if eigval < eps:
            break

        singular = np.sqrt(eigval)
        D[rank] = singular

        V[:, rank] = eigvec
        U[:, rank] = A @ eigvec / eigval

        A -= singular * eigvec @ eigvec.T

    return U, D, V.T

def CompressMatrix(t_min: int, t_max: int, s_min: int, s_max: int,
                   U: Matrix, D: Matrix, V: Matrix, r: int) -> Node:
    pass