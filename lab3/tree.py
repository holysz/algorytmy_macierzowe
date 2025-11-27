import numpy as np
import matplotlib.pyplot as plt
from matrix import Matrix
import lab3

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


class Node:
    def __init__(self, rank, size, U=None, V=None, D=None):
        self.rank = rank
        self.size = size
        self.U = U
        self.V = V
        self.D = D
        self.children = []

def CompressMatrix(A, U, D, V, r, eps):
    significant_singular_values = D[D > eps]
    rank = min(r, len(significant_singular_values))
    return Node(rank, A.shape, U=U[:, :rank], V=V[:rank, :], D=D[:rank])

def rebuild_matrix(node):
    if not node.children:
        return node.U @ node.S @ node.V
    
    return np.vstack((np.hstack((rebuild_matrix(node.children[0]), rebuild_matrix(node.children[1])),
                      np.hstack((rebuild_matrix(node.children[2]), rebuild_matrix(node.children[3]))))))

def create_tree(A, rank, eps=1e-10, min_size=2):
    U, D, V = lab3.truncated_svd(A, rank + 1, eps)

    if min(A.shape) <= min_size or D[rank] <= eps:
        root = CompressMatrix(A, U, D, V, rank, eps)
    else:
        root = Node(0, A.shape)
        mid_row = A.shape[0] // 2
        mid_col = A.shape[1] // 2

        submatrices = [
            A[:mid_row, :mid_col],
            A[:mid_row, mid_col:],
            A[mid_row:, :mid_col],
            A[mid_row:, mid_col:]
        ]

        for subm in submatrices:
            if subm.size > 0:
                root.children.append(create_tree(subm, rank, eps, min_size))

    return root

def rebuild_matrix(node):
    if node.children == []:
        return node.U @ np.diag(node.D) @ node.V
    top_left = rebuild_matrix(node.children[0])
    top_right = rebuild_matrix(node.children[1])
    bottom_left = rebuild_matrix(node.children[2])
    bottom_right = rebuild_matrix(node.children[3])
    return np.vstack((np.hstack((top_left, top_right)),
                      np.hstack((bottom_left, bottom_right))))


def test(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    plt.subplot(1, 4, 1).imshow(img)
    plt.subplot(1, 4, 2).imshow(r)
    plt.subplot(1, 4, 3).imshow(r)
    plt.subplot(1, 4, 4).imshow(r)


test()