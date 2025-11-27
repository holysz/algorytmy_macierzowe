import numpy as np
from matrix import Matrix

class Node:
    def __init__(self, rank, size, U=None, V=None, D=None):
        self.rank = rank
        self.size = size
        self.U = U
        self.V = V
        self.D = D
        self.children = []

def truncatedSVD(A, r, eps):
    raise "UNIMPLEMENTED"
    
def CompressMatrix(A, U, D, V, r, eps):
    significant_singular_values = D[D > eps]
    rank = min(rank, len(significant_singular_values))
    return Node(r, A.shape, U=U[:, :rank], V=V[:rank, :], D=D[:rank])

def rebuild_matrix(node):
    if not node.children:
        return node.U @ node.S @ node.V
    
    return np.vstack((np.hstack((rebuild_matrix(node.children[0]), rebuild_matrix(node.children[1])),
                      np.hstack((rebuild_matrix(node.children[2]), rebuild_matrix(node.children[3]))))))

def create_tree(A, rank, eps=1e-10, min_size=2):
    U, D, V = truncatedSVD(A, rank + 1, eps)

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

