import numpy as np

class Node:
    def __init__(self, rank, size, sing_vals=None, U=None, V=None):
        self.rank = rank
        self.size = size
        self.sing_vals = sing_vals
        self.U = U
        self.V = V
        self.children = []

def svd_decomposition(A, k, eps):
    #TODO
    pass

def svd_compression(A, max_rank, eps):
    #TODO
    pass

def error(A, U, D, V):
    #TODO
    pass

def create_tree(A, max_rank, eps=1e-10, min_size=2):
    U, D, V = svd_compression(A, max_rank, eps)

    root = Node(0, A.shape)

    if min(A.shape) <= min_size or error(A, U, D, V) <= eps:
        root.U = U
        root.V = V
        return root
    
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
            root.children.append(create_tree(subm, max_rank, eps, min_size))

    return root

