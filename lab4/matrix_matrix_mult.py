from lab3.tree import Node, rebuild_matrix, truncated_svd, compress_matrix
from lab4.matrix_matrix_add import matrix_matrix_add
import numpy as np

def matrix_matrix_mult(v, w, max_rank, eps):
    if not v.children and not w.children:
        if v.rank == 0 or w.rank == 0:
            return Node(0, v.size)
        elif v.size == (1,1) and w.size == (1,1):
            val_v = v.U[0,0]*v.D[0]*v.V[0,0] if v.rank>0 else 0
            val_w = w.U[0,0]*w.D[0]*w.V[0,0] if w.rank>0 else 0
            val = val_v * val_w
            U = np.array([[val]], dtype=np.float64)
            D = np.array([1.0], dtype=np.float64)
            V = np.array([[1.0]], dtype=np.float64)
            return Node(1, (1,1), U, D, V)
        else:
            A = v.U @ np.diag(v.D) @ v.V
            B = w.U @ np.diag(w.D) @ w.V
            M = A @ B
            U, D, V = truncated_svd(M, max_rank+1, eps)
            return compress_matrix(M, U, D, V, max_rank, eps)
        
    elif v.children and w.children:
        Y00 = matrix_matrix_add(
            matrix_matrix_mult(v.children[0], w.children[0], max_rank, eps),
            matrix_matrix_mult(v.children[1], w.children[2], max_rank, eps),
            max_rank, eps
        )
        Y01 = matrix_matrix_add(
            matrix_matrix_mult(v.children[0], w.children[1], max_rank, eps),
            matrix_matrix_mult(v.children[1], w.children[3], max_rank, eps),
            max_rank, eps
        )
        Y10 = matrix_matrix_add(
            matrix_matrix_mult(v.children[2], w.children[0], max_rank, eps),
            matrix_matrix_mult(v.children[3], w.children[2], max_rank, eps),
            max_rank, eps
        )
        Y11 = matrix_matrix_add(
            matrix_matrix_mult(v.children[2], w.children[1], max_rank, eps),
            matrix_matrix_mult(v.children[3], w.children[3], max_rank, eps),
            max_rank, eps
        )
        root = Node(0, v.size)
        root.children = [Y00, Y01, Y10, Y11]
        return root
    
    elif not v.children and w.children:
        mid_row = w.children[0].size[0]
        mid_col = w.children[0].size[1]

        U1 = v.U[:mid_row, :]
        U2 = v.U[mid_row:, :]
        V1 = v.V[:, :mid_col]
        V2 = v.V[:, mid_col:]

        Y00 = matrix_matrix_mult(Node(v.rank, U1.shape, U1, v.D, V1), w.children[0], max_rank, eps)
        Y01 = matrix_matrix_mult(Node(v.rank, U1.shape, U1, v.D, V2), w.children[1], max_rank, eps)
        Y10 = matrix_matrix_mult(Node(v.rank, U2.shape, U2, v.D, V1), w.children[2], max_rank, eps)
        Y11 = matrix_matrix_mult(Node(v.rank, U2.shape, U2, v.D, V2), w.children[3], max_rank, eps)

        root = Node(0, v.size)
        root.children = [Y00, Y01, Y10, Y11]
        return root
    
    else:
        mid_row = v.children[0].size[0]
        mid_col = v.children[0].size[1]

        U1 = w.U[:mid_row, :]
        U2 = w.U[mid_row:, :]
        V1 = w.V[:, :mid_col]
        V2 = w.V[:, mid_col:]

        Y00 = matrix_matrix_mult(v.children[0], Node(w.rank, U1.shape, U1, w.D, V1), max_rank, eps)
        Y01 = matrix_matrix_mult(v.children[1], Node(w.rank, U1.shape, U1, w.D, V2), max_rank, eps)
        Y10 = matrix_matrix_mult(v.children[2], Node(w.rank, U2.shape, U2, w.D, V1), max_rank, eps)
        Y11 = matrix_matrix_mult(v.children[3], Node(w.rank, U2.shape, U2, w.D, V2), max_rank, eps)

        root = Node(0, v.size)
        root.children = [Y00, Y01, Y10, Y11]
        return root
    

# def test_matrix_mult_extended():
#     max_rank = 2
#     eps = 1e-8

#     print("=== Test 1: liście rank=0 ===")
#     v0 = Node(0, (2,2))
#     w0 = Node(0, (2,2))
#     Y0 = matrix_matrix_mult(v0, w0, max_rank, eps)
#     print("Y0 rank:", Y0.rank, "size:", Y0.size)
#     print("Odbudowana:\n", rebuild_matrix(Y0))

#     print("\n=== Test 2: liście rank>0 ===")
#     A = np.random.rand(2,2)
#     B = np.random.rand(2,2)
#     U1, D1, V1 = truncated_svd(A, max_rank, eps)
#     U2, D2, V2 = truncated_svd(B, max_rank, eps)
#     v1 = compress_matrix(A, U1, D1, V1, max_rank, eps)
#     w1 = compress_matrix(B, U2, D2, V2, max_rank, eps)
#     Y1 = matrix_matrix_mult(v1, w1, max_rank, eps)
#     print("Błąd:", np.linalg.norm(rebuild_matrix(Y1) - (A @ B)))

#     print("\n=== Test 3: liczby 1x1 ===")
#     A_num = np.array([[0.5]])
#     B_num = np.array([[0.2]])
#     U1, D1, V1 = truncated_svd(A_num, 1, eps)
#     U2, D2, V2 = truncated_svd(B_num, 1, eps)
#     v_num = compress_matrix(A_num, U1, D1, V1, 1, eps)
#     w_num = compress_matrix(B_num, U2, D2, V2, 1, eps)
#     Y_num = matrix_matrix_mult(v_num, w_num, 1, eps)
#     print("Odbudowane 1x1:", rebuild_matrix(Y_num))

#     print("\n=== Test 4: blok × blok 2x2 z dziećmi ===")
#     childA = []
#     childB = []
#     for i in range(2):
#         for j in range(2):
#             valA = np.array([[i+j]])
#             valB = np.array([[i*j]])
#             U,D,V = truncated_svd(valA, 1, eps)
#             if U.shape[1] == 0:
#                 U = np.array([[1.0]])
#                 D = np.array([valA[0,0]])
#                 V = np.array([[1.0]])
#             childA.append(compress_matrix(valA, U, D, V, 1, eps))
            
#             U,D,V = truncated_svd(valB, 1, eps)
#             if U.shape[1] == 0:
#                 U = np.array([[1.0]])
#                 D = np.array([valB[0,0]])
#                 V = np.array([[1.0]])
#             childB.append(compress_matrix(valB, U, D, V, 1, eps))

#     v_block = Node(0, (2,2))
#     v_block.children = childA
#     w_block = Node(0, (2,2))
#     w_block.children = childB

#     Y_block = matrix_matrix_mult(v_block, w_block, max_rank, eps)
#     print("Odbudowane blok × blok:\n", rebuild_matrix(Y_block))

# test_matrix_mult_extended()