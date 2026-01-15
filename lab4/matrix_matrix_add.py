from tree import Node, compress_matrix, truncated_svd, rebuild_matrix
import numpy as np

def matrix_matrix_add(v, w, max_rank, eps):
    if not v.children and not w.children:
        if v.rank == 0 and w.rank == 0:
            return Node(0, v.size)
        elif v.size == (1,1) and w.size == (1,1):
            val_v = v.U[0,0]*v.D[0]*v.V[0,0] if v.rank > 0 else 0
            val_w = w.U[0,0]*w.D[0]*w.V[0,0] if w.rank > 0 else 0
            val = val_v + val_w
            U = np.array([[val]])
            D = np.array([1.0])
            V = np.array([[1.0]])
            return Node(1, (1,1), U, D, V)
        elif v.rank != 0 and w.rank != 0:
            A = v.U @ np.diag(v.D) @ v.V
            B = w.U @ np.diag(w.D) @ w.V
            M = A + B
            U, D, V = truncated_svd(M, max_rank + 1, eps)
            return compress_matrix(M, U, D, V, max_rank, eps)
            
    elif v.children and w.children:
        Y = Node(0, v.size)
        for vc, wc in zip(v.children, w.children):
            Y.children.append(
                matrix_matrix_add(vc, wc, max_rank, eps)
            )
        return Y
    
    elif not v.children and w.children:
        mid_row = w.children[0].size[0]
        mid_col = w.children[0].size[1]

        U1 = v.U[:mid_row, :]
        U2 = v.U[mid_row:, :]
        V1 = v.V[:, :mid_col]
        V2 = v.V[:, mid_col:]

        Y = Node(0, v.size)
        Y.children = [
            matrix_matrix_add(Node(v.rank, U1.shape, U1, v.D, V1), w.children[0], max_rank, eps),
            matrix_matrix_add(Node(v.rank, U1.shape, U1, v.D, V2), w.children[1], max_rank, eps),
            matrix_matrix_add(Node(v.rank, U2.shape, U2, v.D, V1), w.children[2], max_rank, eps),
            matrix_matrix_add(Node(v.rank, U2.shape, U2, v.D, V2), w.children[3], max_rank, eps)
        ]
        return Y
    
    else:
        mid_row = v.children[0].size[0]
        mid_col = v.children[0].size[1]

        U1 = w.U[:mid_row, :]
        U2 = w.U[mid_row:, :]
        V1 = w.V[:, :mid_col]
        V2 = w.V[:, mid_col:]

        Y = Node(0, v.size)
        Y.children = [
            matrix_matrix_add(v.children[0], Node(w.rank, U1.shape, U1, w.D, V1), max_rank, eps),
            matrix_matrix_add(v.children[1], Node(w.rank, U1.shape, U1, w.D, V2), max_rank, eps),
            matrix_matrix_add(v.children[2], Node(w.rank, U2.shape, U2, w.D, V1), max_rank, eps),
            matrix_matrix_add(v.children[3], Node(w.rank, U2.shape, U2, w.D, V2), max_rank, eps)
        ]
        return Y
    

# def test_matrix_add_extended():
#     max_rank = 2
#     eps = 1e-8

#     print("=== Test 1: liście rank=0 ===")
#     v0 = Node(0, (2,2))
#     w0 = Node(0, (2,2))
#     Y0 = matrix_matrix_add(v0, w0, max_rank, eps)
#     print("Y0 rank:", Y0.rank, "size:", Y0.size)

#     print("\n=== Test 2: liście rank>0 ===")
#     A = np.random.rand(2,2)
#     B = np.random.rand(2,2)
#     U1, D1, V1 = truncated_svd(A, max_rank, eps)
#     U2, D2, V2 = truncated_svd(B, max_rank, eps)
#     v1 = compress_matrix(A, U1, D1, V1, max_rank, eps)
#     w1 = compress_matrix(B, U2, D2, V2, max_rank, eps)
#     Y1 = matrix_matrix_add(v1, w1, max_rank, eps)
#     print("Błąd:", np.linalg.norm(rebuild_matrix(Y1) - (A+B)))

#     print("\n=== Test 3: liczby 1x1 ===")
#     A_num = np.array([[0.5]])
#     B_num = np.array([[0.2]])
#     U1, D1, V1 = truncated_svd(A_num, 1, eps)
#     U2, D2, V2 = truncated_svd(B_num, 1, eps)
#     v_num = compress_matrix(A_num, U1, D1, V1, 1, eps)
#     w_num = compress_matrix(B_num, U2, D2, V2, 1, eps)
#     Y_num = matrix_matrix_add(v_num, w_num, 1, eps)
#     print("Odbudowane 1x1:", rebuild_matrix(Y_num))

#     print("\n=== Test 4: blok × blok 2x2 z dziećmi ===")
#     childA = []
#     for i in range(2):
#         for j in range(2):
#             val = np.array([[i+j]])
#             U, D, V = truncated_svd(val, 1, eps)
#             childA.append(compress_matrix(val, U, D, V, 1, eps))

#     v_block = Node(0, (2,2))
#     v_block.children = childA

#     childB = []
#     for i in range(2):
#         for j in range(2):
#             val = np.array([[i*j]])
#             U, D, V = truncated_svd(val, 1, eps)
#             childB.append(compress_matrix(val, U, D, V, 1, eps))

#     w_block = Node(0, (2,2))
#     w_block.children = childB

#     Y_block = matrix_matrix_add(v_block, w_block, max_rank, eps)
#     print("Odbudowane blok × blok:\n", rebuild_matrix(Y_block))

# test_matrix_add_extended()