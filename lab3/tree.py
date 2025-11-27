import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def power_iteration(M, max_iter=1000, eps=1e-8):
    v = np.random.rand(M.shape[1])
    v /= np.linalg.norm(v)
    prev = np.empty(M.shape[1])
    for _ in range(max_iter):
        prev[:] = v
        v = np.dot(M, v)
        norm = np.linalg.norm(v)
        if norm < eps:
            return v, 0
        v /= norm
        if np.allclose(v, prev, atol=eps):
            break
    eigval = np.dot(v, np.dot(M, v)) / np.dot(v, v)
    return v, eigval

def truncated_svd(M, r, eps):
    M = M.astype(np.float64)
    U = np.zeros([M.shape[0], r])
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
        U[:, rank] = M @ eigvec / singular

        A -= eigval * np.outer(eigvec, eigvec)

    return U, D, V.T


class Node:
    def __init__(self, rank, size, U=None, V=None, D=None):
        self.rank = rank
        self.size = size
        self.U = U
        self.V = V
        self.D = D
        self.children = []
    
    def draw_compression(self, matrix, x_bounds=None, y_bounds=None):
        if x_bounds is None or y_bounds is None:
            x_bounds=(0, self.size[0])
            y_bounds=(0, self.size[1])
        if len(self.children) == 0:
            matrix[x_bounds[0] : x_bounds[1], y_bounds[0] : (y_bounds[0] + self.rank)] = 0
            matrix[x_bounds[0] : (x_bounds[0] + self.rank), y_bounds[0] :y_bounds[1]] = 0
            return
        
        x_mid = (x_bounds[0] + x_bounds[1]) // 2
        y_mid = (y_bounds[0] + y_bounds[1]) // 2
        self.children[0].draw_compression(matrix, (x_bounds[0], x_mid), (y_bounds[0], y_mid))
        self.children[1].draw_compression(matrix, (x_bounds[0], x_mid), (y_mid, y_bounds[1]))
        self.children[2].draw_compression(matrix, (x_mid, x_bounds[1]), (y_bounds[0], y_mid))
        self.children[3].draw_compression(matrix, (x_mid, x_bounds[1]), (y_mid, y_bounds[1]))


def CompressMatrix(A, U, D, V, r, eps):
    significant = D > eps
    rank = min(r, significant.sum())

    U_c = U[:, :rank]
    D_c = D[:rank]
    V_c = V[:rank, :]

    return Node(rank, A.shape, U=U_c, V=V_c, D=D_c)


def rebuild_matrix(node):
    if not node.children:
        return node.U @ np.diag(node.D) @ node.V
    
    return np.vstack((np.hstack((rebuild_matrix(node.children[0]), rebuild_matrix(node.children[1]))),
                      np.hstack((rebuild_matrix(node.children[2]), rebuild_matrix(node.children[3])))))

def create_tree(A, rank, eps=1e-10, min_size=2):
    U, D, V = truncated_svd(A, rank + 1, eps)

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

def test(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    plt.subplot(4, 4, 1).imshow(img)
    plt.subplot(4, 4, 2).imshow(r, cmap='Reds')
    plt.subplot(4, 4, 3).imshow(g, cmap='Greens')
    plt.subplot(4, 4, 4).imshow(b, cmap='Blues')
    max_rank = 100
    epsilon = 1

    tree = create_tree(r, 1, eps=epsilon)
    r_res = rebuild_matrix(tree)
    r_mat = np.ones(r.shape)
    tree.draw_compression(r_mat)
    plt.subplot(4, 4, 10).imshow(r_mat)
    plt.subplot(4, 4, 6).imshow(r_res, cmap='Reds')
    tree = create_tree(g, 1, eps=epsilon)
    g_res = rebuild_matrix(tree)
    g_mat = np.ones(g.shape)
    tree.draw_compression(g_mat)
    plt.subplot(4, 4, 11).imshow(g_mat)
    plt.subplot(4, 4, 7).imshow(g_res, cmap='Greens')
    tree = create_tree(b, 1, eps=epsilon)
    b_res = rebuild_matrix(tree)
    b_mat = np.ones(b.shape)
    tree.draw_compression(b_mat)
    plt.subplot(4, 4, 12).imshow(b_mat)
    plt.subplot(4, 4, 8).imshow(b_res, cmap='Blues')
    img_res = np.stack((r_res.clip(0,1), g_res.clip(0,1), b_res.clip(0,1)), axis=-1)
    plt.subplot(4, 4, 5).imshow(img_res)

    _, rS, _ = truncated_svd(r, max_rank, epsilon)
    _, rG, _ = truncated_svd(g, max_rank, epsilon)
    _, rB, _ = truncated_svd(b, max_rank, epsilon)    
    plt.subplot(4,4,14).bar(range(1, len(rS)+1), rS, color='red')
    plt.subplot(4,4,15).bar(range(1, len(rG)+1), rG, color='green')
    plt.subplot(4,4,16).bar(range(1, len(rB)+1), rB, color='blue')

    plt.show()

    print(r_res)
    print(img_res)

img = Image.open("lab3/coffee.png")
img = img.convert('RGB')

coffee = np.asarray(img) / 255
test(coffee)