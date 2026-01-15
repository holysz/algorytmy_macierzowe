import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2hsv, hsv2rgb

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
    def __init__(self, rank, size, U=None, D=None, V=None):
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
            matrix[x_bounds[0] : x_bounds[1], y_bounds[0] : (y_bounds[0] + self.rank*2)] = 0
            matrix[x_bounds[0] : (x_bounds[0] + self.rank*2), y_bounds[0] :y_bounds[1]] = 0
            return
        
        x_mid = (x_bounds[0] + x_bounds[1]) // 2
        y_mid = (y_bounds[0] + y_bounds[1]) // 2
        self.children[0].draw_compression(matrix, (x_bounds[0], x_mid), (y_bounds[0], y_mid))
        self.children[1].draw_compression(matrix, (x_bounds[0], x_mid), (y_mid, y_bounds[1]))
        self.children[2].draw_compression(matrix, (x_mid, x_bounds[1]), (y_bounds[0], y_mid))
        self.children[3].draw_compression(matrix, (x_mid, x_bounds[1]), (y_mid, y_bounds[1]))


def compress_matrix(A, U, D, V, r, eps):
    significant = D > eps
    rank = min(r, significant.sum())

    U_c = U[:, :rank]
    D_c = D[:rank]
    V_c = V[:rank, :]

    return Node(rank, A.shape, U_c, D_c, V_c)


def rebuild_matrix(node):
    if not node.children:
        if node.rank == 0 or node.D.size == 0:
            return np.zeros(node.size)
        return node.U @ np.diag(node.D) @ node.V
    
    return np.vstack((np.hstack((rebuild_matrix(node.children[0]), rebuild_matrix(node.children[1]))),
                      np.hstack((rebuild_matrix(node.children[2]), rebuild_matrix(node.children[3])))))

def create_tree(A, rank, eps=1e-10, min_size=2):
    U, D, V = truncated_svd(A, rank + 1, eps)

    if min(A.shape) <= min_size or D[rank] <= eps:
        root = compress_matrix(A, U, D, V, rank, eps)
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

def test_rgb(img, max_rank, epsilon, singular=False):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    plt.suptitle(rf"Kompresja kanałów RGB, $max\_rank: {max_rank}$, $\epsilon: {epsilon}$")
    plt.axis('off')
    plt.subplots_adjust(wspace=0.3)

    plt.subplot(3, 4, 1).imshow(img)
    plt.title("Oryginał")
    plt.axis('off')
    plt.subplot(3, 4, 2).imshow(r, cmap='Reds_r')
    plt.title("R")
    plt.axis('off')
    plt.subplot(3, 4, 3).imshow(g, cmap='Greens_r')
    plt.title("G")
    plt.axis('off')
    plt.subplot(3, 4, 4).imshow(b, cmap='Blues_r')
    plt.title("B")
    plt.axis('off')

    tree = create_tree(r, max_rank, eps=epsilon)
    r_res = rebuild_matrix(tree)
    r_mat = np.ones(r.shape)
    tree.draw_compression(r_mat)
    plt.subplot(3, 4, 10).imshow(r_mat)
    plt.title("Macierz kompresji R")
    plt.axis('off')
    plt.subplot(3, 4, 6).imshow(r_res, cmap='Reds_r')
    plt.title("R")
    plt.axis('off')

    tree = create_tree(g, max_rank, eps=epsilon)
    g_res = rebuild_matrix(tree)
    g_mat = np.ones(g.shape)
    tree.draw_compression(g_mat)
    plt.subplot(3, 4, 11).imshow(g_mat)
    plt.title("Macierz kompresji G")
    plt.axis('off')
    plt.subplot(3, 4, 7).imshow(g_res, cmap='Greens_r')
    plt.title("G")
    plt.axis('off')

    tree = create_tree(b, max_rank, eps=epsilon)
    b_res = rebuild_matrix(tree)
    b_mat = np.ones(b.shape)
    tree.draw_compression(b_mat)
    plt.subplot(3, 4, 12).imshow(b_mat)
    plt.title("Macierz kompresji B")
    plt.axis('off')
    plt.subplot(3, 4, 8).imshow(b_res, cmap='Blues_r')
    plt.title("B")
    plt.axis('off')
    
    img_res = np.stack((r_res.clip(0,1), g_res.clip(0,1), b_res.clip(0,1)), axis=-1)
    plt.subplot(3, 4, 5).imshow(img_res)
    plt.title("Skompresowany obraz")
    plt.axis('off')
    plt.show()

    if singular:
        _, rS, _ = truncated_svd(r, max_rank, epsilon)
        _, rG, _ = truncated_svd(g, max_rank, epsilon)
        _, rB, _ = truncated_svd(b, max_rank, epsilon)  
        plt.figure(figsize=(12, 3))  
        plt.subplot(1,3,1).bar(range(1, len(rS)+1), rS, color='red')
        plt.title("Wartości osobliwe kanału R")
        plt.xlabel("Indeks $k$ wartości osobliwej")
        plt.ylabel(r"$\sigma_k$")
        plt.subplot(1,3,2).bar(range(1, len(rG)+1), rG, color='green')
        plt.title("Wartości osobliwe kanału G")
        plt.xlabel("Indeks $k$ wartości osobliwej")
        plt.ylabel(r"$\sigma_k$")
        plt.subplot(1,3,3).bar(range(1, len(rB)+1), rB, color='blue')
        plt.title("Wartości osobliwe kanału B")
        plt.xlabel("Indeks $k$ wartości osobliwej")
        plt.ylabel(r"$\sigma_k$")
        plt.tight_layout()
        plt.show()

    


def test_hsv(img, max_rank, epsilon, singular=False):
    hsv = rgb2hsv(img)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    plt.suptitle(rf"Kompresja kanałów HSV, $max\_rank: {max_rank}$, $\epsilon: {epsilon}$")
    plt.axis('off')
    plt.subplots_adjust(wspace=0.3)

    plt.subplot(3, 4, 1).imshow(img)
    plt.title("Oryginał")
    plt.axis('off')
    plt.subplot(3, 4, 2).imshow(h, cmap='hsv', vmin=0, vmax=1)
    plt.title("H")
    plt.axis('off')
    plt.subplot(3, 4, 3).imshow(s, cmap='gray', vmin=0, vmax=1)
    plt.title("S")
    plt.axis('off')
    plt.subplot(3, 4, 4).imshow(v, cmap='gray', vmin=0, vmax=1)
    plt.title("V")
    plt.axis('off')

    tree = create_tree(h, max_rank, eps=epsilon)
    h_res = rebuild_matrix(tree)
    h_mat = np.ones(h.shape)
    tree.draw_compression(h_mat)
    plt.subplot(3, 4, 10).imshow(h_mat)
    plt.title("Macierz kompresji H")
    plt.axis('off')
    plt.subplot(3, 4, 6).imshow(h_res, cmap='hsv', vmin=0, vmax=1)
    plt.title("H")
    plt.axis('off')

    tree = create_tree(s, max_rank, eps=epsilon)
    s_res = rebuild_matrix(tree)
    s_mat = np.ones(s.shape)
    tree.draw_compression(s_mat)
    plt.subplot(3, 4, 11).imshow(s_mat)
    plt.title("Macierz kompresji S")
    plt.axis('off')
    plt.subplot(3, 4, 7).imshow(s_res, cmap='gray', vmin=0, vmax=1)
    plt.title("S")
    plt.axis('off')

    tree = create_tree(v, max_rank, eps=epsilon)
    v_res = rebuild_matrix(tree)
    v_mat = np.ones(v.shape)
    tree.draw_compression(v_mat)
    plt.subplot(3, 4, 12).imshow(v_mat)
    plt.title("Macierz kompresji V")
    plt.axis('off')
    plt.subplot(3, 4, 8).imshow(v_res, cmap='gray', vmin=0, vmax=1)
    plt.title("V")
    plt.axis('off')

    img_res = np.stack((h_res.clip(0,1), s_res.clip(0,1), v_res.clip(0,1)), axis=-1)
    img_res = hsv2rgb(img_res)
    plt.subplot(3, 4, 5).imshow(img_res, vmin=0, vmax=1)
    plt.title("Skompresowany obraz")
    plt.axis('off')
    plt.show()
    
    if singular:
        _, rH, _ = truncated_svd(h, max_rank, epsilon)
        _, rS, _ = truncated_svd(s, max_rank, epsilon)
        _, rV, _ = truncated_svd(v, max_rank, epsilon)
        plt.figure(figsize=(12, 3)) 
        plt.subplot(1,3,1).bar(range(1, len(rH)+1), rH)
        plt.title("Wartości osobliwe kanału H")
        plt.xlabel("Indeks $k$ wartości osobliwej")
        plt.ylabel(r"$\sigma_k$")
        plt.subplot(1,3,2).bar(range(1, len(rS)+1), rS)
        plt.title("Wartości osobliwe kanału S")
        plt.xlabel("Indeks $k$ wartości osobliwej")
        plt.ylabel(r"$\sigma_k$")
        plt.subplot(1,3,3).bar(range(1, len(rV)+1), rV)
        plt.title("Wartości osobliwe kanału R")
        plt.xlabel("Indeks $k$ wartości osobliwej")
        plt.ylabel(r"$\sigma_k$")
        plt.tight_layout()
        plt.show()
    

# img = Image.open("lab3/coffee.png")
# img = img.convert('RGB')

<<<<<<< HEAD
coffee = np.asarray(img) / 255
#test_rgb(coffee, 4, 2)

test_hsv(coffee, 5, 2)
=======
# coffee = np.asarray(img) / 255
# test_rgb(coffee, 4, 0.05)
>>>>>>> ff802cb23a4b309a2a782442a301bb2cf2c8a728
