from tree import *
from numpy.random import rand
from matrix_vector_mult import matrix_vector_mult
from matrix_matrix_mult import matrix_matrix_mult
from time import monotonic

def draw(tree, shape):
    m = np.ones(shape)
    tree.draw_compression(m)
    plt.subplot().imshow(m)
    plt.show()

def generate_matrix(k):
    return rand(2**(3*k), 2**(3*k))

def generate_vector(k):
    return rand(2**(3*k))

def demo(eps):
    for k in [2,3,4]:
        M = generate_matrix(k)
        X = generate_vector(k)
        tree = create_tree(M, 4)
        draw(tree, M.shape)
        start_vec = monotonic()
        result_vec = matrix_vector_mult(tree, X)
        end_vec = monotonic()
        true_vec = M @ X
        mse_vec = sum((result_vec - true_vec)**2)
        time_vec = end_vec - start_vec

        start_mat = monotonic()
        result_mat = matrix_matrix_mult(tree, tree, 4, eps)
        end_mat = monotonic()
        true_mat = M @ M
        mse_mat = sum((result_mat - true_mat)**2)
        time_mat = end_mat - start_mat

        print(k, mse_vec, time_vec, mse_mat, time_mat)

demo(0.05)