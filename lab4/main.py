from tree import *
from numpy.random import rand
from matrix_vector_mult import matrix_vector_mult
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

def demo():
    for k in [2,3,4]:
        M = generate_matrix(k)
        X = generate_vector(k)
        tree = create_tree(M, 4)
        draw(tree, M.shape)
        start = monotonic()
        N = matrix_vector_mult(tree, X)
        end = monotonic()
        N_true = M @ X
        mse = sum((N - N_true)**2)
        print(k, mse, end - start)

demo()