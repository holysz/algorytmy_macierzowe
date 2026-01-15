from tree import *

def matrix_vector_mult(node: Node, X: np.ndarray):
    if not node.children:
        if node.rank == 0:
            return np.zeros((X.size, X.size))
        result = node.U @ np.diag(node.D) @ (node.V @ X)
        return result
    
    Xs = np.split(X, 2)
    Y = [matrix_vector_mult(node.children[i], Xs[i%2]) for i in range(4)]
    return np.hstack((Y[0] + Y[1], Y[2] + Y[3]))