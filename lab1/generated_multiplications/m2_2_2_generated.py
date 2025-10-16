# Matrix multiplication: C = A @ B
# A: 2×2, B: 2×2, C: 2×2
# Using 7 multiplications (Strassen-like algorithm)

from multiply import Matrix

def multiply(A, B):
	C = Matrix([[0 for _ in range(2)] for _ in range(2)])
	M0 = (A[1][0] + -A[1][1]) * B[0][1]
	M1 = (A[0][0] + A[1][0] + -A[1][1]) * (B[0][1] + B[1][0] + B[1][1])
	M2 = (A[0][0] + -A[0][1] + A[1][0] + -A[1][1]) * (B[1][0] + B[1][1])
	M3 = A[0][1] * B[1][0]
	M4 = (A[0][0] + A[1][0]) * (B[0][0] + B[0][1] + B[1][0] + B[1][1])
	M5 = A[0][0] * B[0][0]
	M6 = A[1][1] * (B[0][1] + B[1][1])

# Assemble result matrix
	C[0][0] = M3 + M5
	C[1][0] = -M1 + M4 + -M5 + -M6
	C[0][1] = -M0 + M1 + -M2 + -M3
	C[1][1] = M0 + M6
	return (C, 7)