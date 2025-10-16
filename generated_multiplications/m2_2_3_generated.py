# Matrix multiplication: C = A @ B
# A: 2×2, B: 2×3, C: 2×3
# Using 11 multiplications (Strassen-like algorithm)

from multiply import Matrix

def multiply(A, B):
	C = Matrix([[0 for _ in range(2)] for _ in range(3)])
	M0 = A[1][0] * (B[0][1] + -B[1][1])
	M1 = (A[1][0] + A[1][1]) * B[1][1]
	M2 = (A[0][1] + A[1][0]) * (B[0][2] + B[1][1])
	M3 = A[0][1] * (B[0][2] + -B[1][2])
	M4 = (A[0][0] + A[0][1]) * B[0][2]
	M5 = (A[0][0] + -A[1][0]) * (B[0][1] + B[0][2])
	M6 = A[0][0] * (B[0][0] + -B[0][2])
	M7 = (A[0][1] + -A[1][1]) * (B[1][1] + B[1][2])
	M8 = (A[1][0] + -A[1][1]) * B[0][0]
	M9 = A[0][1] * (B[0][2] + -B[1][0])
	M10 = A[1][1] * (B[0][0] + B[1][0])

# Assemble result matrix
	C[0][0] = M4 + M6 + -M9
	C[1][0] = M8 + M10
	C[2][0] = M0 + M2 + -M4 + M5
	C[0][1] = M0 + M1
	C[1][1] = -M3 + M4
	C[2][1] = -M1 + M2 + -M3 + -M7
	return (C, 11)