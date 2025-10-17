import numpy as np
from os import listdir

def generate_matmul_code(U, V, W, var_names=('A', 'B', 'C')):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    r = U.shape[1]
    m_k = U.shape[0]
    k_n = V.shape[0]
    m_n = W.shape[0]
    
    A_name, B_name, C_name = var_names
    
    code_lines = []
    code_lines.append(f"# Matrix multiplication using {r} intermediate values")
    code_lines.append(f"# Generated from tensor factorization\n")
    
    for i in range(r):
        u_terms = []
        v_terms = []
        
        for j in range(m_k):
            if U[j, i] != 0:
                m_idx = j // int(np.sqrt(m_k) + 0.5)
                k_idx = j % int(np.sqrt(m_k) + 0.5)
                coef = U[j, i]
                if coef == 1:
                    u_terms.append(f"{A_name}[{m_idx},{k_idx}]")
                elif coef == -1:
                    u_terms.append(f"-{A_name}[{m_idx},{k_idx}]")
                else:
                    u_terms.append(f"{coef}*{A_name}[{m_idx},{k_idx}]")
        
        for j in range(k_n):
            if V[j, i] != 0:
                k_idx = j // int(np.sqrt(k_n) + 0.5)
                n_idx = j % int(np.sqrt(k_n) + 0.5)
                coef = V[j, i]
                if coef == 1:
                    v_terms.append(f"{B_name}[{k_idx},{n_idx}]")
                elif coef == -1:
                    v_terms.append(f"-{B_name}[{k_idx},{n_idx}]")
                else:
                    v_terms.append(f"{coef}*{B_name}[{k_idx},{n_idx}]")
        
        left_expr = " + ".join(u_terms) if u_terms else "0"
        right_expr = " + ".join(v_terms) if v_terms else "0"
        
        if len(u_terms) > 1:
            left_expr = f"({left_expr})"
        if len(v_terms) > 1:
            right_expr = f"({right_expr})"
        
        code_lines.append(f"M{i} = {left_expr} * {right_expr}")
    
    code_lines.append("")
    
    code_lines.append(f"# Compute output matrix {C_name}")
    for i in range(m_n):
        m_idx = i // int(np.sqrt(m_n) + 0.5)
        n_idx = i % int(np.sqrt(m_n) + 0.5)
        
        terms = []
        for j in range(r):
            if W[i, j] != 0:
                coef = W[i, j]
                if coef == 1:
                    terms.append(f"M{j}")
                elif coef == -1:
                    terms.append(f"-M{j}")
                else:
                    terms.append(f"{coef}*M{j}")
        
        if terms:
            expr = " + ".join(terms)
            code_lines.append(f"{C_name}[{m_idx},{n_idx}] = {expr}")
    
    return "\n".join(code_lines)


def generate_matmul_code_explicit(U, V, W, m, k, n, var_names=('A', 'B', 'C')):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    r = U.shape[1]
    A_name, B_name, C_name = var_names
    
    code_lines = []
    code_lines.append(f"# Matrix multiplication: {C_name} = {A_name} @ {B_name}")
    code_lines.append(f"# {A_name}: {m}×{k}, {B_name}: {k}×{n}, {C_name}: {m}×{n}")
    code_lines.append(f"# Using {r} multiplications (Strassen-like algorithm)\n")
    code_lines.append(f"from multiply import Matrix\n")
    code_lines.append("def multiply(A, B):")
    code_lines.append(f"\tC = Matrix([[0 for _ in range({m})] for _ in range({n})])")
    
    for i in range(r):
        u_terms = []
        v_terms = []
        
        for row in range(m):
            for col in range(k):
                idx = row * k + col
                if U[idx, i] != 0:
                    coef = U[idx, i]
                    if coef == 1:
                        u_terms.append(f"{A_name}[{row}][{col}]")
                    elif coef == -1:
                        u_terms.append(f"-{A_name}[{row}][{col}]")
                    else:
                        u_terms.append(f"{coef}*{A_name}[{row}][{col}]")
        
        for row in range(k):
            for col in range(n):
                idx = row * n + col
                if V[idx, i] != 0:
                    coef = V[idx, i]
                    if coef == 1:
                        v_terms.append(f"{B_name}[{row}][{col}]")
                    elif coef == -1:
                        v_terms.append(f"-{B_name}[{row}][{col}]")
                    else:
                        v_terms.append(f"{coef}*{B_name}[{row}][{col}]")
        
        left_expr = " + ".join(u_terms) if u_terms else "0"
        right_expr = " + ".join(v_terms) if v_terms else "0"
        
        if len(u_terms) > 1:
            left_expr = f"({left_expr})"
        if len(v_terms) > 1:
            right_expr = f"({right_expr})"
        
        code_lines.append(f"\tM{i} = {left_expr} * {right_expr}")
    
    code_lines.append("")
    
    code_lines.append(f"# Assemble result matrix")
    for row in range(m):
        for col in range(n):
            idx = row * n + col
            terms = []
            
            for j in range(r):
                if W[idx, j] != 0:
                    coef = W[idx, j]
                    if coef == 1:
                        terms.append(f"M{j}")
                    elif coef == -1:
                        terms.append(f"-M{j}")
                    else:
                        terms.append(f"{coef}*M{j}")
            
            if terms:
                expr = " + ".join(terms)
                code_lines.append(f"\t{C_name}[{col}][{row}] = {expr}")
    
    code_lines.append(f"\treturn (C, {r})")
    return "\n".join(code_lines)


def load_tensor_factorization(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data[0], data[1], data[2]


def infer_dimensions(U, V, W):
    mk = U.shape[0]
    kn = V.shape[0]
    mn = W.shape[0]
    r = U.shape[1]
    
    for m in range(1, 20):
        for k in range(1, 20):
            if m * k == mk:
                n = kn // k
                if k * n == kn and m * n == mn:
                    return m, k, n
    
    m = k = n = int(round(mk ** (1/2)))
    if m * k == mk and k * n == kn and m * n == mn:
        return m, k, n
    
    raise ValueError(f"Cannot infer dimensions from shapes: U={U.shape}, V={V.shape}, W={W.shape}")


for file in listdir("/home/hszymon/Dokumenty/agh/am/npy/"):
    import sys
    
    tensor_file = "/home/hszymon/Dokumenty/agh/am/npy/" + file
    
    if len(sys.argv) > 1:
        tensor_file = sys.argv[1]
    
    try:
        print(f"Loading tensor factorization from: {tensor_file}")
        print("="*50)
        
        U, V, W = load_tensor_factorization(tensor_file)
        
        print(f"Loaded factorization:")
        print(f"  U shape: {U.shape}")
        print(f"  V shape: {V.shape}")
        print(f"  W shape: {W.shape}")
        print(f"  Rank (# multiplications): {U.shape[1]}")
        print()
        
        m, k, n = infer_dimensions(U, V, W)
        print(f"Inferred dimensions: A is {m}×{k}, B is {k}×{n}, C is {m}×{n}")
        print("="*50)
        print()
        
        code = generate_matmul_code_explicit(U, V, W, m, k, n)
        print(code)
        
        output_file = tensor_file.replace('.npy', '_generated.py')
        with open(output_file, 'w') as f:
            f.write(code)
        print(f"\n{'='*50}")
        print(f"Code saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{tensor_file}' not found")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} [tensor_file.npy]")
        print("\nExpected tensor format:")
        print("  - Dictionary with keys 'U', 'V', 'W'")
        print("  - 3D array of shape (3, max_dim, r)")
        print("  - Tuple/list of three arrays [U, V, W]")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()