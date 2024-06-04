"""
Numerical Methods Package: Linear Algebra
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""
import math
from Complex import Complex

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))
'''
def norm(v):
    return math.sqrt(sum(x.real**2 + x.imag**2 for x in v))
'''
def norm(v):
    return math.sqrt(sum(x * x for x in v))

def gaussian_elimination(n, aug_matrix):
    for i in range(n):
        if aug_matrix[i][i] == 0:
            raise ValueError(f"Zero diagonal element encountered at index {i}, naive Gaussian elimination cannot proceed without pivoting.")
        
        for k in range(i+1, n):
            factor = aug_matrix[k][i] / aug_matrix[i][i]
            for j in range(i, n+1):
                aug_matrix[k][j] -= factor * aug_matrix[i][j]

    solution = [0] * n
    for i in range(n-1, -1, -1):
        solution[i] = aug_matrix[i][n]
        for j in range(i+1, n):
            solution[i] -= aug_matrix[i][j] * solution[j]
        solution[i] = solution[i] / aug_matrix[i][i]

    return solution

def gaussian_elimination_pp(n, aug_matrix):
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(aug_matrix[r][i]))
        if aug_matrix[max_row][i] == 0:
            raise ValueError("Matrix is singular and cannot be solved by Gaussian elimination with partial pivoting.")

        if max_row != i:
            aug_matrix[i], aug_matrix[max_row] = aug_matrix[max_row], aug_matrix[i]

        for k in range(i+1, n):
            factor = aug_matrix[k][i] / aug_matrix[i][i]
            for j in range(i, n+1):
                aug_matrix[k][j] -= factor * aug_matrix[i][j]

    solution = [0] * n
    for i in range(n-1, -1, -1):
        solution[i] = aug_matrix[i][n]
        for j in range(i+1, n):
            solution[i] -= aug_matrix[i][j] * solution[j]
        solution[i] = solution[i] / aug_matrix[i][i]

    return solution

def gaussian_elimination_spp(n, aug_matrix):
    scale_factors = [max(abs(x) for x in row[:-1]) for row in aug_matrix]

    for i in range(n):
        max_index = i + max(range(n - i), key=lambda k: abs(aug_matrix[i + k][i]) / scale_factors[i + k])
        if aug_matrix[max_index][i] == 0:
            raise ValueError("Matrix is singular and cannot be solved by Gaussian elimination with scaled partial pivoting.")

        if max_index != i:
            aug_matrix[i], aug_matrix[max_index] = aug_matrix[max_index], aug_matrix[i]
            scale_factors[i], scale_factors[max_index] = scale_factors[max_index], scale_factors[i]

        for k in range(i + 1, n):
            factor = aug_matrix[k][i] / aug_matrix[i][i]
            for j in range(i, n+1):
                aug_matrix[k][j] -= factor * aug_matrix[i][j]

    solution = [0] * n
    for i in range(n - 1, -1, -1):
        solution[i] = aug_matrix[i][n]
        for j in range(i + 1, n):
            solution[i] -= aug_matrix[i][j] * solution[j]
        solution[i] /= aug_matrix[i][i]

    return solution

def is_symmetric(A):
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] != A[j][i]:
                return False
    return True

def is_positive_definite(A):
    n = len(A)
    for i in range(1, n + 1):
        sub_matrix = [row[:i] for row in A[:i]]
        if determinant(sub_matrix) <= 0:
            return False
    return True

def determinant(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    elif n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0
        for c in range(n):
            sub_matrix = [[matrix[r][col] for col in range(n) if col != c] for r in range(1, n)]
            det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix)
        return det

def cholesky_decomposition(A):
    n = len(A)
    L = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            csum = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(A[i][i] - csum)
            else:
                L[i][j] = (A[i][j] - csum) / L[j][j]

    return L

def cholesky_solve(L, b):
    n = len(L)
    y = [0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / L[i][i]

    return x

def lu_decomposition(A):
    n = len(A)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
                if i == j:
                    L[i][i] = 1
            else:
                L[i][j] = (A[i][j] - sum(L[i][k] * U[k][j] for k in range(j))) / U[j][j]
    return L, U

def forward_substitution(L, b):
    n = len(L)
    y = [0] * n
    
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    return y

def backward_substitution(U, y):
    n = len(U)
    x = [0] * n
    
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][k] * x[k] for k in range(i+1, n))) / U[i][i]
    return x

def solve_system(A, b):
    if is_symmetric(A) and is_positive_definite(A):
        L = cholesky_decomposition(A)
        x = cholesky_solve(L, b)
        method_used = "Cholesky"
        return x, L, [[L[j][i] for j in range(len(L))] for i in range(len(L[0]))], method_used
    else:
        L, U = lu_decomposition(A)
        y = forward_substitution(L, b)
        x = backward_substitution(U, y)
        method_used = "LU"
        return x, L, U, method_used
'''
def power_method(A, num_iter=1000, tol=1e-10):
    n = len(A)
    b_k = [Complex(1, 0)] * n
    
    for _ in range(num_iter):
        b_k1 = [sum(A[i][j] * b_k[j] for j in range(n)) for i in range(n)]
        b_k1_norm = norm(b_k1)
        b_k = [Complex(x.real / b_k1_norm, x.imag / b_k1_norm) for x in b_k1]
        
        if norm([sum(A[i][j] * b_k[j] for j in range(n)) - Complex(b_k1_norm * b_k[i].real, b_k1_norm * b_k[i].imag) for i in range(n)]) < tol:
            break
    
    eigenvalue = b_k1_norm
    eigenvector = b_k
    
    return eigenvalue, eigenvector

'''

def power_method(A, num_iter=1000, tol=1e-10):
    n = len(A)
    b_k = [1] * n
    
    for _ in range(num_iter):
        b_k1 = [sum(A[i][j] * b_k[j] for j in range(n)) for i in range(n)]
        b_k1_norm = norm(b_k1)
        b_k = [x / b_k1_norm for x in b_k1]
        
        if norm([sum(A[i][j] * b_k[j] for j in range(n)) - b_k1_norm * b_k[i] for i in range(n)]) < tol:
            break
    
    eigenvalue = b_k1_norm
    eigenvector = b_k
    
    return eigenvalue, eigenvector
