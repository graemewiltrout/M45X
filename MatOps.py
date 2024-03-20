"""
Numerical Methods Package: Matrices
@author: Graeme Wiltrout
@advisor: T. Fogarty
"""

import numpy as np

def gaussian_elimination(n, aug_matrix):
    # Forward elimination
    for i in range(n):
        # Make sure the diagonal element is not zero
        if aug_matrix[i][i] == 0:
            raise ValueError(f"Zero diagonal element encountered at index {i}, naive Gaussian elimination cannot proceed without pivoting.")
        
        # Eliminate the entries below the i-th diagonal element
        for k in range(i+1, n):
            factor = aug_matrix[k][i] / aug_matrix[i][i]
            for j in range(i, n+1):
                aug_matrix[k][j] -= factor * aug_matrix[i][j]

    # Backward substitution
    solution = np.zeros(n)
    for i in range(n-1, -1, -1):
        solution[i] = aug_matrix[i][n]
        for j in range(i+1, n):
            solution[i] -= aug_matrix[i][j] * solution[j]
        solution[i] = solution[i] / aug_matrix[i][i]

    return solution

def gaussian_elimination_pp(n, aug_matrix):
    # Forward elimination with partial pivoting
    for i in range(n):
        # Find the maximum element for partial pivoting
        max_row = max(range(i, n), key=lambda r: abs(aug_matrix[r][i]))
        if aug_matrix[max_row][i] == 0:
            raise ValueError("Matrix is singular and cannot be solved by Gaussian elimination with partial pivoting.")

        # Swap the rows if necessary
        if max_row != i:
            aug_matrix[[i, max_row]] = aug_matrix[[max_row, i]]

        # Eliminate the entries below the i-th diagonal element
        for k in range(i+1, n):
            factor = aug_matrix[k][i] / aug_matrix[i][i]
            for j in range(i, n+1):
                aug_matrix[k][j] -= factor * aug_matrix[i][j]

    # Backward substitution
    solution = np.zeros(n)
    for i in range(n-1, -1, -1):
        solution[i] = aug_matrix[i][n]
        for j in range(i+1, n):
            solution[i] -= aug_matrix[i][j] * solution[j]
        solution[i] = solution[i] / aug_matrix[i][i]

    return solution

def gaussian_elimination_spp(n, aug_matrix):
    # Create scale factors
    scale_factors = np.max(np.abs(aug_matrix[:, :-1]), axis=1)

    for i in range(n):
        # Scaled partial pivoting
        max_index = i + np.argmax(np.abs(aug_matrix[i:n, i]) / scale_factors[i:n])
        if aug_matrix[max_index, i] == 0:
            raise ValueError("Matrix is singular and cannot be solved by Gaussian elimination with scaled partial pivoting.")

        # Swap the rows in both the matrix and the scale factors
        if max_index != i:
            aug_matrix[[i, max_index]] = aug_matrix[[max_index, i]]
            scale_factors[[i, max_index]] = scale_factors[[max_index, i]]

        # Eliminate the entries below the i-th diagonal element
        for k in range(i + 1, n):
            factor = aug_matrix[k][i] / aug_matrix[i][i]
            aug_matrix[k, i:] -= factor * aug_matrix[i, i:]

    # Backward substitution
    solution = np.zeros(n)
    for i in range(n - 1, -1, -1):
        solution[i] = (aug_matrix[i, -1] - np.dot(aug_matrix[i, i + 1:], solution[i + 1:])) / aug_matrix[i, i]

    return solution


def is_symmetric(A):
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != A[j, i]:
                return False
    return True

def is_positive_definite(A):
    n = A.shape[0]
    for i in range(1, n + 1):
        if np.linalg.det(A[:i, :i]) <= 0:
            return False
    return True


def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i+1):
            sum = 0
            if j == i:  # Diagonal elements
                for k in range(j):
                    sum += L[j, k] ** 2
                L[j, j] = np.sqrt(A[j, j] - sum)
            else:
                for k in range(j):
                    sum += L[i, k] * L[j, k]
                L[i, j] = (A[i, j] - sum) / L[j, j]

    return L

def cholesky_solve(L, b):
    n = L.shape[0]

    # Forward substitution to solve Ly = b
    y = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i, j] * y[j]
        y[i] = (b[i] - sum) / L[i, i]

    # Backward substitution to solve L^T x = y
    x = np.zeros_like(y, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        sum = 0
        for j in range(i + 1, n):
            sum += L[j, i] * x[j]
        x[i] = (y[i] - sum) / L[i, i]

    return x

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
                if i == j:
                    L[i, i] = 1
            else:
                L[i, j] = (A[i, j] - sum(L[i, k] * U[k, j] for k in range(j))) / U[j, j]
    return L, U

def forward_substitution(L, b):
    n = len(L)
    y = np.zeros(n)
    
    for i in range(n):
        y[i] = (b[i] - sum(L[i, k] * y[k] for k in range(i))) / L[i, i]
    return y

def backward_substitution(U, y):
    n = len(U)
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i, k] * x[k] for k in range(i+1, n))) / U[i, i]
    return x

def solve_system(A, b):
    if is_symmetric(A) and is_positive_definite(A):
        L = cholesky_decomposition(A)
        x = cholesky_solve(L, b)
        method_used = "Cholesky"
        return x, L, L.T, method_used
    else:
        L, U = lu_decomposition(A)
        y = forward_substitution(L, b)
        x = backward_substitution(U, y)
        method_used = "LU"
        return x, L, U, method_used
