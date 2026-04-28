import numpy as np
from scipy.sparse import diags
import time
import matplotlib.pyplot as plt

def generate_safe_system(n):
    """
    Generate a linear system A x = b where A is strictly diagonally dominant,
    ensuring LU factorization without pivoting will work.

    Parameters:
        n (int): Size of the system (n x n)

    Returns:
        A (ndarray): n x n strictly diagonally dominant matrix
        b (ndarray): RHS vector
        x_true (ndarray): The true solution vector
    """

    k = [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)]
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()

    # Solution is always all ones
    x_true = np.ones((n, 1))

    # Compute b = A @ x_true
    b = A @ x_true

    return A, b, x_true

def lu_factorisation(A):
    """
    Compute the LU factorisation of a square matrix A.

    The function decomposes a square matrix ``A`` into the product of a lower
    triangular matrix ``L`` and an upper triangular matrix ``U`` such that:

    .. math::
        A = L U

    where ``L`` has unit diagonal elements and ``U`` is upper triangular.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the square matrix to
        factorise.

    Returns
    -------
    L : numpy.ndarray
        A lower triangular matrix with shape ``(n, n)`` and unit diagonal.
    U : numpy.ndarray
        An upper triangular matrix with shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")

    # construct arrays of zeros
    L, U = np.zeros_like(A), np.zeros_like(A)


    # imagine 3 x 3 matrix
    # 1 4 6
    # 2 3 1
    # 5 3 7
    #

    for d in range(n):
        L[d, d] = 1

    # for j columns, i rows
    for j in range(n): # columns
        for i in range(j+1): # L
            sum_val = 0
            for v in range(j):
                sum_val = sum_val + (L[i, v] * U[v, j])
            U[i, j] = A[i, j] - sum_val
            #


        for i in range(j+1, n): # U
            sum_val = 0
            for v in range(i):
                sum_val = sum_val + (L[i, v] * U[v, j])
            L[i, j] = (A[i, j] - sum_val) / U[j, j]
    
    return L, U

def determinant(A):
    n = A.shape[0]
    L, U = lu_factorisation(A)

    det_L = 1.0
    det_U = 1.0

    for i in range(n):
        det_L *= L[i, i]
        det_U *= U[i, i]

    return det_L * det_U

def system_size(A, b):
    if A.ndim != 2:
        raise ValueError(f"Matrix A must be 2D, but got {A.ndim}D array")
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A must be square, but got A.shape={A.shape}")
    if b.shape[0] != n:
        raise ValueError(f"System shapes are not compatible: A.shape={A.shape}, b.shape={b.shape}")
    return n

def row_add(A, b, p, k, q):
    n = system_size(A, b)
    for j in range(n):
        A[p, j] = A[p, j] + k * A[q, j]
    b[p, 0] = b[p, 0] + k * b[q, 0]

def gaussian_elimination(A, b):
    n = system_size(A, b)
    for i in range(n - 1):
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            row_add(A, b, j, -factor, i)


if __name__ == "__main__":
    # --- Test 1: LU factorisation on a known matrix ---
    A_test = np.array([[2., 1., 1.],
                       [4., 3., 3.],
                       [8., 7., 9.]])

    L, U = lu_factorisation(A_test)
    print("=== Test 1: LU Factorisation ===")
    print("L =\n", L)
    print("U =\n", U)
    print("L @ U == A:", np.allclose(L @ U, A_test))

    # --- Test 2: L is lower triangular with unit diagonal ---
    print("\n=== Test 2: Structure checks ===")
    print("L is lower triangular:", np.allclose(L, np.tril(L)))
    print("U is upper triangular:", np.allclose(U, np.triu(U)))
    print("L has unit diagonal:  ", np.allclose(np.diag(L), 1.0))

    # --- Test 3: Determinant on A_large (n=100) ---
    print("\n=== Test 3: Determinant of A_large ===")
    A_large, b_large, x_large = generate_safe_system(100)
    det = determinant(A_large)
    print(f"determinant(A_large) = {det:.6f}")
    print(f"Expected (n+1 = 101): {101.0:.6f}")
    print(f"Correct: {np.isclose(det, 101.0)}")

    sizes = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    lu_times, ge_times = [], []

    for n in sizes:
        A, b, _ = generate_safe_system(n)

        t0 = time.perf_counter()
        lu_factorisation(A.copy())
        lu_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        gaussian_elimination(A.copy(), b.copy())
        ge_times.append(time.perf_counter() - t0)

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, lu_times, 'o-', color='steelblue', label='LU Factorisation (my code)', linewidth=2, markersize=5)
    plt.plot(sizes, ge_times, 's--', color='tomato', label='Gaussian Elimination (notes)', linewidth=2, markersize=5)
    plt.xlabel('Matrix size $n$')
    plt.ylabel('Run time (seconds)')
    plt.title('Run time: LU Factorisation vs Gaussian Elimination')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('runtimes.png')
    print("Plot saved as runtimes.png")
