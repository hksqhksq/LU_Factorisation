import numpy as np
from scipy.sparse import diags

def gram_schmidt_qr(A):
    """
    Compute the QR factorisation of a square matrix using the classical
    Gram-Schmidt process.

    Parameters
    ----------
    A : numpy.ndarray
        A square 2D NumPy array of shape ``(n, n)`` representing the input
        matrix.

    Returns
    -------
    Q : numpy.ndarray
        Orthonormal matrix of shape ``(n, n)`` where the columns form an
        orthonormal basis for the column space of A.
    R : numpy.ndarray
        Upper triangular matrix of shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"the matrix A is not square, {A.shape=}")

    Q = np.empty_like(A)
    R = np.zeros_like(A)

    for j in range(n):
        # Start with the j-th column of A
        u = A[:, j].copy()

        # Orthogonalize against previous q vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])  # projection coefficient
            u -= R[i, j] * Q[:, i]  # subtract the projection

        # Normalize u to get q_j
        R[j, j] = np.linalg.norm(u)
        Q[:, j] = u / R[j, j]

    return Q, R

epsilons = [10**(-k) for k in range(6, 17)]

print("eps", "error1", "error2", "error3")
for eps in epsilons:
    A = A_eps = np.array([[1, 1+eps], [1+eps, 1]], dtype=float) # your code here
    Q, R = gram_schmidt_qr(A)

    error1 = np.linalg.norm(A - Q @ R, 2)
    error2 = np.linalg.norm(Q.T @ Q - np.eye(2), 2)
    error3 = np.linalg.norm(R - np.triu(R), 2)

    print(eps, error1, error2, error3)

print(A)
print("Q =", Q)
print("R =", R)