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
