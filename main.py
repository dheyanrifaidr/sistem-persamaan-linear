import numpy as np
from scipy.linalg import lu

# Metode 1: Matriks Balikan
def inverse_matrix_method(A, B):
    """
    Menyelesaikan sistem persamaan linear menggunakan metode invers matriks.
    """
    try:
        A_inv = np.linalg.inv(A)  # Invers matriks A
        X = np.dot(A_inv, B)      # X = A^(-1) * B
        return X
    except np.linalg.LinAlgError:
        print("Matriks A tidak memiliki invers!")
        return None


# Metode 2: Dekomposisi LU Gauss
def lu_decomposition_gauss(A, B):
    """
    Menyelesaikan sistem persamaan linear menggunakan dekomposisi LU (Gauss).
    """
    # Lakukan dekomposisi LU
    P, L, U = lu(A)

    # Substitusi maju: Ly = Pb
    B_permuted = np.dot(P, B)
    y = np.linalg.solve(L, B_permuted)

    # Substitusi mundur: Ux = y
    X = np.linalg.solve(U, y)
    return X


# Metode 3: Dekomposisi Crout
def crout_decomposition(A, B):
    """
    Menyelesaikan sistem persamaan linear menggunakan dekomposisi Crout.
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.identity(n)  # U diagonal utama = 1

    # Dekomposisi Crout
    for i in range(n):
        for j in range(i, n):  # Menghitung elemen L
            L[j][i] = A[j][i] - sum(L[j][k] * U[k][i] for k in range(i)
