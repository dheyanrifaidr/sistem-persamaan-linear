import numpy as np
from main import inverse_matrix_method, lu_decomposition_gauss, crout_decomposition

def test_methods():
    # Matriks koefisien A dan vektor konstanta B
    A = np.array([[2, -1, 1],
                  [3, 3, 9],
                  [3, 3, 5]], dtype=float)
    B = np.array([3, -42, -12], dtype=float)
    
    # Solusi yang diharapkan (dihitung secara manual atau dengan metode terpercaya)
    expected_solution = np.array([1, -2, -3], dtype=float)
    
    # Uji Metode Matriks Balikan
    print("=== Uji Metode Matriks Balikan ===")
    X_inverse = inverse_matrix_method(A, B)
    print("Hasil:", X_inverse)
    assert np.allclose(X_inverse, expected_solution), "Metode Matriks Balikan gagal!"

    # Uji Metode LU Gauss
    print("\n=== Uji Metode LU Gauss ===")
    X_lu = lu_decomposition_gauss(A, B)
    print("Hasil:", X_lu)
    assert np.allclose(X_lu, expected_solution), "Metode LU Gauss gagal!"

    # Uji Metode Crout
    print("\n=== Uji Metode Dekomposisi Crout ===")
    X_crout = crout_decomposition(A, B)
    print("Hasil:", X_crout)
    assert np.allclose(X_crout, expected_solution), "Metode Crout gagal!"

if __name__ == "__main__":
    test_methods()
