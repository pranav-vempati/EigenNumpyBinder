import numpy as np
import EigenNumpyBinder

arr_eig = np.random.rand(3,3)

print('Random matrix: ', arr_eig)

eig_instance = EigenNumpyBinder.EigenVectorWrapper(arr_eig)

print('First eigenvector: ', eig_instance.extract_first_eigenvector())

matrixA = np.random.randn(5,5)

matrixB = np.random.randn(5,5)

lu_instance = EigenNumpyBinder.LUWrapper(matrixA, matrixB)

print(' matrixA:', matrixA)

print('matrixB',  matrixB)

print('Solution matrix X, AX = B: ', lu_instance.lu_solver())

qr_mat = np.random.randn(5,5)

qr_instance = EigenNumpyBinder.QRWrapper(qr_mat)

print('Data matrix: ', qr_mat)

print('Orthogonal matrix, Q: ', qr_instance.calculate_householder())

