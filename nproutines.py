import numpy as np

import EigenNumpyBinder

arr = np.random.rand(3,3)

print('Random matrix: ', arr)

matrix = EigenNumpyBinder.MatrixWrapper(arr)

print('First eigenvector: ', matrix.extract_first_eigenvector())




