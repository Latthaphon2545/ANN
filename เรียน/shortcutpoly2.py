"""
Close form solution to zero gradient of polynomial curve fitting
min E = (poly(x, w) - y)^2
equiv. solve grad_w = 0 for w.

return w
"""

import numpy as np


def super_train_poly(w, DataX, DataY):
    '''
    y [1 x N] = wt [1 x M] * Phi [M x N]

    Grad [M x 1] = Phi [M x N] * (yt [N x 1] - Yt [N x 1])
    Thus, at Grad = 0, solve for w in Phi [M x N] * (yt [N x 1] - Yt [N x 1]) = Phi * t(Phi) * w - Phi * Yt = 0
    '''

    Mp = len(w)
    N = len(DataX)

    # Compose Phi matrix [M x N]
    Phi = np.vstack((np.ones((1, N)),
                     np.tile(DataX, (Mp - 1, 1))
                     ))

    # Make it [1 .. 1; x0 x1 .. xn; x0^2, x1^2 .. xn^2; .. xn^M]
    Phi = np.cumprod(Phi, axis=0)

    # Use linear solver A x = b

    A = np.matmul(Phi, np.transpose(Phi))
    b = np.matmul(Phi, np.transpose(DataY))

    w = np.linalg.solve(A, b)

    return w


def test_sc():
    DatX = np.array([1, 2, 3, 4, 5, 6])
    DatY = np.array([8, 7, 6, 8, 6, 7])

    w = super_train_poly([0], DatX, DatY)
    totol = 1
    for i in range(0, len(w)):
        totol *= w[i]
    
    print(f'total = {totol}')
    print('w = ', w)


if __name__ == '__main__':
    print('test')
    test_sc()
