import numpy as np
import sympy as sp

# A = np.array([[2, 1], [1.5, 3]])
# b = np.array([[-7], [9]])

def loss(uvec):
    loss_value = np.dot(np.dot(uvec.T, A),uvec) + np.dot(b.T, uvec)
    return loss_value[0,0]

def grad(uvec):
    gra1 = 4.0*uvec[0] + 1.5*uvec[1] - 7.0
    gra2 = 1.5*uvec[0] + 6.0*uvec[1] + 9.0
    # return np.array([[gra1], [gra2]])
    return np.dot(A + A.T, uvec) + b

def minimizer(uvec, lr, N):
    u = uvec.copy().astype(np.float64)  
    for _ in range(N):
        gradient = grad(u)
        u -= lr * gradient
    return u

if __name__ == '__main__':
    u0 = np.array([[0.0], [0.0]])  
    uz = minimizer(u0, lr=0.1, N=500)   
    print('type=', type(uz))
    print('shape=', uz.shape)
    print(': u*=', uz)
    print(': loss(u*)=', loss(uz))
    print(': grad(u*)=', grad(uz))

import numpy as np
import sympy as sp

# Define the variables as symbolic variables
u1, u2 = sp.symbols('u1 u2')
uvec = sp.Matrix([u1, u2])

# Define the loss function
A = np.array([[2, 1], [1.5, 3]])
b = np.array([[-7], [9]])

loss_expr = uvec.T * (A @ uvec) + uvec.T * b

# Compute the gradient
gradient = sp.Matrix([sp.diff(loss_expr, u1), sp.diff(loss_expr, u2)])
print(gradient)
