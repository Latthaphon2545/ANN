import numpy as np

A = np.array([[2, 1], [1.5, 3]])
b = np.array([[-7], [9]])

def loss(uvec):
    loss_value = np.dot(uvec.T, np.dot(A, uvec)) + np.dot(b.T, uvec)
    return loss_value[0,0]

def grad(uvec):
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
