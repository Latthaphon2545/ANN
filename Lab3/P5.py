import numpy as np

def loss(u):
    return ( u - 1.5 )**2 - (3*(np.log(u + 2)))

def grad(u):
    return 2*u - 3 - 3/(u+2)

def minimizer(u, alpha, num_iter):
    for i in range(num_iter):
        u -= alpha * grad(u)
    return u

if __name__ == "__main__":
    u = 1
    print('loss=', loss(u))
    print('grad=', grad(u))
    print('minimizer=', minimizer(u, 0.1, 10))
    print('minimizer=', minimizer(u, 0.3, 10))
    print('minimizer=', minimizer(u, 0.9, 10))
    print('minimizer=', minimizer(u, 0.3, 100))