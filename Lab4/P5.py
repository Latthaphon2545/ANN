import numpy as np

def polyM(Q, w):
    M = len(w) - 1
    y = w[0, 0]
    for i in range(1, M + 1):
        y += w[i, 0] * Q ** i
    return y

def polyM_MSE(DXY, w):
    X = DXY[:, 0]
    Y = DXY[:, 1]
    return np.mean((Y - polyM(X, w))**2)

def polyM_grad(DXY, w):
    X = DXY[:, 0]
    Y = DXY[:, 1]
    M = len(w) - 1
    gradient = np.zeros((M + 1, 1))
    y = polyM(X, w)
    gradient[0, 0] = np.mean(2 * (y - Y))
    for i in range(1, M + 1):
        gradient[i, 0] = np.mean(2 * (y - Y) * X ** i)
    return gradient

def polyM_traingd(DXY, w0, lr, epochs):
    w = w0
    for i in range(epochs):
        w -= lr*polyM_grad(DXY, w)
    return w


if __name__ == "__main__":
    DX = [0.000, 0.111, 0.222, 0.333, 0.444, 0.556, 0.667, 0.778, 0.889, 1]
    DY = [-0.028, 0.988, 1.387, 1.625, 1.089, 0.713, 0.328, 0.535, 1.112, 2.004] 
    DXY = np.array([DX, DY]).T
    w = np.array([1, 2, 1.5, -0.3, 1.2]).reshape((-1,1)) 
    print('polyM =', polyM(3, w))
    print('polyM_MSE =', polyM_MSE(DXY, w)) 
    print('polyM_grad =\n', polyM_grad(DXY, w))
    w0 = w
    wt3 = polyM_traingd(DXY, w0, lr=1e-2, epochs=3) 
    print('wt3 =\n', wt3)