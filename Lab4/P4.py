import numpy as np

Pipe = np.array([[20.545, 28.500, 30.142, 35.603], [160, 256, 272, 320]]).T

def poly1(Q, w):
    return w[0][0] + w[1][0]*Q

def poly1_MSE(HQ, w):
    N = HQ.shape[0]
    Q = HQ[:,1].reshape((N,1))
    H = HQ[:,0].reshape((N,1))
    return np.mean((H - poly1(Q, w))**2)

def poly1_grad(HQ, w):
    H = HQ[:, 0]
    Q = HQ[:, 1]
    y = poly1(Q, w)
    M1 = len(w)
    dE = np.zeros((M1,1))
    for i in range(M1):
        dE[i] = 2 * np.mean((y - H) * Q**i)   
    return dE

def poly1_traingd(HQ, w0, lr, epochs):
    w = w0.copy()
    for i in range(epochs):
        grad = poly1_grad(HQ, w)
        w -= lr * grad
    return w

def approx_head(Q):
    while True:
        w0 = np.random.random(2).reshape((2,1))
        lr = 0.00001
        epochs = 500
        w = poly1_traingd(Pipe, w0, lr, epochs)
        # print('w =\n', w)
        if poly1_MSE(Pipe, w) < 2:
            # print('MSE =', poly1_MSE(Pipe, w))
            return poly1(Q, w)
        
def calculate_r_squared(HQ, w):
    H = HQ[:, 0]
    Q = HQ[:, 1]
    y_pred = poly1(Q, w)
    mean_H = np.mean(H)
    ss_total = np.sum((H - mean_H) ** 2)
    ss_residual = np.sum((H - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

if __name__ == "__main__":
    Pipe = np.array([[20.545, 28.500, 30.142, 35.603], [160, 256, 272, 320]]).T
    w = np.array([2, 0.1]).reshape((2,1))

    print('poly1 =', poly1(160, w)) 
    print('poly1_MSE =\n', poly1_MSE(Pipe, w)) 
    print('poly1_grad =\n', poly1_grad(Pipe, w))

    w0 = np.random.random(2).reshape((2,1)) 
    lr = 0.00001
    epochs = 500
    w_gd = poly1_traingd(Pipe, w0, lr, epochs) 
    print('poly1_traingd =\n', w_gd)
    
    for Q in Pipe[:,1]:
        H = approx_head(Q)
        print('Q = {:.3f} , H ~ {:.3f}'.format(Q,H))
        # r_sq = calculate_r_squared(Pipe, w_gd)
        # print('r_squared = {:.3f}'.format(r_sq))