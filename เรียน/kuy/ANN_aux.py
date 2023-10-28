import time

import numpy as np
from matplotlib import pyplot as plt


def mse_loss(X, Y, Yp):
    '''
    X: array N x D
    Y: array N x K
    Yp: array N x K
    '''

    return np.mean((Yp - Y) ** 2)

def mlp2r(c, v, b, w, x):
    '''
    x: array N x D
    w: array M x D
    b: array M x 1
    v: array 1 x M
    c: array 1 x 1
    '''

    z = sigmoid(b + np.dot(w, np.transpose(x)))  # z: array M x N
    f = np.transpose(c + np.dot(v, z))  # f: array N x 1

    return f


def sigmoid(a):
    h = 1/(1 + np.exp(-a))
    return h


def train_mlp2r(c, v, b, w, X, Y, step_size=0.1, epochs=1000,
                loss=mse_loss):
    #     '''
    #     w: array M x D
    #     b: array M x 1
    #     v: array 1 x M
    #     c: array 1 x 1
    #     X: array N x D
    #     Y: array N x 1
    #     step_size: scalar
    #     epochs: scalar (integer)
    #     '''


    N, D = X.shape
    step_size /= N

    losses = []

    for i in range(epochs):
        z = sigmoid(b + np.dot(w, np.transpose(X)))  # z: array M x N
        dz = np.multiply(1 - z, z)  # dz: z' array M x N
        vdz = np.multiply(np.transpose(v), dz)  # vdz: array M x N

        f = np.transpose(c + np.dot(v, z))  # f: array N x 1
        err = f - Y  # err: array N x 1

        dLc = np.dot(np.ones(N).reshape((1, -1)), err)  # array 1 x 1, c.f. np.sum(err)

        dLv = np.transpose(np.dot(z, err))  # array 1 x M, c.f. np.sum(np.multiply(z[0,:], np.transpose(err))), ...

        dLb = np.dot(vdz, err)  # array M x 1, c.f. ..., np.sum(np.multiply(v[0,1] * dz[1,:], np.transpose(err))), ...

        dLw = np.dot(np.multiply(np.transpose(err), vdz), X)  # array M x D

        c -= dLc * step_size
        v -= dLv * step_size
        b -= dLb * step_size
        w -= dLw * step_size

        losses.append(loss(X, Y, f))

    return c, v, b, w, losses

def gen_data(N=500):
    x = np.linspace(0, 1, N)
    noise = np.random.rand(N)
    y = x + 0.3 * np.sin(2 * np.pi * x) + 0.1 * noise

    return x.reshape((-1, 1)), y.reshape((-1, 1))

def gen_data2(N=500):
    x = np.linspace(0, 1, N)
    noise = np.random.rand(N)
    y = x + 0.3 * np.sin(2 * np.pi * x) + 0.1 * noise

    xo = 10*x + 100

    return xo.reshape((-1, 1)), y.reshape((-1, 1))


def gen2(N=100):
    x = np.linspace(0, 1, N)
    y = np.pi*4 * x + 8 * np.sin(4*np.pi*x) + np.random.normal(0, 1, N)

    return x.reshape((N,1)), y.reshape((N,1))



def initUnif(D, M, K, params=(-1,1)):
    '''
    :param D:
    :param M:
    :param K:
    :param params: tuple of (wmin, wmax)
    :return: c, v, b, w
    '''

    wmin = params[0]
    wmax = params[1]

    scale = (wmax - wmin)

    c = np.random.rand(K).reshape((K, 1)) * scale + wmin
    v = np.random.rand(K * M).reshape((K, M)) * scale + wmin

    b = np.random.rand(M).reshape((M, 1)) * scale + wmin
    w = np.random.rand(M * D).reshape((M, D)) * scale + wmin

    return c, v, b, w

def initNorm(D, M, K, params=1):
    '''
    :param D:
    :param M:
    :param K:
    :param params: float of normal SD
    :return: c, v, b, w
    '''

    SD = params
    c = np.random.normal(0, SD, K).reshape((K, 1))
    v = np.random.normal(0, SD, K * M).reshape((K, M))

    b = np.random.normal(0, SD, M).reshape((M, 1))
    w = np.random.normal(0, SD, M * D).reshape((M, D))

    return c, v, b, w


def experiment(Data_Size=100, M=8, rhoo=0.9, rhoh=0.9, num_epochs=1000,
               initf=initUnif, initParams=None, show=True):
    # 1. Acquire Data
    X, Y = gen2(Data_Size)

    # 2. Choose a model and its complexity
    #M

    # 3. Choose training parameters
    # rhoo
    # rhoh
    # num_epochs

    N, D = X.shape
    _, K = Y.shape
    rhoo /= N
    rhoh /= N

    # 4. Initialize weights
    # c, v, b, w = init_weight2(1, M, 1)
    c, v, b, w = initf(D, M, K, initParams)

    losses = []
    # 5. Train the model
    for i in range(num_epochs):

        z = sigmoid(b + np.dot(w, np.transpose(X)))  # z: array M x N
        dz = np.multiply(1 - z, z)  # dz: z' array M x N
        vdz = np.multiply(np.transpose(v), dz)  # vdz: array M x N

        f = np.transpose(c + np.dot(v, z))  # f: array N x 1
        err = f - Y  # err: array N x 1

        dLc = np.dot(np.ones(N).reshape((1, -1)), err)  # array 1 x 1, c.f. np.sum(err)

        dLv = np.transpose(np.dot(z, err))  # array 1 x M, c.f. np.sum(np.multiply(z[0,:], np.transpose(err))), ...

        dLb = np.dot(vdz, err)  # array M x 1, c.f. ..., np.sum(np.multiply(v[0,1] * dz[1,:], np.transpose(err))), ...

        dLw = np.dot(np.multiply(np.transpose(err), vdz), X)  # array M x D

        mse = np.mean(err ** 2)

        losses.append(mse)

        if np.isnan(mse):  # note: for testing, np.exp(800)*0 gives NaN.
            print('Reach NaN. Terminated.')
            return mse, c, v, b, w

        c -= dLc * rhoo
        v -= dLv * rhoo
        b -= dLb * rhoh
        w -= dLw * rhoh

    if show:

        xs = np.linspace(np.min(X), np.max(X), 20).reshape((-1, 1))

        # Plot z response
        plt.subplot(2, 2, 1)
        zs = sigmoid(b + np.dot(w, np.transpose(xs)))
        for m in range(M):
            plt.plot(xs, np.transpose(zs[m, :]))

        # Plot y response
        plt.subplot(2, 2, 3)
        ys = mlp2r(c, v, b, w, xs)
        plt.plot(X, Y, 'bx')
        plt.plot(xs, ys, 'r')
        plt.title(str(i) + ': mse={:.4f}'.format(mse))

        # Plot c, v
        plt.subplot(2, 2, 2)
        pos_c = range(len(c))
        plt.plot(pos_c, c.ravel(), 'bo')
        v_rav = v.ravel()
        # print('debug: v_rav=', v_rav)
        # print('debug: len(v_rav)=', len(v_rav))
        pos_v = np.arange(len(v_rav)) + len(c)
        plt.plot(pos_v, v_rav, 'ro')

        # Plot b, w
        plt.subplot(2, 2, 4)
        b_rav = b.ravel()
        pos_b = range(len(b_rav))
        plt.plot(pos_b, b_rav, 'bo')
        w_rav = w.ravel()
        pos_w = np.arange(len(w_rav)) + len(b_rav)
        plt.plot(pos_w, w_rav, 'ro')

        plt.show()

    return mse, losses, c, v, b, w, X, Y

def visualize_result(c, v, b, w, X, Y):

    _, D = X.shape
    _, M = v.shape

    N_res = 100 # plot resolution
    xs = np.linspace(np.min(X), np.max(X), N_res).reshape((N_res,D))

    # Plot z response
    plt.subplot(2, 2, 1)
    zs = sigmoid(b + np.dot(w, np.transpose(xs)))
    for m in range(M):
        plt.plot(xs, np.transpose(zs[m, :]))

    # Plot y response
    plt.subplot(2, 2, 3)
    plt.plot(X, Y, 'bx')
    ys = mlp2r(c, v, b, w, xs)
    plt.plot(xs, ys, 'r')
    plt.legend(['Data', 'ANN'])

    # Plot c, v
    plt.subplot(2, 2, 2)
    pos_c = range(len(c))
    plt.plot(pos_c, c.ravel(), 'bo')
    v_rav = v.ravel()
    # print('debug: v_rav=', v_rav)
    # print('debug: len(v_rav)=', len(v_rav))
    pos_v = np.arange(len(v_rav)) + len(c)
    plt.plot(pos_v, v_rav, 'ro')

    # Plot b, w
    plt.subplot(2, 2, 4)
    b_rav = b.ravel()
    pos_b = range(len(b_rav))
    plt.plot(pos_b, b_rav, 'bo')
    w_rav = w.ravel()
    pos_w = np.arange(len(w_rav)) + len(b_rav)
    plt.plot(pos_w, w_rav, 'ro')

    plt.show()

def save_ann(c, v, b, w, name='ann'):

    np.save(name+'c.npy', c)
    np.save(name+'v.npy', v)
    np.save(name+'b.npy', b)
    np.save(name+'w.npy', w)


def load_ann(name='ann'):
    c = np.load(name + 'c.npy')
    v = np.load(name + 'v.npy')
    b = np.load(name + 'b.npy')
    w = np.load(name + 'w.npy')

    return c, v, b, w

if __name__ == '__main__':

    t1 = time.time()
    mse, losses, c, v, b, w, X, Y = experiment(Data_Size=100,
        M=8, rhoo=0.3, rhoh=0.3, num_epochs=40000,
        initf=initNorm, initParams=1, show=False)
    t2 = time.time()

    print('training time = ', t2 - t1)


    save_ann(c, v, b, w, 'ann191008b')
    # c, v, b, w = load_ann('ann191008a')

    print('training mse', mse)
    plt.plot(losses)
    plt.show()
    visualize_result(c, v, b, w, X, Y)



