import numpy as np
def pump_head(Q):
    TrainQH = np.load('Lab4/P6_TrainQHv4.npy')
    Qs = TrainQH[:, 0]
    Hs = TrainQH[:, 1]
    H = np.interp(Q, Qs, Hs)
    return H
    

if __name__ == "__main__":
    TrainQH = np.load('Lab4/P6_TrainQHv4.npy')
    print('Q =', (3*"{:.1f} ;").format(*TrainQH[:3, 0]), "...")
    print('H =', (3*"{:.1f} ;").format(*TrainQH[:3, 1]), "...")
    Q = 205
    H = pump_head(Q)
    print('For Q =', Q, '; H =', H)