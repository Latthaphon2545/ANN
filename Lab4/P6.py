import numpy as np

# Define a function to normalize the input flow rate (Q)
def normalize_input(Q):
    min_original = 200
    max_original = 500
    normalized_Q = (Q - min_original) / (max_original - min_original)
    
    return normalized_Q

def pump_head(Q):
    normalized_Q = normalize_input(Q)
    
    data = np.load('P6_TrainQHv4.npy')
    
    normalized_flow_rates = normalize_input(data[:, 0])
    head_pressures = data[:, 1]
    
    estimated_head_pressure = np.interp(normalized_Q, normalized_flow_rates, head_pressures)
    
    return estimated_head_pressure

if __name__ == "__main__":
    TrainQH = np.load('P6_TrainQHv4.npy')
    print('Q =', (3*"{:.1f} ;").format(*TrainQH[:3, 0]), "...")
    print('H =', (3*"{:.1f} ;").format(*TrainQH[:3, 1]), "...")
    Q = 205
    H = pump_head(Q)
    print('For Q =', Q, '; H =', H)
