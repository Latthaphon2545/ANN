import numpy as np

X = np.array([[8, 7, 6, 8, 6, 7]])

y = np.array([7.089])

def predict_wellbeing(w, k, s, t, p, h):
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    input_factors = np.array([w, k, s, t, p, h])
    predicted_wellbeing = np.dot(input_factors, coefficients)
    return predicted_wellbeing

if __name__ == "__main__":
    yhat = predict_wellbeing(8, 7, 6, 8, 6, 7)
    print(yhat)
