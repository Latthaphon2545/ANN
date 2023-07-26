import numpy as np
def activation(w, b, x):
  return b + np.dot(w, x)

if __name__ == "__main__":
  w = np.array([[1, 4],[2, -1],[0.5, 3]])
  b = np.array([[9], [6], [3]])
  x = np.array([[7, 1, 0, 2],[9, -1, 0.5, 1]])
  A = activation(w, b, x)
  print('A=', A)