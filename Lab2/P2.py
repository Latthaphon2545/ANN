import numpy as np
def activation(w, x):
  return np.dot(w, x)

if __name__ == "__main__":
  w = np.array([[1, 4],[2, -1],[0.5, 3]])
  x = np.array([[7],[9]])
  a = activation(w, x)
  print(type(a))
  print(a.shape)
  print('a=', a)