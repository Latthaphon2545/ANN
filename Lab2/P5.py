import numpy as np
def acth(w, x):
  return np.dot(w,np.tanh(x))

if __name__ == "__main__":
  w = np.array([[1, 4],[2, -1],[0.5, 3]])
  x = np.array([[7, 1, 0, 2],[9, -1, 0.5, 1]])
  A = acth(w, x)
  print('A=', A)