import numpy as np

def act(w, x):
  return np.dot(w,np.exp(-(abs(x))))

if __name__ == "__main__":
  w = np.array([[1, 4],[2, -1],[0.5, 3]])
  x = np.array([[7],[9]])
  A = act(w, x)
  print('A=', A)