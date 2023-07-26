"""
P1. Online averaging.
Student name: Latthaphon Phoemmaniart
"""
import numpy as np
def sop(w, x):
  return np.dot(w, x)

if __name__ == "__main__":
# Test your code sufficiently!
  d = np.array([1, 2, 3, 4])
  v = np.array([-1, 0, 1, 2])
  print("dot product =", sop(d, v))