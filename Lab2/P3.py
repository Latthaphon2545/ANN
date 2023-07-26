import numpy as np
def hf(x):
  return (1/(1+(2**(-x))))

def netz(c, x):
  return np.multiply(c, hf(x))
  
if __name__ == "__main__":
  c = np.array([[1, 3, 2],[8, 4, 5.6]])
  x = np.array([[0.5, 1, 2], [-1, 4, 10]])
  print('h(x)=', hf(x))
  print('z=', netz(c, x))