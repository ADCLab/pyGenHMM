from numpy import sum, log, exp

def replaceZeros(data, eps):
   # min_nonzero = min(data[nonzero(data)])
  # data[data == 0] = min_nonzero
  data[data == 0] = eps
  return data

def logsumexptrick(x):
    c = x.max()
    return c + log(sum(exp(x - c)))