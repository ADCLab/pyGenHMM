from cfnc import replaceZeros
from numpy import log

def setT(self,T):
    self.T=T
    self.log_T=log(  replaceZeros(T)  )
            
def setPi0(self, pi0):
    pi0=pi0/pi0.sum()
    self.pi0=pi0
    self.log_pi0=log(  replaceZeros(pi0)  )