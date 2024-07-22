from models.emission_discrete import PoissonModel
from typing import Any, Optional
from scipy.stats import poisson as poisson_dist
from numpy.random import default_rng
from numpy import log as ln
from numpy import matmul
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
class poisson(PoissonModel):
    rng: Optional[Any]=None
    rv: Optional[Any]=None
    
    def __init__(self,**data):
        super().__init__(**data)
        self.rng = default_rng()
        self.rv = poisson_dist(mu=self.mu)
        
    def genObs(self,N=1):
        if N>1:
            return self.rng.poisson(lam=self.mu, size=N).tolist()
        else:
            return self.rng.poisson(lam=self.mu, size=N)[0]
                    
    def probObs(self,x):
        return self.rv.pmf(x).tolist()

    def logProbObs(self,x):
        return ln(self.rv.pmf(x)).tolist()
    
    def estimate(self,gammas,obs,update):
        numObs=len(obs)
        lamb = sum([matmul(gammas[cObs],obs[cObs]) for cObs in range(numObs)])  /  sum([sum(gammas[cObs]) for cObs in range(numObs)])
        if update:
            self.update(lamb)
        return lamb
            
    def update(self,mu):
        self.mu = mu
        self.lamb = 1/mu
        self.rv = poisson_dist(mu=self.mu)