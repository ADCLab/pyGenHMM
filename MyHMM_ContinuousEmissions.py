from models.emission_continuous import ExponentialModel, GaussianModel, LogNormalModel
from typing import Any, Optional
from scipy.stats import expon, norm, lognorm
from numpy.random import default_rng
from numpy import log as ln
from math import exp, sqrt
from numpy import einsum
from numpy import matmul
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
class exponential(ExponentialModel):
    rng: Optional[Any]=None
    rv: Optional[Any]=None
    
    def __init__(self,**data):
        super().__init__(**data)
        self.rng = default_rng()
        self.rv = expon(scale=self.mu)
        
    def genObs(self,N=1):
        if N>1:
            return self.rng.exponential(self.mu, size=N).tolist()
        else:
            return self.rng.exponential(self.mu, size=N)[0]
            
    def probObs(self,x):
        return self.rv.pdf(x).tolist()

    def logProbObs(self,x):
        return ln(self.rv.pdf(x)).tolist()
    
    def estimate(self,gammas,obs,update):
        numObs=len(obs)
        lamb = sum([sum(gammas[cObs]) for cObs in range(numObs)]) / sum([matmul(gammas[cObs],obs[cObs]) for cObs in range(numObs)])
        if update:
            self.update(lamb)
                    
    def update(self,lamb):
        self.lamb = lamb
        self.mu = 1/lamb
        self.rv = expon(scale=self.mu)
    
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
class gaussian(GaussianModel):
    rng: Optional[Any]=None
    rv: Optional[Any]=None
    
    def __init__(self,**data):
        super().__init__(**data)
        self.rng = default_rng()
        self.rv = norm(loc=self.mu, scale=self.std)
        
    def genObs(self,N=1):
        if N>1:
            return self.rng.normal(self.mu,self.std, size=N).tolist()
        else:
            return self.rng.normal(self.mu,self.std, size=N)[0]
            
        
    def probObs(self,x):
        return self.rv.pdf(x).tolist()

    def logProbObs(self,x):
        return ln(self.rv.pdf(x)).tolist()
    
    def estimate(self,gammas,obs,update):
        numObs=len(obs)
        mu = sum([matmul(gammas[cObs],obs[cObs]) for cObs in range(numObs)]) / sum([sum(gammas[cObs]) for cObs in range(numObs)])
        std = sqrt(  sum([matmul(gammas[cObs],[(ob-mu)**2 for ob in obs[cObs]]) for cObs in range(numObs)]) / sum([sum(gammas[cObs]) for cObs in range(numObs)])  )
        if update:
            self.update(mu,std)
            
    def update(self,mu,std):
        self.mu = mu
        self.std = std
        self.rv = norm(loc=self.mu, scale=self.std)
#---------------------------------------------------------------------
#---------------------------------------------------------------------    
#---------------------------------------------------------------------
#---------------------------------------------------------------------
class logNormal(LogNormalModel):
    rng: Optional[Any]=None
    rv: Optional[Any]=None
    
    def __init__(self,**data):
        super().__init__(**data)
        self.rng = default_rng()
        self.rv = lognorm(s=self.std, scale=exp(self.mu) )
        
    def genObs(self,N=1):
        k=1
        if self.isNeg:
            k=-1
        if N>1:
            return (k*self.rng.lognormal(self.mu,self.std, size=N)).tolist()
        else:
            return k*self.rng.lognormal(self.mu,self.std, size=N)[0]
                  
    def probObs(self,x):
        if self.isNeg:
            x = [-val for val in x]
        return self.rv.pdf(x).tolist()
        
    def logProbObs(self,x):
        if self.isNeg:
            x = [-val for val in x]
        return ln(self.rv.pdf(x)).tolist()
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------    
# class NegativeLogNormal(NegativeLogNormalModel):
#     rng: Optional[Any]=None
#     rv: Optional[Any]=None
    
#     def __init__(self,**data):
#         super().__init__(**data)
#         self.rng = default_rng()
#         self.rv = lognorm(s=self.std, scale=exp(self.mu) )
        
#     def genObs(self,N=1):
#         if N>1:
#             return (-self.rng.lognormal(self.mu,self.std, size=N)).tolist()
#         else:
#             return -self.rng.lognormal(self.mu,self.std, size=N)[0]
                  
#     def probObs(self,x):
#         return self.rv.pdf(-x).tolist()

#     def logProbObs(self,x):
#         return ln(self.rv.pdf(-x)).tolist()