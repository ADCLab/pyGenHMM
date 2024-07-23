from pydantic import BaseModel, confloat, root_validator,Field 
from typing import Literal, Optional, List, Union
from numpy import random
from math import exp, sqrt
from math import log as ln

# Continuous Distributions: 'uniform','gaussian','exponential','lognormal','triangular','gamma','weibull','beta'

class EmissionContinuousModel(BaseModel):
    variableType: str = Field("continuous", frozen=True)
    isDiscrete: bool = Field(False, frozen=True)

class ContinuousUniformModel(EmissionContinuousModel):
    name: str = Field("uniform", frozen=True)
    a: float
    b: float

class GaussianModel(EmissionContinuousModel):
    name: str = Field("gaussian", frozen=True)
    mu: float
    std:float=1

class LogNormalModel(EmissionContinuousModel):
    name: str = Field("Log-Normal", frozen=True)
    mu: Optional[float] = 0
    std: Optional[float] = 1
    mean: Optional[float] = None
    variance: Optional[float] = None
    isNeg: Optional[bool] = False

    @root_validator(pre=True)
    def check_values(cls, values):
        mu = values.get('mu')
        std = values.get('std')
        mean = values.get('mean')
        variance = values.get('variance')
        isNeg = values.get('isNeg')
        
        assert ((mu is not None) and (std is not None)) or ((mean is not None) and (variance is not None)), 'Must provide parameters of cooresponding Normal distribution or the mean and variance of the Log-Normal' 
        if ((mu is not None) and (std is not None)):
            values['mean'] = exp(mu+std**2/2)
            values['variance'] = (exp(std**2)-1)*exp(2*mu+std**2)
        else:
            assert (mean<0 and isNeg==True) or (mean>0 and isNeg==False), 'If mean is negative must assert isNeg=True as input'
            mean = -mean
            values['mu'] = ln(mean**2/sqrt(mean**2+variance**2))
            values['std'] = ln(1+(variance/mean)**2)            
        return values  


class NegativeLogNormalModel(EmissionContinuousModel):
    name: str = Field("Log-Normal (Negative -Y)", frozen=True)
    mu: float
    std:float=1


class CauchyModel(EmissionContinuousModel):
    name: str = Field("Cauchy", frozen=True)
    gamma: float
    x0:float=0

class ExponentialModel(EmissionContinuousModel):
    name: str = Field("exponential")
    lamb: Optional[float]=None
    mu: Optional[float]=None    

    @root_validator(pre=True)
    def check_values(cls, values):
        lamb = values.get('lamb')
        mu = values.get('mu')
        
        assert (lamb is not None) or (mu is not None), 'Must provide either lambda or mu value'
        if lamb is None:
            values['lamb']=1/mu
        elif mu is None:
            values['mu']=1/lamb
        elif mu!=1/lamb:
            print('Correcting mu so that mu=1/lambda')
            values['mu']=1/lamb
        return values        