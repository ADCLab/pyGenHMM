from pydantic import BaseModel, confloat, root_validator,Field 
from typing import Literal, Optional, List, Union
from numpy import random
# Continuous Distributions: 'poisson','binomial','choice','hypergeometric','negbin'

class EmissionDiscreteModel(BaseModel):
    variableType: str = Field("discrete", frozen=True)
    isDiscrete: bool = Field(True, frozen=True)

class PoissonModel(EmissionDiscreteModel):
    name: str = Field("poisson", frozen=True)
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

    
class DiscreteUniformModel(EmissionDiscreteModel):
    name: str = Field("uniform", frozen=True)
    a: int
    b: int