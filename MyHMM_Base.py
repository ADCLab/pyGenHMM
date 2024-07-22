from models.HMM import HMMModel
from typing import Any, Optional
from numpy import set_printoptions
from numpy import array
from numpy import log





class myHMM(HMMModel):
    T: Optional[Any]=None
    log_T: Optional[Any]=None
    pi0: Optional[Any]=None
    log_pi0: Optional[Any]=None
    emission: Optional[list]=[]
    numFeatures: Optional[int]=0
    logLikelihood: Optional[list]=[]
    
    
    
    
    def __init__(self, **data):
        super().__init__(**data)
        set_printoptions(precision=5)
        self.T=array(self.T_est)
        self.log_T=log(self.T)
        self.pi0=array(self.pi0_est).T
        self.log_pi0=log(self.pi0)
        
        
