from pydantic import BaseModel, root_validator
from typing import List, Union
from emission_continuous import ContinuousUniformModel, GaussianModel, ExponentialModel
from emission_discrete import DiscreteUniformModel,PoissonModel

    
class EmissionListModel(BaseModel):
    __root__: List[  Union[ContinuousUniformModel,
                           GaussianModel,
                           ExponentialModel,
                           DiscreteUniformModel,
                           PoissonModel
                           ]  
                   ]    # Using __root__ to define a list of Item objects
    