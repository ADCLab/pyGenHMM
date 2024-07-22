from models.emission_continuous import EmissionContinuousModel
from models.emission_discrete import EmissionDiscreteModel
from numpy import log
from numpy import array
from cfcn import replaceZeros

def calcProbObsState(self,obs,method='log'):
    if method=='log':
        return log(  replaceZeros(  array([[self.emission[cFeature][cState].probObs(obs[cFeature]) for cState in range(self.numStates)] for cFeature in range(self.numFeatures)]), self.epsilon  )  ).sum(axis=0)
    else:
        return array([[self.emission[cFeature][cState].probObs(obs[cFeature]) for cState in range(self.numStates)] for cFeature in range(self.numFeatures)]).prod(axis=0)


def addFeatureEmissions(self, emissionList):
    assert isinstance(emissionList, list), 'Must provide a list of emission distributions'
    assert all([isinstance(em, EmissionContinuousModel) for em in emissionList]) or all([isinstance(em, EmissionDiscreteModel) for em in emissionList]), ''
    assert self.numStates==len(emissionList), 'The number of emission distributions must match the number of states'
    assert all([em.isDiscrete==False for em in emissionList]) or all([em.isDiscrete==True for em in emissionList]), 'All emission distribution must either be continuous or discrete, not a mix of each.'
    self.emission.append(emissionList)
    self.numFeatures=self.numFeatures+1
    
