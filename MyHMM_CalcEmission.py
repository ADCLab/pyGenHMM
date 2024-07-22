from numpy import array

def probObsInStates(self,obs):
    return array([self.probFeatureObs(cFeat,obs[cFeat]) for cFeat in range(self.numFeatures)]).prod(axis=0)
    
def probFeatureObs(self,cFeature,obs):
    return [self.probFeatureStateObs(cFeature,cState,obs) for cState in range(self.numStates)]

def probFeatureStateObs(self,cFeature,cState,obs):
    return self.emission[cFeature][cState].probObs(obs)