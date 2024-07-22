from multiprocessing import Pool
from numpy.random import choice

def genSequences(self,numSequences=10,maxLength=20,checkAbsorbing=False,method='iter',numCores=1):
    if method=='iter':
        stateSeqs=[self.genStateSequence(maxLength,checkAbsorbing) for x in range(numSequences)]
        outputSeqs=[self.genOutputSequence(stateSeq) for stateSeq in stateSeqs]
        # outputSeqs=[self.genOutputSequence(stateSeq,asList) for stateSeq in stateSeqs]   
    # map is not any faster than iteratively computing
    # elif method=='map':
        # stateSeqs=list(map(self.genStateSequence,NumSequences*[maxLength],NumSequences*[CheckAbsorbing]))
        # outputSeqs=list(map(self.genOutputSequence,stateSeqs))
    elif method=='pool':
        with Pool(numCores) as pool:
            stateSeqs = pool.starmap(self.genStateSequence, zip(numSequences*[maxLength],numSequences*[checkAbsorbing]))
            # outputSeqs=pool.map(self.genOutputSequence,stateSeqs)      
    return stateSeqs, outputSeqs

def genStateSequence(self,maxLength=20,checkAbsorbing=False):
    stateSeq=[choice(self.numStates,p=self.pi0)]
    for cTimestep in range(maxLength-1):
        if checkAbsorbing and self.T[stateSeq[-1],stateSeq[-1]]==1:
            break
        stateSeq.append(choice(self.numStates,p=self.T[stateSeq[-1]]))
    return stateSeq
    
def genOutputSequence(self,stateSeq,asList=True):
    return [[self.emission[cFeat][cState].genObs() for cState in stateSeq] for cFeat in range(self.numFeatures)]