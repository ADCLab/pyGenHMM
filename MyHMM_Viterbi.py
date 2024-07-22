from multiprocessing import Pool
from numpy import empty, einsum

# probability == p. Tm: the transition matrix. Em: the emission matrix.
def viterbi(self,allYs, method='exact',usePool=False,numCores=6):
    if usePool:
        with Pool(numCores) as pool: 
            if method=='exact':
                optPaths=pool.map(self.viterbi_exact,allYs)
    else:
        optPaths=[]
        for Ys in allYs:
            optPaths.append(self.viterbi_exact(Ys))

    return optPaths
   
def viterbi_exact(self,obs):
    Tsteps=len(obs[0])
    mu=empty((self.numStates,Tsteps))
    bestPriorState=empty((self.numStates,Tsteps)) 
    optPath=[]

    probObsState=self.probObsInStates(obs)    
    mu[:,0]=einsum('i,i->i',self.pi0,probObsState[:,0])
    mu[:,0]=mu[:,0]/mu[:,0].sum()
    for t in range(Tsteps-1):
        mu_notopt=einsum('k,pk,p->kp',probObsState[:,t+1],self.T,mu[:,t])
        mu[:,t+1]=mu_notopt.max(axis=1)
        mu[:,t+1]=mu[:,t+1]/mu[:,t+1].sum()
        bestPriorState[:,t]=mu_notopt.argmax(axis=1)
    
    LastStateInPath=int(mu[:,t+1].argmax())
    optPath.append(LastStateInPath)
    for t in range(Tsteps-2,-1,-1):
        try:
            LastStateInPath=int(bestPriorState[LastStateInPath,t])
            optPath.append(LastStateInPath)
        except Warning:
            5
    optPath.reverse()
    return optPath   