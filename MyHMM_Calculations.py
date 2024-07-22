from numpy import array, empty, zeros
from numpy import exp
from numpy import fliplr, einsum
from cfcn import logsumexptrick

#Log-sum-exp trick


def calc_gamma(alpha,beta,method=None):
    if method=='log':
        topPart=alpha+beta
        bottomPart=[logsumexptrick(cCol) for cCol in topPart.T]
        gamma=exp(topPart-bottomPart)
        gamma=gamma/gamma.sum(axis=0)
    else:
        topPart=einsum('it,it->it',alpha,beta)
        gamma=einsum('it,t->it',topPart,1/topPart.sum(axis=0))
    return gamma

def calc_logbeta(log_A,log_probObsState):
    Tsteps=log_probObsState.shape[1]
    numStates=log_A.shape[0]
    log_beta=zeros((numStates,Tsteps))
    log_probObsState=fliplr(log_probObsState)
    for tb in range(1,Tsteps):
        for i in range(numStates):
            terms=log_probObsState[:,tb-1]+log_beta[:,tb-1]+log_A[i,:]
            log_beta[i,tb]=logsumexptrick(terms)
    return fliplr(log_beta)

def calc_beta_scale1(A,probObsState,rescale=True):
    Tsteps=probObsState.shape[1]
    numStates=A.shape[0]
    beta=empty((numStates,Tsteps))
    beta[:,0]=numStates*[1]
    if rescale:
        beta[:,0]=beta[:,0]/beta[:,0].sum()
    probObsState=fliplr(probObsState)
    for tb in range(1,Tsteps):
        beta[:,tb]=einsum('j,ij,j->i',beta[:,tb-1],A,probObsState[:,tb-1])
        if rescale:
            beta[:,tb]=beta[:,tb]/beta[:,tb].sum()
    return fliplr(beta)


def calc_alpha_scale1(A,pi0,probObsState,rescale=True):
    Tsteps=probObsState.shape[1]
    numStates=A.shape[0]
    alpha=empty((numStates,Tsteps))
    # equivilent of elment-by-element multiplication
    alpha[:,0]=einsum('i,i->i',pi0,probObsState[:,0])
    if rescale:
        alpha[:,0]=alpha[:,0]/alpha[:,0].sum()
    for t in range(1,Tsteps):
        alpha[:,t]=einsum('j,ji,i->i',alpha[:,t-1],A,probObsState[:,t])
        if rescale:
            alpha[:,t]=alpha[:,t]/alpha[:,t].sum()
    return alpha

def calc_logalpha(log_A,log_pi0,log_probObsState):
    Tsteps=log_probObsState.shape[1]
    numStates=log_A.shape[0]
    log_alpha=empty((numStates,Tsteps))
    # equivilent of elment-by-element multiplication
    log_alpha[:,0]=log_pi0+log_probObsState[:,0]
    # Is there a fast way to perform this nested calculation?
    for t in range(1,Tsteps):
        for i in range(numStates):
            terms=log_probObsState[i,t]+log_alpha[:,t-1]+log_A[:,i]
            log_alpha[i,t]=logsumexptrick(terms)
    return log_alpha

def calc_xi(A,alpha,beta,probStateObs,method=None):
    Tsteps=probStateObs.shape[1]
    numStates=A.shape[0]
    if method=='log':
        topPart=array([[[alpha[i,t]+A[i,j]+beta[j,t+1]+probStateObs[j,t+1] for t in range(Tsteps-1)] for j in range(numStates)] for i in range(numStates)])
        bottomPart=array([logsumexptrick(cMat.reshape(-1)) for cMat in topPart.T])
        eta=exp(topPart-bottomPart)
        eta=array([cMat/cMat.sum() for cMat in eta.T]).T
    else:
        topPart=einsum('it,ij,jt,jt->ijt',alpha[:,0:-1],A,beta[:,1:],probStateObs[:,1:])
        # bottomPart=einsum('kt,kw,wt,wt->t',alpha[:,0:-1],A,beta[:,1:],probStateObs[:,1:])
        # eta=einsum('ijt,t->ijt',topPart,1/bottomPart)
        eta=einsum('ijt,t->ijt', topPart,1/topPart.sum(axis=(0,1)))
    return eta