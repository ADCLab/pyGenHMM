import numpy
from numpy import array, prod, empty, multiply, dot, ones, fliplr, matmul

from numpy import array, random, diag, einsum, zeros, log, inf
from numpy.linalg import eigh, inv, norm

 


def calc_alpha(obs,T,pi0,probObsState):
    Tsteps=obs[0].shape[0]
    numStates=T.shape[0]
    alpha=empty((numStates,Tsteps))
    # equivilent of elment-by-element multiplication
    alpha[:,0]=einsum('i,i->i',pi0,probObsState[:,0])
    alpha[:,0]=alpha[:,0]/alpha[:,0].sum()
    for t in range(1,Tsteps):
        alpha[:,t]=einsum('j,ji,i->i',alpha[:,t-1],T,probObsState[:,t])
        alpha[:,t]=alpha[:,t]/alpha[:,t].sum()
    return alpha

def calc_beta(obs,T,probObsState):
    Tsteps=obs[0].shape[0]
    numStates=T.shape[0]
    beta=empty((numStates,Tsteps))
    beta[:,0]=numStates*[.5]
    probObsState=fliplr(probObsState)
    for tb in range(1,Tsteps):
        beta[:,tb]=einsum('j,ij,j->i',beta[:,tb-1],T,probObsState[:,tb-1])
        beta[:,tb]=beta[:,tb]/beta[:,tb].sum()
    return fliplr(beta)

def calc_gamma(alpha,beta):
    topPart=einsum('it,it->it',alpha,beta)
    gamma=einsum('it,t->it',topPart,1/topPart.sum(axis=0))
    return gamma

def calc_eta(obs,T,alpha,beta,probStateObs):
    topPart=einsum('it,ij,jt,jt->ijt',alpha[:,0:-1],T,beta[:,1:],probStateObs[:,1:])
    bottomPart=einsum('kt,kw,wt,wt->t',alpha[:,0:-1],T,beta[:,1:],probStateObs[:,1:])
    eta=einsum('ijt,t->ijt',topPart,1/bottomPart)
    return eta

def viterbi(obs,HMM):
    T=len(obs)
    delta=empty((HMM.numStates,T))
    psi=empty((HMM.numStates,T))
    delta[:,0]=multiply(HMM.pi0,[HMM.emission.probStateObs(cState,obs[:,0]) for cState in range(HMM.numStates)])
    psi[:,0]=HMM.numStates*[-1]
    for t in range(T-1):
        bestPathCostSoFar=array([multiply(delta[:,t], cCol) for cCol in HMM.T.T]).T.max(axis=0)
        probStateObs=[HMM.emission.probStateObs(cState,obs[:,t+1]) for cState in range(HMM.numStates)]
        delta[:,t+1]=multiply(bestPathCostSoFar,probStateObs)
        psi[:,t+1]=array([multiply(delta[:,t], cCol) for cCol in HMM.T.T]).T.argmax(axis=0)
    optPath=(T+1)*[None]
    optPath[T]=delta[:,T-1].argmax()
    for t in range(T-1,0,-1):
        print(t)
        print(optPath)
        optPath[t]=int(psi[optPath[t+1],t])
    return delta,psi,optPath

    
            
            
def calc_ProbObs(obs,HMM):
    return sum(calc_alpha(obs,HMM)[:,-1])


class MyValidationError(Exception):
    pass

class MarkovSeqLibrary():
    def __init__(self):
        self.__stateSeqs=[]
        self.__outputSeqs=[]

    def addSeq(self,stateSeq,outputSeq):
        self.__stateSeqs.append(stateSeq)
        self.__outputSeqs.append(outputSeq)
        
        
        

class MarkovSeq():
    numSeqs=None
    stateSeq=None
    outputSeq=None
    def __init__(self,stateSeq,outputSeq):
        self.__stateSeq=stateSeq
        self.__outputSeq=outputSeq
        
    @property
    def state(self):
        return self.__stateSeq
    
    @property
    def output(self):
        return self.__outputSeq
    

class Emission():
    def __init__(self):
        self.properties=None
        self.emType=None

class discreteEmission(Emission):


    
    def __init__(self,numStates,emMat=None,numOutputsPerFeature=None):
        self.emMat=None
        self.numOutputsPerFeature=None
                                  
        if emMat is None:
            if numOutputsPerFeature is None:
                raise MyValidationError("Must provide Emission matrix or numOutputFeatures and numOutputsPerFeature")
            else:
                A=random.rand(numStates,numOutputsPerFeature)
                self.emMat=(A.T/A.sum(axis=1)).T
        elif isinstance(emMat,(numpy.ndarray, numpy.generic)) and (emMat.ndim==2) and (emMat.shape[0]==numStates):
            self.emMat=emMat
        else:
            raise MyValidationError("Emission matrix not valid type or shape")
        self.emType='discrete'
        self.numOutputsPerFeature=self.emMat.shape[1]
        self.topPart=zeros(self.emMat.shape) 
    
    def generateEmission(self,stateSeq):
        return numpy.array([numpy.random.choice(self.numOutputsPerFeature,p=self.emMat[cState]) for cState in stateSeq])

    def probObs(self,Obs):
        return array([self.emMat[:,cObs] for cObs in Obs]).T
    
    def probStateObs(self,cState,cObs):
        return self.emMat[cState,cObs]
        
    def calcTopPart(self,obs,gamma):
        Tsteps=obs.shape[0]
        Indicator=zeros((self.numOutputsPerFeature, Tsteps))
        Indicator[obs,range(Tsteps)]=1
        topPart=einsum('kt,it->ik',Indicator,gamma)
        self.topPart=self.topPart+topPart

    def updateEmission(self,bottomPart):
        self.emMat = einsum('ik,i->ik',self.topPart,1/bottomPart)
        self.topPart=zeros(self.emMat.shape) 

    
        
        
            
class myHMM():
    def __init__(self, T=None,numStates=None,pi0=None):
        self.T=None
        self.numStates=None
        self.numOutputFeatures=0
        self.emission=[]
        self.pi0=None
        if isinstance(numStates,int) and (T is None):
            tmpMat=random.rand(numStates,numStates)
            T=(tmpMat/tmpMat.sum(axis=0)).T
        elif T is not None:
            self.addT(T)
            
        else:
            raise MyValidationError("Must provide Transition Matrix or number of states")
        self.addT(T)
        self.addPi0(pi0)
                          
    def addT(self,T):
        if isinstance(T,(numpy.ndarray, numpy.generic)) and T.shape[0]==T.shape[1]:
            self.T=T 
            self.numStates=self.T.shape[0]
        else:
            raise MyValidationError("Unexpected Transition Matrix")
                   
    def addEmission(self, emType=None,**kargs):
        if self.numStates is None:
            raise MyValidationError("Must first define transition matrix or number of states")
            
        if emType=='discrete':
            newEmission=discreteEmission(self.numStates,**kargs)
            self.emission.append(newEmission)
        else:
            raise MyValidationError("Must provide valid emission type")
        self.numOutputFeatures=self.numOutputFeatures+1

    def addPi0(self, pi0=None,useSteadyState=True):
        if pi0 is None:
            # if useSteadyState:
                
            #     else
            tmpMat=random.rand(self.numStates)
            pi0=tmpMat/tmpMat.sum()
         

            self.pi0=pi0
        elif isinstance(pi0,(numpy.ndarray, numpy.generic)) and self.T.shape[0]==self.numStates:
            pi0=pi0/pi0.sum()
            self.pi0=pi0
        else:
            raise MyValidationError("Something wrong with pi0 input")
            
    def genSequences(self,NumSequences=100,maxLength=100,CheckAbsorbing=False):
        stateSeqs=[self.genStateSequence(maxLength,CheckAbsorbing) for x in range(NumSequences)]
        outputSeqs=[self.genOutputSequence(stateSeq) for stateSeq in stateSeqs]
        return stateSeqs, outputSeqs
        # return([[stateSeq,outputSeq] for stateSeq,outputSeq in zip(stateSeqs,outputSeqs)])
    
    def genStateSequence(self,maxLength=100,CheckAbsorbing=False):
        stateSeq=[numpy.random.choice(self.numStates,p=self.pi0)]
        for cState in range(maxLength-1):
            if CheckAbsorbing and self.T[stateSeq[-1],stateSeq[-1]]==1:
                break
            stateSeq.append(numpy.random.choice(self.numStates,p=self.T[stateSeq[-1]]))
        return stateSeq
        
    def genOutputSequence(self,stateSeq):
        outputSeq=[]
        for cEmission in self.emission:
            outputSeq.append(cEmission.generateEmission(stateSeq))
        return outputSeq


    


    def train(self,Ys,iterations=20,Ttrue=None,pi0true=None):
        lastLogProb=-inf
        for iter in range(iterations):
            pi0_topPart=zeros((self.numStates))
            T_topPart=zeros((self.numStates,self.numStates))
            T_bottomPart=zeros((self.numStates))
            b_bottomPart=zeros((self.numStates))
            b_topParts=[zeros((cEmission.emMat.shape)) for cEmission in self.emission]
            logProb=0
            for obs in Ys:
                probObsState=array([self.emission[cFeat].probObs(obs[cFeat]) for cFeat in range(self.numOutputFeatures)]).prod(axis=0)
                
                # probObsState=array([cEmission.probObs(cObs) for cEmission,cObs in zip(self.emission,obs)]).prod(axis=0)
                # probObsState=array([cEmission.probObs(cObs) for cEmission,cObs in zip(self.emission,obs)]).prod(axis=0)


    
                alpha=calc_alpha(obs,self.T,self.pi0,probObsState)
                beta=calc_beta(obs,self.T,probObsState)
                gamma=calc_gamma(alpha, beta)
                eta=calc_eta(obs, self.T, alpha, beta, probObsState)
                pi0_topPart=pi0_topPart+gamma[:,0]
                T_topPart=T_topPart+eta.sum(axis=2)
                T_bottomPart=T_bottomPart+gamma[:,0:-1].sum(axis=1)
                
                
                b_bottomPart=b_bottomPart+gamma.sum(axis=1)
                
                # logProb=logProb+log(alpha[:,-1].sum())
                for cFeat in range(len(obs)):
                    self.emission[cFeat].calcTopPart(obs[cFeat],gamma)

            if logProb<lastLogProb:
                a=5
            else:
                lastLogProb=logProb
            R=len(Ys)
            pi0=pi0_topPart/R
            T=einsum('ij,i->ij',T_topPart,1/T_bottomPart)
            
            # emMat=einsum('ik,i->ik',b_topPart,1/b_bottomPart)
            self.T=T
            self.pi0=pi0
            for cFeat in range(len(obs)):
                self.emission[cFeat].updateEmission(b_bottomPart)
            # for cEmission in self.emission:
            #     cEmission.updateEmission(b_bottomPart)
            # print(norm(self.pi0-pi00))
            # if Ttrue is not None:
            #     print(norm(self.T-Ttrue))
            # if pi0true is not None:
            #     print(norm(self.pi0-pi0true))            
            # print(logProb)
            print(self.T)
            
        

        
                   
                 


def createRandomEmission(numStates,numOutputFeatures,NumOutputsPerFeature):
    if not isinstance(NumOutputsPerFeature,list):
        NumOutputsPerFeature=numStates*[NumOutputsPerFeature]
        
    for cState in range(numStates):
        random.randint(0,100,(numOutputFeatures,NumOutputsPerFeature))
        