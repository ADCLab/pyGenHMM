from numpy import array, zeros, empty
from numpy import inf
from numpy import log
from numpy import nonzero
from random import shuffle
import warnings
from multiprocessing import Pool
from itertools import repeat
import os
from MyHMM_Calculations import calc_gamma, calc_logbeta, calc_logalpha, calc_xi, calc_alpha_scale1, calc_beta_scale1


def mini_batch_generator(AllYs, batch_size):
    """
    Generator function that yields mini-batches from a large list.

    :param AllYs: List of data items.
    :param batch_size: Size of each mini-batch.
    :yield: Mini-batch of data items.
    """
    for i in range(0, len(AllYs), batch_size):
        yield AllYs[i:i + batch_size]


def train_batch(self,allYs,iterations=20,method='log',numCores=6,FracOrNum=None,printHMM=[],optimizeHMM=['A','pi0','emission']):
    fitness=[]
    with Pool(numCores) as pool:            
        for iter in range(iterations):
            Number=len(allYs)
            if FracOrNum is None:
                Number=min(Number,1000)
            elif FracOrNum<=1:
                Number=round(FracOrNum*Number)
            elif FracOrNum is not None:
                Number=FracOrNum
            
            shuffle(allYs)
            # Ys=allYs[0:Number]        
            for Ys in mini_batch_generator(allYs, 200):
    
                # log_alpha_all = pool.starmap(calc_logalpha,zip(repeat(self.log_T),repeat(self.log_pi0),log_probObsState_all))
                # log_probObsState_pool = log_probObsState_all[0:Number]
                # log_alpha_pool = log_alpha_all[0:Number]
                # log_probObsState_pool = pool.map(self.calcLogProbObsState,Ys)
                # log_alpha_pool = pool.starmap(calc_logalpha,zip(repeat(self.log_T),repeat(self.log_pi0),log_probObsState_pool))
                if method=='log':
                    log_probObsState_pool = pool.map(self.calcProbObsState,Ys)         
                    log_alpha_pool = pool.starmap(calc_logalpha,zip(repeat(self.log_T),repeat(self.log_pi0),log_probObsState_pool))
                    log_beta_pool=pool.starmap(calc_logbeta,zip(repeat(self.log_T),log_probObsState_pool))
                    log_gamma_pool=pool.starmap(calc_gamma,zip(log_alpha_pool, log_beta_pool,repeat('log')))
                    log_xi_pool=pool.starmap(calc_xi,zip(repeat(self.log_T), log_alpha_pool, log_beta_pool, log_probObsState_pool,repeat('log')))
                    gamma_pool=log_gamma_pool
                    xi_pool=log_xi_pool
                else:
                    probObsState_pool= pool.starmap(self.calcProbObsState,zip(Ys,repeat(None)))
                    alpha_pool=pool.starmap(calc_alpha_scale1,zip(repeat(self.A),repeat(self.pi0),probObsState_pool,repeat(True)))
                    beta_pool=pool.starmap(calc_beta_scale1,zip(repeat(self.A),probObsState_pool,repeat(True)))
                    gamma_pool=pool.starmap(calc_gamma,zip(alpha_pool, beta_pool))
                    xi_pool=pool.starmap(calc_xi,zip(repeat(self.A), alpha_pool, beta_pool, probObsState_pool))  
                
            if 'emission' in optimizeHMM:
                [[self.emission[cFeature][cState].estimate([gamma[cState] for gamma in gamma_pool],[y[cFeature] for y in Ys],update=True) for cFeature in range(self.numFeatures)] for cState in range(self.numStates)]
                  
            # Estimate pi0
            # Normally this should be pi0_topPart/len(Ys)
            # Alternative calculation ensures pi0 sums to 1 and is a proper distribution.
            # Issue is related to numerical calculations
    
            if 'pi0' in optimizeHMM:  
                pi0_topPart=array([gamma[:,0] for gamma in gamma_pool]).sum(axis=0)
                pi0=pi0_topPart/pi0_topPart.sum()
                self.setPi0(pi0)
            
            # Estimate transition matrix A
            if 'A' in optimizeHMM:             
                T_topPart=array([xi.sum(axis=2) for xi in xi_pool]).sum(axis=0)
                T=(T_topPart.T/T_topPart.sum(axis=1)).T  
                self.setT(T)

        
            # Calculate fitness via log likelihood functions
            # with Pool(numCores) as pool:    
            #     log_probObsState_all = pool.map(self.calcProbObsState,allYs)         
            #     log_alpha_pool = pool.starmap(calc_logalpha,zip(repeat(self.log_T),repeat(self.log_pi0),log_probObsState_all))
            logProb=sum([log_alpha[:,-1].sum() for log_alpha in log_alpha_pool])
            fitness.append(logProb)
    
        
                
                # b_bottomPart=b_bottomPart+gamma.sum(axis=1)
    
    
    
            # os.system('clear')
            # print('= EPOCH #{} ='.format(iter))
            # self.printHMM(printHMM)
            print('Fitness:', logProb)

    return fitness


def train_pool(self,allYs,iterations=20,method='log',numCores=6,FracOrNum=None,printHMM=[],optimizeHMM=['A','pi0','emission']):
    fitness=[]

    with Pool(numCores) as pool: 
        for iter in range(iterations):
            Number=len(allYs)
            if FracOrNum is None:
                Number=min(Number,1000)
            elif FracOrNum<=1:
                Number=round(FracOrNum*Number)
            elif FracOrNum is not None:
                Number=FracOrNum
            
            shuffle(allYs)
            Ys=allYs[0:Number]
        
   
            if method=='log':
                log_probObsState_pool = pool.map(self.calcProbObsState,Ys)         
                log_alpha_pool = pool.starmap(calc_logalpha,zip(repeat(self.log_T),repeat(self.log_pi0),log_probObsState_pool))
                log_beta_pool=pool.starmap(calc_logbeta,zip(repeat(self.log_T),log_probObsState_pool))
                log_gamma_pool=pool.starmap(calc_gamma,zip(log_alpha_pool, log_beta_pool,repeat('log')))
                log_xi_pool=pool.starmap(calc_xi,zip(repeat(self.log_T), log_alpha_pool, log_beta_pool, log_probObsState_pool,repeat('log')))
                gamma_pool=log_gamma_pool
                xi_pool=log_xi_pool
            else:
                probObsState_pool= pool.starmap(self.calcProbObsState,zip(Ys,repeat(None)))
                alpha_pool=pool.starmap(calc_alpha_scale1,zip(repeat(self.A),repeat(self.pi0),probObsState_pool,repeat(True)))
                beta_pool=pool.starmap(calc_beta_scale1,zip(repeat(self.A),probObsState_pool,repeat(True)))
                gamma_pool=pool.starmap(calc_gamma,zip(alpha_pool, beta_pool))
                xi_pool=pool.starmap(calc_xi,zip(repeat(self.A), alpha_pool, beta_pool, probObsState_pool))  
            
            if 'emission' in optimizeHMM:
                [[self.emission[cFeature][cState].estimate([gamma[cState] for gamma in gamma_pool],[y[cFeature] for y in Ys],update=True) for cFeature in range(self.numFeatures)] for cState in range(self.numStates)]
                  
            # Estimate pi0
            # Normally this should be pi0_topPart/len(Ys)
            # Alternative calculation ensures pi0 sums to 1 and is a proper distribution.
            # Issue is related to numerical calculations
    
            if 'pi0' in optimizeHMM:  
                pi0_topPart=array([gamma[:,0] for gamma in gamma_pool]).sum(axis=0)
                pi0=pi0_topPart/pi0_topPart.sum()
                self.setPi0(pi0)
            
            # Estimate transition matrix A
            if 'A' in optimizeHMM:             
                T_topPart=array([xi.sum(axis=2) for xi in xi_pool]).sum(axis=0)
                T=(T_topPart.T/T_topPart.sum(axis=1)).T  
                self.setT(T)
    
            
            # Calculate fitness via log likelihood functions
            # with Pool(numCores) as pool:    
            #     log_probObsState_all = pool.map(self.calcProbObsState,allYs)         
            #     log_alpha_pool = pool.starmap(calc_logalpha,zip(repeat(self.log_T),repeat(self.log_pi0),log_probObsState_all))
            logProb=sum([log_alpha[:,-1].sum() for log_alpha in log_alpha_pool])
            fitness.append(logProb)

    
            
            # b_bottomPart=b_bottomPart+gamma.sum(axis=1)



            # os.system('clear')
            # print('= EPOCH #{} ='.format(iter))
            # self.printHMM(printHMM)
            print('Fitness:', logProb)

    return fitness


def train(self,allYs,iterations=20,method='log',FracOrNum=None,printHMM=[],optimizeHMM=['T','pi0','emission']):
    fitness=[]
    for iter in range(iterations):
        Number=len(allYs)
        if FracOrNum is None:
            Number=min(Number,1000)
        elif FracOrNum<=1:
            Number=round(FracOrNum*Number)
        elif FracOrNum is not None:
            Number=FracOrNum
        shuffle(allYs)
        Ys=allYs[0:Number]
        
        pi0_topPart=zeros((self.numStates))
        T_topPart=zeros((self.numStates,self.numStates))
        T_bottomPart=zeros((self.numStates))
        b_bottomPart=zeros((self.numStates))
        # b_topParts=[] [zeros((cEmission.emMat.shape)) for cEmission in self.emission]
        b_topParts=[]
        logProb=0
        log_alpha_all = []
        gamma_all = []
        
        for obs in Ys:
            # test=array([self.emission[cFeat].probObs(obs[cFeat]) for cFeat in range(self.numOutputFeatures)])
            # with warnings.catch_warnings():

            # Normally we would use the equation below.  However, there are can be numerical issues for long chains
            # log_probObsState=log(  array([[self.emission[cFeat][cState].probObs(obs[cFeat]) for cState in range(self.numStates)] for cFeat in range(self.numFeatures)])  ).sum(axis=0)    

            # As an alternative, any zero values are replaced with the next non-zero smallest term.  This allows for the log function to be taken of the resulting array         
            # Consider changing replaceZeros function with epsilon instead of 0.
            # log_probObsState=log(  replaceZeros(  array([[self.emission[cFeat][cState].probObs(obs[cFeat]) for cState in range(self.numStates)] for cFeat in range(self.numFeatures)]), self.epsilon  )  ).sum(axis=0)    
            log_probObsState = self.calcProbObsState(obs)
            log_alpha=calc_logalpha(self.log_T,self.log_pi0,log_probObsState)
            log_alpha_all.append(log_alpha)
            if method=='log':
                log_beta=calc_logbeta(self.log_T,log_probObsState)
                log_gamma=calc_gamma(log_alpha, log_beta,method='log') # Despite name of variable, it is not in the log-space, however, calculation is performed in log-space
                log_xi=calc_xi(self.log_T, log_alpha, log_beta, log_probObsState,method='log')
                gamma=log_gamma
                eta=log_xi
            else:
                probObsState=array([self.emission[cFeat].probObs(obs[cFeat]) for cFeat in range(self.numOutputFeatures)]).prod(axis=0)
                alpha=calc_alpha_scale1(self.A,self.pi0,probObsState,rescale=True)
                beta=calc_beta_scale1(self.A,probObsState,rescale=True)
                gamma=calc_gamma(alpha, beta)
                eta=calc_xi(self.A, alpha, beta, probObsState)

            # For estimating pi0                            
            pi0_topPart=pi0_topPart+gamma[:,0]

            # For estimating transition matrix                            
            T_topPart=T_topPart+eta.sum(axis=2)
            T_bottomPart=T_bottomPart+gamma[:,0:-1].sum(axis=1)

            # Calculate fitness via log likelihood functions            
            logProb=logProb+log_alpha[:,-1].sum()

            # Estimate emission matrix
            gamma_all.append(gamma)

        fitness.append(logProb)
        # Estimate emmission parameters for each states        
        # [self.emission[cState][cFeat].estimate(gamma_all,Ys) for cS
        if 'emission' in optimizeHMM:
            [[self.emission[cFeature][cState].estimate([gamma[cState] for gamma in gamma_all],[y[cFeature] for y in Ys],update=True) for cFeature in range(self.numFeatures)] for cState in range(self.numStates)]


        ## Normally this should be pi0_topPart/len(Ys)
        ## This ensures pi0 sumers to 0. 
        if 'pi0' in optimizeHMM:
            pi0=pi0_topPart/pi0_topPart.sum()
            self.setPi0(pi0)
        
        # Estimage Transition Matrix
        if 'T' in optimizeHMM:
            T=(T_topPart.T/T_topPart.sum(axis=1)).T               
            self.setT(T)


        print(logProb)
        # os.system('clear')
        # print('= EPOCH #{} ='.format(iter))
        # self.printHMM(printHMM)
        # print('Fitness:', logProb)
    return fitness