# from MyHMMscaled import myHMM
from MyHMMStateDistribution import myHMM, loadSequences
from matplotlib import pyplot as plt
from numpy import array,set_printoptions, save, seterr
import pandas
import time
import json
from numpyencoder import NumpyEncoder


# seterr(all='warn')
# import warnings
# warnings.filterwarnings('error')

fin = open('/data/git-ucf/ElectionViolence/RedditTrajectories/HMM_Analysis/initialGuess.json')
fin=open('TrainedIncelHMM_400.json','r')
for cLine in fin:
    HMMinit=json.loads(cLine)    
fin.close()

pi0=array(HMMinit['pi0'])
A=array(array(HMMinit['A']))


set_printoptions(precision=5)


stateSeqs,emissionSeqs=loadSequences('/data/git-ucf/ElectionViolence/RedditTrajectories/HMM_Analysis/userTrajSimple.njson')



HMM=myHMM(numStates=len(pi0),A=A,pi0=pi0)
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=HMMinit['b']['0'])
HMM.addEmission(emType='discrete',emDists=['Poisson'],params=HMMinit['b']['1'])

# cStart=time.time()
# optPaths=HMM.viterbi_pool(emissionSeqs)
# cEnd=time.time()
# print(cEnd-cStart)

for optimize in [['A','emission'],['pi0','A'],['emission','A'],['emission','pi0'],['A','emission']]:
    fit=HMM.train_pool(emissionSeqs,iterations=20,method='log',FracOrNum=.5,printHMM=['A','pi0','emission'],optimizeHMM=optimize,numCores=8 )
#fit=HMM.train(emissionSeqs,iterations=10,method='log',FracOrNum=.5,printHMM=['A','pi0','emission'])


# feat1=[[HMM.emission[0].state[cState].lamb for cState in range(HMM.numStates)]]
# feat2=[[HMM.emission[1].state[cState].lamb for cState in range(HMM.numStates)]]
feat1=[[HMM.emission[0].state[cState].lamb] for cState in range(HMM.numStates)]
feat2=[[HMM.emission[1].state[cState].lamb] for cState in range(HMM.numStates)]

TrainedHMM={'pi0':HMM.pi0.tolist(),'b':{'0':feat1,'1':feat2},'A':HMM.A.tolist()}
fout=open('TrainedIncelHMM_500.json','w')
json.dump(TrainedHMM, fout,cls=NumpyEncoder)
fout.close()

# # # fitness=fitness+fit
# # # plt.plot(fitness)