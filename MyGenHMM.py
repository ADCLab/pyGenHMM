# combined_class.py
import sys
from MyHMM_Base import myHMM  # Import the main class definition
from MyHMM_Setters import setT, setPi0  # Import additional method
from MyHMM_GenSequence import genSequences, genStateSequence, genOutputSequence
from MyHMM_AddEmission import addFeatureEmissions, calcProbObsState
from MyHMM_CalcEmission import probObsInStates, probFeatureObs, probFeatureStateObs
from MyHMM_Viterbi import viterbi, viterbi_exact
from MyHMM_Train import train,train_pool, train_batch

# System epsilon to resolve numerical computation issues
myHMM.epsilon = sys.float_info.epsilon

# Extend MyClass by adding method3
# For creating basic myHMM structure
myHMM.setT = setT
myHMM.setPi0 = setPi0
myHMM.addFeatureEmissions = addFeatureEmissions
myHMM.calcProbObsState = calcProbObsState

# For generating random data
myHMM.genSequences = genSequences
myHMM.genStateSequence = genStateSequence
myHMM.genOutputSequence = genOutputSequence

# For performing calculations related to estimation (viterbi) and training (baum-walch)
myHMM.probObsInStates = probObsInStates
myHMM.probFeatureObs = probFeatureObs
myHMM.probFeatureStateObs = probFeatureStateObs



myHMM.viterbi = viterbi
myHMM.viterbi_exact = viterbi_exact

myHMM.train = train
myHMM.train_pool = train_pool
myHMM.train_batch = train_batch
