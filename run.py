import tensorflow as tf
import numpy as np
from utils import *
from layers import *
from goModels import *
from hdf5Reader import *
from constants import *
from DataGen import *
import random 

numGames = 10
numPlays = 100
numEpochs = 3



def trainPolicyClassification(filePath):
	supervisedPolicyNetwork =  PolicyNetworkAgent(BATCH_SIZE)
	layerOuts, weights, biases, cumCost, train_op, neg_train_op = supervisedPolicyNetwork.trainSupervisedNetwork('/data2/actionFixed/human700pachi500_actionFixed.hdf5')
	return supervisedPolicyNetwork, layerOuts, weights, biases, cumCost, train_op, neg_train_op



def trainPolicyRL(opponentModel,oppWeights, oppBiases):
	global numGames
	global numPlays
	global numEpochs
	rlPolicyNetwork =  PolicyNetworkAgent(numGames)
	
	layerOuts, playerWeights, playerBiases, cumCost, train_op = rlPolicyNetwork.createPolicyAgent()
	rlPolicyNetwork.updateWeights(oppWeights, oppBiases)

	for i in range(0,numPlays):
		playoutList = RL_Playout(numGames, rlPolicyNetwork, filename=None, opponentModel=opponentModel, verbose=True, playbyplay=False)
		trainStatesBatch, trainLabelsBatch, testStatesBatch, testLabelsBatch = rlPolicyNetwork.preProcessInputs(inStates, inActions, numGames)
		rlPolicyNetwork.trainAgent(trainStatesBatch, trainLabelsBatch, numGames, numEpochs)

		if i%numUpdate == 0:
			opponentModel.updateWeights(playerWeights, playerBiases)


def main():

	filePath = '/data/go/augmented/human700_augmented.hdf5'
	supervisedPolicyNetwork, layerOuts, weights, biases, cumCost, train_op, neg_train_op = trainPolicyClassification(filePath)






if __name__ == "__main__":
	main()