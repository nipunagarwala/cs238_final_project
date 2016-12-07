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



def trainPolicyClassification(filePath):
	supervisedPolicyNetwork =  PolicyNetworkAgent(BATCH_SIZE)
	supervisedPolicyNetwork.trainSupervisedNetwork('/data2/actionFixed/human700pachi500_actionFixed.hdf5')
	return supervisedPolicyNetwork



def trainPolicyRL(opponentModel):
	global numGames
	rlPolicyNetwork =  PolicyNetworkAgent(numGames)
	
	layerOuts, weights, biases, cumCost, train_op = rlPolicyNetwork.createPolicyAgent()
	states,actions,rewards = RL_Playout(numGames, rlPolicyNetwork, filename=None, opponentModel=opponentModel, verbose=True, playbyplay=False)





def main():
	'''
	trainPolicyClassification()

	'''






if __name__ == "__main__":
	main()