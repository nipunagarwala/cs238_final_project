import tensorflow as tf
import numpy as np
from utils import *
from layers import *
from goModels import *
from hdf5Reader import *
from constants import *
from DataGen import *
import random 
import argparse
import os
from NeuralNetworkAgent import *

numGames = 1
numPlays = 100
numEpochs = 3
numUpdate = 20



def trainPolicyClassification(args):
	supervisedPolicyNetwork =  PolicyNetworkAgent(BATCH_SIZE)
	if args.train:
		layerOuts, weights, biases, cumCost, train_op = supervisedPolicyNetwork.trainSupervisedNetwork(args.trainpath, args.chkpointpath)
	elif args.test:
		supervisedPolicyNetwork.testSupervisedNetwork(args.testpath, args.chkpointpath)



def trainPolicyRL(args):
	global numGames
	global numPlays
	global numEpochs

	sess = tf.get_default_session()

	opponentModel = PolicyNetworkAgent(numGames)
	oppLayerOuts, oppWeights, oppBiases = None
	if args.chkpt:
		oppLayerOuts, oppWeights, oppBiases, oppBetas, oppScales, _, _ = opponentModel.createSLPolicyAgent()
		saver.restore(sess, args.chkpointpath)

	rlPolicyNetwork =  PolicyNetworkAgent(numGames)
	playerLayerOuts, playerWeights, playerBiases, playerBetas, playerScales, _, _ , _ = rlPolicyNetwork.createRLPolicyAgent()
	rlPolicyNetwork.updateWeights(oppWeights, oppBiases, oppBetas, oppScales)

	for i in range(0,numPlays):
		playoutList = RL_Playout(numGames, rlPolicyNetwork, filename=None, opponentModel=opponentModel, verbose=True, playbyplay=False)
		if playoutList[2] > 0:
			PostrainStatesBatch, PostrainLabelsBatch, _,_ = rlPolicyNetwork.preProcessInputs(playoutList[0], playoutList[1], playoutList[2])
			rlPolicyNetwork.trainAgent(PostrainStatesBatch, PostrainLabelsBatch, playoutList[2], numEpochs,playoutList[2], None, False)
		
		if playoutList[5] > 0:
			NegtrainStatesBatch, NegtrainLabelsBatch, _,_ = rlPolicyNetwork.preProcessInputs(playoutList[3], playoutList[4], playoutList[5])
			rlPolicyNetwork.trainAgent(NegtrainStatesBatch, NegtrainLabelsBatch, playoutList[5], numEpochs,playoutList[5], None, True)

		if i%numUpdate == 0:
			opponentModel.updateWeights(playerWeights, playerBiases,playerBetas, playerScales)



def extract_parser():
	parser = argparse.ArgumentParser(description='Train and Test mechanism for DeepGo')
	stage_group = parser.add_mutually_exclusive_group()
	train_group = parser.add_mutually_exclusive_group()
	stage_group.add_argument('--model', choices=['sl', 'rl'],
                        default='sl', help='Select model to run.')

	train_group.add_argument('--train', action="store_true", help='Training the model')
	train_group.add_argument('--test', action="store_true", help='Testing the model')
	train_group.add_argument('--chkpt', action="store_true", help='Loading from the checkpoint')

	parser.add_argument('--chkpointpath', default=None,
                        help='Path to the checkpoint file to load checkpoints. Default is None')
	parser.add_argument('--trainpath', default='/data/go/augmented/human700_augmented.hdf5',
                        help='Path to the training dataset')
	parser.add_argument('--testpath', default='/data/go/augmented/human700_augmented.hdf5',
                        help='Path to the training dataset')

	return parser.parse_args()

def main():
	sess = tf.Session()
	with sess.as_default():
		args = extract_parser()
		if args.model == 'sl':
			trainPolicyClassification(args)
		elif args.model == 'rl':
			trainPolicyRL(args)


	# filePath = '/data/go/augmented/human700_augmented.hdf5'
	# supervisedPolicyNetwork, layerOuts, weights, biases, cumCost, train_op, neg_train_op = trainPolicyClassification(filePath)






if __name__ == "__main__":
	main()