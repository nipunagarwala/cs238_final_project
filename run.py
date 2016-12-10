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
numPlays = 100000
numEpochs = 1
numUpdate = 100
numOppAdd = 50



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
	oppList = []

	opponentModel = PolicyNetworkAgent(numGames)
	oppLayerOuts, oppWeights, oppBiases,oppBetas, oppScales = None, None, None, None, None
	if args.chkpt:
		oppLayerOuts, oppWeights, oppBiases, oppBetas, oppScales, _, _ = opponentModel.createSLPolicyAgent()
		saver = tf.train.Saver()
		saver.restore(sess, args.chkpointpath)
		print("Restored Supervised Policy Network from Checkpoint")

	slVarList = [oppWeights, oppBiases, oppBetas, oppScales]
	oppList.append(slVarList)
	rlPolicyNetwork =  PolicyNetworkAgent(numGames)
	print("Created Reinforcement Policy Network")
	playerLayerOuts, playerWeights, playerBiases, playerBetas, playerScales, _, _ , _ = rlPolicyNetwork.createRLPolicyAgent()
	# saver.restore(sess, RL_TRAIN_CHKPT)
	rlPolicyNetwork.updateWeights(oppWeights, oppBiases, oppBetas, oppScales)
	print("Updated weights for Reinforcement Policy Network")

	for i in range(0,numPlays):
		winPos, winActions, losePos, loseActions, winNum = RL_Playout(numGames, rlPolicyNetwork, filename=None, 
								opponentModel=opponentModel, verbose=True, playbyplay=False)

		loseNum = numGames - winNum
		print("Played another {} games with random opponent at iteration {}".format(numGames, i))
		if winNum > 0:
			indices = [i for i, x in enumerate(winActions) if x == 81 or x == 82]
			for index in sorted(indices, reverse=True):
				del winPos[index]
				del winActions[index]

			numTrainPos = len(winActions)
			winPosArray = np.asarray(winPos)
			winActArray = np.asarray(winActions)
			PostrainStatesBatch, PostrainLabelsBatch, _,_ = rlPolicyNetwork.preProcessInputs(winActArray,winPosArray, numTrainPos)
			curLayerOuts, curWeight, curBias, curBeta, curScale =  rlPolicyNetwork.trainAgent(PostrainStatesBatch, PostrainLabelsBatch, 
														numTrainPos, numEpochs,numTrainPos, None, False)
		
		if loseNum > 0:
			indices = [i for i, x in enumerate(loseActions) if x == 81 or x == 82]
			for index in sorted(indices, reverse=True):
				del losePos[index]
				del loseActions[index]

			numTrainPos = len(loseActions)
			losePosArray = np.asarray(losePos)
			loseActArray = np.asarray(loseActions)
			NegtrainStatesBatch, NegtrainLabelsBatch, _,_ = rlPolicyNetwork.preProcessInputs(loseActArray, losePosArray, numTrainPos)
			curLayerOuts, curWeight, curBias, curBeta, curScale =  rlPolicyNetwork.trainAgent(NegtrainStatesBatch, NegtrainLabelsBatch, 
														numTrainPos, numEpochs, numTrainPos, None, True)

		if (i+1)%numUpdate == 0:
			curOpp = [playerWeights, playerBiases,playerBetas, playerScales]
			oppList.append(curOpp)

		nextOpp = random.choice(oppList)
		opponentModel.updateWeights(nextOpp[0], nextOpp[1],nextOpp[2], nextOpp[3])
		print("Updated weights for Opponent model")

		if (i+1)%100 == 0:
			saver.save(sess, 'rl_random_self_training', global_step=i+1)
			print("Saved checkpoint for epoch {}".format(i+1))

# def createValueData(args):
# 	sess = tf.get_default_session()
# 	slModel = PolicyNetworkAgent(1)
# 	rlModel = PolicyNetworkAgent(1)

# 	saver_sl = tf.train.Saver()

# 	slLayerOuts, slWeights, slBiases, slBetas, slScales, _, _ = slModel.createSLPolicyAgent()
# 	saver_sl.restore(sess, SL_CHECKPOINT)
# 	print("Restored Supervised Policy Network from Checkpoint")

# 	saver_rl = tf.train.Saver()
# 	rlLayerOuts, rlWeights, rlBiases, rlBetas, rlScales, _, _ , _ = rlModel.createRLPolicyAgent()
# 	saver_rl.restore(sess, RL_CHECKPOINT)

# 	states,rewards = Value_Playout(NUM_VAL_GAMES, slModel, rlModel, filename='valueNetworkData', U_MAX=90, verbose=False, playbyplay=False)









def extract_parser():
	parser = argparse.ArgumentParser(description='Train and Test mechanism for DeepGo')
	stage_group = parser.add_mutually_exclusive_group()
	train_group = parser.add_mutually_exclusive_group()
	stage_group.add_argument('--model', choices=['sl', 'rl', 'value'],
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
		# elif args.model == 'value':
		# 	createValueData(args)


	# filePath = '/data/go/augmented/human700_augmented.hdf5'
	# supervisedPolicyNetwork, layerOuts, weights, biases, cumCost, train_op, neg_train_op = trainPolicyClassification(filePath)






if __name__ == "__main__":
	main()