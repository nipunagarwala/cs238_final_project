from constants import *
from NeuralNetworkAgent import *

import os

def testTrainAccuracies(chkptDir, dataPath, outputFilePath):
	f = open(outputFilePath, 'w')
	f.write('Filename,Testing Accuracy,Training Accuracy')

	testFiles = os.listdir(chkptDir)
	chkptFiles = [f for f in testFiles if not f.endswith('.meta')]
	chkptFiles = sorted(chkptFiles)

	slModel =  PolicyNetworkAgent(BATCH_SIZE)

	trainStatesBatch, trainLabelsBatch, testStatesBatch, testLabelsBatch = slModel.readInputs(dataPath)
	for chkpt in chkptFiles:
		print 'Loading checkpoint %s' % f

		layerOuts, weights, biases, betas, scales, cumCost, train_op = slModel.createSLPolicyAgent()

		testAccuracy = slModel.testAgentAccuracy(trainStatesBatch, trainLabelsBatch,
										  	 	 trainStatesBatch.shape[0], 
							   			  	 	 slModel.batch_size, chkpt)

		testAccuracy = slModel.testAgentAccuracy(testStatesBatch, testLabelsBatch,
										  	 	 testStatesBatch.shape[0], 
							   			  	 	 slModel.batch_size, chkpt)

		print '\tTraining Error: %f, Testing Error: %f' %(trainAccracy,testAccuracy)
		f.write('%s,%f,%f' % (chkpt,testAccuracy,trainAccracy))

	f.close()

def playPachi(chkptDir, gamesPerModel, outputFilePath):
	f = open(outputFilePath, 'w')
	f.write('Filename,Win Percentage')

	testFiles = os.listdir(chkptDir)
	chkptFiles = [f for f in testFiles if not f.endswith('.meta')]
	chkptFiles = sorted(chkptFiles)

	rlModel =  PolicyNetworkAgent(BATCH_SIZE)

	for chkpt in chkptFiles:
		print 'Loading checkpoint %s' % f

		layerOuts, weights, biases, betas, scales, cumCost, train_op = rlModel.createSLPolicyAgent()


		_, _, _, _, winNum = RL_Playout(gamesPerModel, rlModel, filename=None, opponentModel=None, 
										doRecord=False, verbose=False, playbyplay=False)

		winRate = float(winNum)/gamesPerModel
		print '\tWin rate: %f' % winRate
		f.write('%s,%f' % (chkpt,winRate))

	f.close()

if __name__ == "__main__":
	testTrainAccuracies('./tmpchkpt','/data/go/augmented/human700_augmented.hdf5','result.csv')