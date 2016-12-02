import tensorflow as tf
import numpy as np
from utils import *
from utils import *
from layers import *
from goModels import *
from hdf5Reader import *
from constants import *
import random 

# '/data2/features.hdf5'


class PolicyNetworkAgent(CNNLayers):
	def __init__(self, batch_size):
		CNNLayers.__init__(self)
		self.batch_size = batch_size
		self.prev_train = False


	def createVariables(self, inShape, outShape):
		X = tf.placeholder("float", shape=inShape, name='inputPosition')
		Y = tf.placeholder("float", shape=outShape, name='outputAction')
		return X, Y

	def createPolicyAgent(self):
		inTrainShape = [None, BOARD_SZ, BOARD_SZ, DEPTH]
		outTrainShape = [None, NUM_ACTIONS]

		print("Defining input and output placeholder variables")
		self.inputTrainPos, self.trainLabel = self.createVariables(inTrainShape,outTrainShape)

		print("Creating Policy Network Model Object")
		self.network = PolicyNetwork(self.inputTrainPos,self.trainLabel,NUM_LAYERS, self.batch_size, num_filters=NUM_FILTERS,
								 learning_rate=1e-4, beta1=0.9, beta2=None, lmbda = CNN_REG_CONSTANTS, op='Rmsprop')
		
		print("Building the Policy Network Model")
		self.layerOuts, self.weights, self.biases = self.network.build_model()

		print("Setting up training policies and structure for Policy Network")
		self.cumCost, self.train_op, self.prob = self.network.train()
		return self.layerOuts, self.weights, self.biases, self.cumCost, self.train_op


	def trainAgent(self, inputData, inputLabels, numTrain, numEpochs):
		with tf.Session() as sess:
			print("Initializing variables in Tensorflow")
			if not self.prev_train:
				init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
				init_op.run()
				self.prev_train = True

			shuffData = inputData
			shuffLabels = inputLabels
			trainRange = range(0, numTrain)
			for epoch in range(0, numEpochs):
				random.shuffle(trainRange)
				shuffData = inputData[trainRange,:,:,:]
				shuffLabels = inputLabels[trainRange,:]
				for i in range(0,numTrain, self.batch_size):
					curBatch = inputData[range(i,i+ self.batch_size),:,:,:]
					curLabels = inputLabels[i:i+ self.batch_size, :]
					_, loss, probList = sess.run([self.train_op,self.cumCost, self.prob], feed_dict={self.inputTrainPos: curBatch,
																	self.trainLabel: curLabels})
					
					if i%1000 == 0:
						print("The current iteration is {}".format(i))
						print("The training loss of the current loss is: " + str(loss))
						print("This is the max probability of the output layer: {}".format(np.max(probList)))
						print("This is the min probability of the output layer: {}".format(np.min(probList)))


				print("Completed epoch {}".format(epoch))

			numCorrect = 0
			numIncorrect = 0
			for i in range(0,numTrain):
				curBatch = inputData[i,:,:,:]
				curLabels = inputLabels[i, :]
				pyx, loss = sess.run([self.layerOuts['pred'], self.cumCost], feed_dict={self.inputTrainPos: curBatch,
																self.trainLabel: curLabels})
				print("The current test iteration is: {}".format(i))
				prediction = tf.argmax(pyx)
				realOutput = np.where(curLabels == 1)[0]
				if prediction == realOutput:
					numCorrect +=1 
				else:
					numIncorrect += 1

			
			accuracy = (numCorrect)/(numCorrect + numIncorrect)
			print("The accuracy of this batch is {}".format(accuracy))

		return self.layerOuts, self.weights, self.biases



	def testAgentAccuracy(self, inputData, inputLabels, numTest):
		with tf.Session() as sess:
			if not self.prev_train:
				init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
				init_op.run()

			for i in range(0,numTest):
				curBatch = inputData[i,:,:,:]
				curLabels = inputLabels[i, :]
				pyx, loss = sess.run([self.layersOut['pred'], self.cumCost], feed_dict={self.inputTrainPos: curBatch,
																self.trainLabel: curLabels})
				print("The current test iteration is: {}".format(i))
				predictionLabels = np.zeros(pyx.shape)
				predictionLabels[pyx >= 0.5] = 1

				predValue = np.argmax(predictionLabels, axis=1)
				realOutput = np.where(curLabels == predValue)[0]
				if prediction == realOutput:
					numCorrect +=1 
				else:
					numIncorrect += 1

			
			accuracy = (numCorrect)/(numCorrect + numIncorrect)
			print("The accuracy of this batch is {}".format(accuracy))

	def trainSupervisedNetwork(self, filePath):
		print("Reading the input HDF5 File")
		self.hdf5Rd = HDF5Reader(filePath)

		print("Extracting data from the input HDF5 File")
		self.actions,self.states = (self.hdf5Rd).getData()

		self.numExamples = self.actions.shape[0]
		print(self.states.shape)
		self.states = np.transpose(self.states, axes=[0,2,3,1])
		actionsShape = self.actions.shape
		self.encodedActions = np.zeros((actionsShape[0],81))

		for i in range(0,actionsShape[0]):
			self.encodedActions[i,self.actions[i]] = 1 

		trainStatesBatch = self.states[:NUM_TRAIN_LARGE,:,:,:]
		trainLabelsBatch = self.encodedActions[:NUM_TRAIN_LARGE,:]
		layerOuts, weights, biases, cumCost, train_op = self.createPolicyAgent()
		self.trainAgent(trainStatesBatch, trainLabelsBatch,NUM_TRAIN_LARGE, 32)

	def updateWeights(self, updatedWeights, updatedBiases):
		with tf.Session() as sess:
			for i in range(0, NUM_LAYERS):
				updateOpWeight = self.weights['w'+str(i+1)].assign(updatedWeights['w'+str(i+1)])
				updateOpBias = self.biases['b'+str(i+1)].assign(updatedBiases['b'+str(i+1)])
				sess.run(updateOpWeight)
				sess.run(updateOpBias)




# class ValueNetworkAgent(CNNLayers):
# 	def __init__(self, batch_size):
# 		CNNLayers.__init__(self)
# 		self.batch_size = batch_size
# 		self.prev_train = False















# 	# Take current state as input, and then return an action prediction
# 	def make_move(self, state):
