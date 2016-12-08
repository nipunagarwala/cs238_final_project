import tensorflow as tf
import numpy as np
from utils import *
from layers import *
from goModels import *
from hdf5Reader import *
from constants import *
import random 
np.set_printoptions(threshold='nan')

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

	def createSLPolicyAgent(self):
		inTrainShape = [None, BOARD_SZ, BOARD_SZ, NUM_FEATURES]
		outTrainShape = [None, NUM_ACTIONS]

		print("Defining input and output placeholder variables")
		self.inputTrainPos, self.trainLabel = self.createVariables(inTrainShape,outTrainShape)

		print("Creating Policy Network Model Object")
		self.network = PolicyNetwork(self.inputTrainPos,self.trainLabel,NUM_LAYERS, self.batch_size, num_filters=NUM_FILTERS,
								 learning_rate=1e-4, beta1=0.9, beta2=None, lmbda = CNN_REG_CONSTANTS, op='Rmsprop')
		
		print("Building the Policy Network Model")
		self.layerOuts, self.weights, self.biases, self.betas, self.scales = self.network.build_model()

		print("Setting up training policies and structure for Policy Network")
		self.cumCost, self.train_op, self.prob, self.action, self.actionMean, self.realLabels = self.network.SLtrain()
		return self.layerOuts, self.weights, self.biases,self.betas, self.scales, self.cumCost, self.train_op

	def createRLPolicyAgent(self):
		inTrainShape = [None, BOARD_SZ, BOARD_SZ, NUM_FEATURES]
		outTrainShape = [None, NUM_ACTIONS]

		print("Defining input and output placeholder variables")
		self.inputTrainPos, self.trainLabel = self.createVariables(inTrainShape,outTrainShape)

		print("Creating Policy Network Model Object")
		self.network = PolicyNetwork(self.inputTrainPos,self.trainLabel,NUM_LAYERS, self.batch_size, num_filters=NUM_FILTERS,
								 learning_rate=1e-4, beta1=0.9, beta2=None, lmbda = CNN_REG_CONSTANTS, op='Rmsprop')
		
		print("Building the Policy Network Model")
		self.layerOuts, self.weights, self.biases,self.betas, self.scales = self.network.build_model()

		print("Setting up training policies and structure for Policy Network")
		self.cumCost, self.train_op, self.neg_train_op, self.prob, self.action, self.actionMean, self.realLabels = self.network.RLtrain()
		return self.layerOuts, self.weights, self.biases, self.betas, self.scales,self.cumCost, self.train_op, self.neg_train_op


	def trainAgent(self, inputData, inputLabels, numTrain, numEpochs, cur_batch_size, chkptFile=None, negTrain=False):
		saver = tf.train.Saver()
		sess = tf.get_default_session()
		print("Initializing variables in Tensorflow")
		if not self.prev_train:
			init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
			init_op.run()
			self.prev_train = True
		if chkptFile is not None:
			saver = tf.train.Saver()
			saver.restore(sess, chkptFile)
			print("Fetching checkpoint data from {}".format(chkptFile))

		# all_vars = tf.trainable_variables()
		# for v in all_vars:
		# 	print(v.name)

		shuffData = inputData
		shuffLabels = inputLabels
		trainRange = range(0, numTrain)
		for epoch in range(0, numEpochs):
			random.shuffle(trainRange)
			shuffData = inputData[trainRange,:,:,:]
			shuffLabels = inputLabels[trainRange,:]
			for i in range(0,numTrain, cur_batch_size):
				curBatch = inputData[range(i,i+ cur_batch_size),:,:,:]
				curLabels = inputLabels[i:i+ cur_batch_size, :]
				curOp = self.train_op
				if negTrain:
					curOp = self.neg_train_op
				_, loss, probList, actionVal, actMeanVal,labelVal = sess.run([curOp,self.cumCost, self.prob, self.action, self.actionMean,self.realLabels],
							 feed_dict={self.inputTrainPos: curBatch, self.trainLabel: curLabels})
				
				if i%1000 == 0:
					print("The current iteration is {}".format(i))
					print("The training loss of the current loss is: " + str(loss))
					print("The shape of the probability distribution is: {}".format( probList.shape))
					print("This is the max probability of the output layer: {}".format(np.amax(probList, axis=1)))
					print("This is the min probability of the output layer: {}".format(np.amin(probList, axis=1)))
					# print("These are the action labels: {}".format(actionVal))
					#print("This is the reducedsum for action prediction: {}".format(actMeanVal))
					#print("These are the real labels: {}".format(labelVal))
					predValue = np.argmax(probList, axis=1)
					realOutput = np.argmax(labelVal, axis=1)
					correctOut = np.equal(realOutput, predValue)
					numCorrect = np.sum(correctOut) 
					accuracy = float((numCorrect))/cur_batch_size
					print("The accuracy of this batch is {}".format(accuracy))


			if (epoch+1)%2 == 0:
				saver.save(sess, 'human-aug-pachi5000', global_step=epoch+1)
				print("Saved checkpoint for epoch {}".format(epoch+1))


			print("Completed epoch {}".format(epoch))

		numCorrect = 0
		total = 0
		for i in range(0,numTrain,cur_batch_size):
			curBatch = inputData[range(i,i+ cur_batch_size),:,:,:]
			curLabels = inputLabels[range(i,i+ cur_batch_size), :]
			pyx, loss = sess.run([self.layerOuts['pred'], self.cumCost], feed_dict={self.inputTrainPos: curBatch,
															self.trainLabel: curLabels})
			print("The current test iteration is: {}".format(i))

			predValue = np.argmax(pyx, axis=1)
			realOutput = np.argmax(curLabels, axis=1)
			correctOut = np.equal(realOutput, predValue)
			numCorrect += np.sum(correctOut) 
			total += cur_batch_size

		
		accuracy = float((numCorrect))/float(total)
		print("The accuracy of this batch is {}".format(accuracy))

		# sess = tf.Session()
		# new_saver = tf.train.import_meta_graph('test-model.meta')
		# new_saver.restore(sess, tf.train.latest_checkpoint('./'))
		# all_vars = tf.trainable_variables()
		# for v in all_vars:
		#     print(v.name)

		return self.layerOuts, self.weights, self.biases, self.betas, self.scales



	def testAgentAccuracy(self, inputData, inputLabels, numTest,cur_batch_size, chkptFile):
		total = 0
		numCorrect = 0
		saver = tf.train.Saver()
		sess = tf.get_default_session()
		# init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
		# init_op.run()
		# new_saver = tf.train.import_meta_graph(metaFile)
		if  chkptFile is not None:
			saver.restore(sess, chkptFile)
			print("Fetching checkpoint data from {}".format(chkptFile))

		all_vars = tf.trainable_variables()
		for v in all_vars:
		    print(v.name)

		for i in range(0,numTest-cur_batch_size, cur_batch_size):
			curBatch = inputData[range(i,i+ cur_batch_size),:,:,:].reshape((cur_batch_size,BOARD_SZ,BOARD_SZ,NUM_FEATURES))
			curLabels = inputLabels[range(i,i+ cur_batch_size), :].reshape((cur_batch_size,NUM_ACTIONS))
			pyx, loss = sess.run([self.layerOuts['pred'], self.cumCost], feed_dict={self.inputTrainPos: curBatch,
															self.trainLabel: curLabels})
			
			if i%1000 == 0:
				print("The current test iteration is: {}".format(i))

			# print("This are the predicted values: {}".format(np.amax(pyx, axis=1)))
			predValue = np.argmax(pyx, axis=1)
			realOutput = np.argmax(curLabels, axis=1)
			correctOut = np.equal(realOutput, predValue)
			numCorrect += np.sum(correctOut) 
			total += cur_batch_size


		
		accuracy = float((numCorrect))/float((total))
		print("The accuracy of this batch is {}".format(accuracy))


	def trainSupervisedNetwork(self, filePath, chkptFile):
		trainStatesBatch, trainLabelsBatch, testStatesBatch, testLabelsBatch = self.readInputs(filePath)
		layerOuts, weights, biases, betas, scales, cumCost, train_op = self.createSLPolicyAgent()
		self.trainAgent(trainStatesBatch, trainLabelsBatch,NUM_TRAIN_LARGE, NUM_EPOCHS, self.batch_size, chkptFile)
		# self.testAgentAccuracy(testStatesBatch, testLabelsBatch,testStatesBatch.shape[0], './human-aug-model-25.meta', './human-aug-model-25' )

		return layerOuts, weights, biases, cumCost, train_op

	def testSupervisedNetwork(self, filePath, chkptFile):
		trainStatesBatch, trainLabelsBatch, testStatesBatch, testLabelsBatch = self.readInputs(filePath)
		layerOuts, weights, biases, betas, scales, cumCost, train_op = self.createSLPolicyAgent()
		self.testAgentAccuracy(testStatesBatch, testLabelsBatch,testStatesBatch.shape[0], self.batch_size, chkptFile)

	# 	# Take current state as input, and then return an action prediction
	def make_move(self, state):
		sess = tf.get_default_session()
		pyx = sess.run([self.layerOuts['pred']], feed_dict={self.inputTrainPos: state})
		return pyx



	def updateWeights(self, updatedWeights, updatedBiases, updatedBetas, updatedScales):
		sess = tf.get_default_session()
		for i in range(0, NUM_LAYERS):
			updateOpWeight = self.weights['w'+str(i+1)].assign(updatedWeights['w'+str(i+1)])
			updateOpBias = self.biases['b'+str(i+1)].assign(updatedBiases['b'+str(i+1)])
			if i < (NUM_LAYERS - 1):
				updatedOpBetas =  self.betas['beta'+str(i+1)].assign(updatedBetas['beta'+str(i+1)])
				updatedOpScales =  self.scales['scale'+str(i+1)].assign(updatedScales['scale'+str(i+1)])
				sess.run(updatedOpBetas)
				sess.run(updatedOpScales)

			sess.run(updateOpWeight)
			sess.run(updateOpBias)



	def readInputs(self, filePath):
		print("Reading the input HDF5 File")
		self.hdf5Rd = HDF5Reader(filePath)

		print("Extracting data from the input HDF5 File")
		inActions,inStates = (self.hdf5Rd).getData()
		pachiFile = HDF5Reader('/data/go/augmented/pachi10000_dense_pos_60000_augmented.hdf5')
		inActPachi,inStatesPachi = pachiFile.getData()
		totActions = np.hstack((inActPachi,inActions))
		totStates = np.vstack((inStatesPachi,inStates ))
		print("Combined Pachi Data also.....")
		return self.preProcessInputs(totActions,totStates, NUM_TRAIN_LARGE)

	def preProcessInputs(self, inActions,inStates, numTrain):
		numExamples = inStates.shape[0]
		print(inStates.shape)
		transStates = np.transpose(inStates, axes=[0,2,3,1])
		actionsShape = inActions.shape
		encodedActions = np.zeros((actionsShape[0],81))

		for i in range(0,actionsShape[0]):
			encodedActions[i,inActions[i]] = 1 

		trainStatesBatch = transStates[:numTrain,:,:,:NUM_FEATURES]
		trainLabelsBatch = encodedActions[:numTrain,:]
		testStatesBatch = transStates[numTrain:,:,:,:NUM_FEATURES]
		testLabelsBatch = encodedActions[numTrain:,:]

		return trainStatesBatch, trainLabelsBatch, testStatesBatch, testLabelsBatch



# class ValueNetworkAgent(CNNLayers):
# 	def __init__(self, batch_size):
# 		CNNLayers.__init__(self)
# 		self.batch_size = batch_size
# 		self.prev_train = False












