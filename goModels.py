import tensorflow as tf
import numpy as np
from utils import *
from goUtils import *
from layers import *


class PolicyNetwork(CNNLayers):
	def __init__(self, train, data_list,input_shape, output_shape, num_filters, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, lmbda = None, op='Rmsprop'):
        CNNLayers.__init__(self)
		self.input = tf.placeholder("float", input_shape)
		self.output = tf.placeholder("float", output_shape)
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.num_filters = num_filters
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.strides = [1,1,1,1]
		self.filters = [3,3,num_filters, num_filters]
		self.num_layers = 13
		self.beta1 = beta1
		self.beta2 = beta2
		self.lmbda = lmbda
		self.op = op

	def build_model(self):
		layerOuts = {}
		weights = {}

		layersOut['input'] = self.input
        layersOut['output'] = self.output
        prev_layer = self.input
        prev_shape = (prev_layer.get_shape().as_list())[1]

        layersOut['layer1'], weights['w1'] = self.conv_layer(self, self.input, [5,5,1, self.num_filters], self.strides, 'layer1',
        											 num_dim = '2d', padding='SAME',if_relu = True, batchNorm = False)

        for i in range(1,num_layers):
        	layersOut['layer'+str(i+1)], weights['w'+str(i+1)] = self.conv_layer(self, layersOut['layer'+str(i)], self.filters, self.strides, 
        											'layer'+str(i+1), num_dim = '2d', padding='SAME',if_relu = True, batchNorm = False)
		
		
		layersOut['layer'+str(num_layers)], weights['w'+str(num_layers)] = self.conv_layer(self, layersOut['layer'+str(num_layers-1)],
							 [1,1,self.num_filters, 1], self.strides, 'layer'+str(num_layers), num_dim = '2d',
							  padding='SAME',if_relu = True, batchNorm = False)


		fcShapeConv = layersOut['layer'+str(num_layers)].get_shape().as_list()
        numParams = reduce(lambda x, y: x*y, fcShapeConv)
		layersOut['pred'] = tf.reshape(layersOut['layer'+str(num_layers)], [1, numParams])



        self.layersOut = layersOut
        self.weights = weights

        return layersOut, weights


	def train(self):
		cost = self.cost_function( self.layersOut['pred'], self.output, op='softmax')
        cumCost = cost
        numEntries = len(self.weights)

        if self.lmbda not None:
			weightVals = self.weights.values()
			for i in range(numEntries):
			    cumCost = self.add_regularization( cumCost, weightVals[i], self.lmbda[i], None, op='l2')

        train_op = self.minimization_function(cumCost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cumCost, train_op


class ValueNetwork(CNNLayers):
	def __init__(self, train, data_list,input_shape, output_shape, num_filters, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, lmbda = None, op='Rmsprop'):
    	CNNLayers.__init__(self)
		self.input = tf.placeholder("float", input_shape)
		self.output = tf.placeholder("float", output_shape)
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.num_filters = num_filters
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.strides = [1,1,1,1]
		self.filters = [3,3,num_filters, num_filters]
		self.hidden_units = 256
		self.num_layers = 15
		self.beta1 = beta1
		self.beta2 = beta2
		self.lmbda = lmbda
		self.op = op


	def build_model(self):
		layerOut = {}
		weights = {}

		layersOut['input'] = self.input
        layersOut['output'] = self.output
        prev_layer = self.input
        prev_shape = (prev_layer.get_shape().as_list())[1]

        # The 1st layer was not described in the paper, and hence this is just a tentative layer subject to change
        layersOut['layer1'], weights['w1'] = self.conv_layer(self, self.input, [5,5,1, self.num_filters], self.strides, 'layer1',
        											 num_dim = '2d', padding='SAME',if_relu = True, batchNorm = False)

        ''' As per the AlphaGo paper, layers 2-11 is exactly the same as the Policy Network. The 12th layer is an additional
        Convolutional layer, which is not described. Hence, we imitate the 2-11 layers like in the Policy Network. As a result,
        we iterate until num_layers-3 (12) instead of num_layers - 4 (11).
        '''
        for i in range(1,num_layers-2):
        	layersOut['layer'+str(i+1)], weights['w'+str(i+1)] = self.conv_layer(self, layersOut['layer'+str(i)], self.filters, self.strides, 
        											'layer'+str(i+1), num_dim = '2d', padding='SAME',if_relu = True, batchNorm = False)
		
		
		layersOut['layer'+str(num_layers-2)], weights['w'+str(num_layers-2)] = self.conv_layer(self, layersOut['layer'+str(num_layers-3)],
							 [1,1,self.num_filters, self.num_filters], self.strides, 'layer'+str(num_layers-2), num_dim = '2d',
							  padding='SAME',if_relu = True, batchNorm = False)


		fcShapeConv = layersOut['layer'+str(num_layers-2)].get_shape().as_list()
        numParams = reduce(lambda x, y: x*y, fcShapeConv)
		layersOut['layer'+str(num_layers-2)+'-fc'] = tf.reshape(layersOut['layer'+str(num_layers-2)], [1, numParams])

		layersOut['layer'+str(num_layers-1)], weights['w'+str(num_layers-1)] = self.fcLayer(layersOut['layer'+str(num_layers-2)+'-fc'], 
																				[ numParams, self.hidden_units], True, False)
		

		# Output (Regression/value prediction) layer. Need a tanh activation to map output between 1 and -1, for corresponding rewards
		layersOut['layer'+str(num_layers)], weights['w'+str(num_layers)] = self.fcLayer(layersOut['layer'+str(num_layers-1)], 
																				[ self.hidden_units, 1], False, False)

		layersOut['layer'+str(num_layers)] = self.tanh(layersOut['layer'+str(num_layers)])

		layerOut['pred'] = layersOut['layer'+str(num_layers)]


        self.layersOut = layersOut
        self.weights = weights

        return layersOut, weights



	def train(self):
		cost = self.cost_function( self.layersOut['pred'], self.output, op='square')
        cumCost = cost
        numEntries = len(self.weights)

        if self.lmbda not None:
			weightVals = self.weights.values()
			for i in range(numEntries):
			    cumCost = self.add_regularization( cumCost, weightVals[i], self.lmbda[i], None, op='l2')

        train_op = self.minimization_function(cumCost, self.learning_rate, self.beta1, self.beta2, self.op)
        return cumCost, train_op


