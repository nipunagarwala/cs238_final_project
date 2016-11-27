import tensorflow as tf
import numpy as np
from utils import *
from layers import *
import copy


class SupervisedLearningNN(CNNLayers):
	def __init__(self, train, data_list,input_shape, output_shape, batch_size=1, learning_rate=1e-3, beta1=0.99, beta2=0.99, lmbda = None, op='Rmsprop'):
        CNNLayers.__init__(self)
		_, self.input, self.output, self.p_keep = self.createVariables(train, data_list, batch_size)
		self.input = tf.reshape(self.input, [batch_size,29 ], name=None)
		self.output = tf.reshape(self.output, [batch_size,1 ], name=None)
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.lmbda = lmbda
		self.op = op

	def build_model(self):



	def train(self):
		

