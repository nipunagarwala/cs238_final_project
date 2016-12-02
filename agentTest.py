import tensorflow as tf
import numpy as np
from utils import *
from utils import *
from layers import *
from hdf5Reader import *
from constants import *
import random 
import NeuralNetworkAgent as NNA




def main():
	supervisedPolicyNetwork =  NNA.NeuralNetworkAgent(BATCH_SIZE)
	supervisedPolicyNetwork.trainSupervisedNetwork('/data2/features_augmented.hdf5')








if __name__ == "__main__":
	main()