import tensorflow as tf
import numpy as np
from utils import *
from utils import *
from layers import *
from hdf5Reader import *
from constants import *
import random 
from NeuralNetworkAgent import *




def main():
	supervisedPolicyNetwork =  PolicyNetworkAgent(BATCH_SIZE)
	supervisedPolicyNetwork.trainSupervisedNetwork('/data2/human450.hdf5')



if __name__ == "__main__":
	main()