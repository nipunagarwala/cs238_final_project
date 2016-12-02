from __future__ import print_function
import numpy as np
import h5py
import copy


class HDF5Reader(object):
	def __init__(self, filePath):
		self.filePath = filePath

	def getData(self):
		with h5py.File(self.filePath,'r') as hf:
		    datasets = hf.keys()
		    print('List of arrays in this file:\n\t%s\n' %datasets)

		    actions = None
		    states = None
		    for dName in datasets:
		        data = hf.get(dName)
		        np_data = np.array(data)
		       	# print(np_data)
		        if dName == 'actions':
					actions = copy.deepcopy(np_data)
		        elif dName == 'states':
					states = copy.deepcopy(np_data)
		        print('[%s]: %s\n' %(dName,np_data.shape))

		    # stData = hf.get("states")
		    # np_data = np.array(stData)
		    # print(np_data[20,4,:,:])
		    return actions, states