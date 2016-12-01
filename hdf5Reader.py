from __future__ import print_function
import numpy as np
import h5py


# class HDF5Reader(object):
# 	def __init__(self, fileName):
# 		self.fileName = fileName

# 	def getData(self):
with h5py.File('features.hdf5','r') as hf:
    datasets = hf.keys()
    print('List of arrays in this file:\n\t%s\n' %datasets)

    for dName in datasets:
        data = hf.get(dName)
        np_data = np.array(data)
        print('[%s]: %s\n' %(dName,np_data.shape))

    stData = hf.get("states")
    np_data = np.array(stData)
    print(np_data[20,4,:,:])