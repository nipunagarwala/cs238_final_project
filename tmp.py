from hdf5Reader import HDF5Reader
import tensorflow as tf
from graph_saver_api import *
import numpy as np

# prepare the sample
filePath = '/data/go/augmented/human700_augmented.hdf5'
print("Reading the input HDF5 File")
hdf5Rd = HDF5Reader(filePath)

print("Extracting data from the input HDF5 File")
inputLabels, inputData = hdf5Rd.getData()

batch_size = 16

# get the session ready
graph = load_graph('/home/yinoue/go/cs238_final_project/human-aug-model-25_frozen_model.pb')

# # We can verify that we can access to the list of operations in the graph
# for op in graph.get_operations():
# 	if 'input' in op.name:
# 	    print(op.name)
#     # prefix/Placeholder/inputs_placeholder
#     # ...
#     # prefix/Accuracy/predictions

# We access the input and output nodes 
x = graph.get_tensor_by_name('prefix/inputPosition:0')
y = graph.get_tensor_by_name('prefix/Softmax:0')

# We launch a Session
with tf.Session(graph=graph) as sess:
	# Note: we didn't initialize/restore anything, everything is stored in the graph_def
	for i in range(10):
		tt = i*batch_size
		curBatch = inputData[tt:tt+batch_size,:,:,:]
		curBatch = curBatch.reshape(batch_size,9,9,48)
		curLabels = inputLabels[tt:tt+batch_size]

		y_out = sess.run(y, feed_dict={
			x: curBatch # < 45
		})

		print '-'*20
		print np.argmax(y_out,axis=1)
		print y_out[:,46:49]
		print curLabels
