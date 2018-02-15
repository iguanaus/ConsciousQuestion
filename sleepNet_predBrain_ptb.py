from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

from ptb_iterator import *

import numpy as np
import argparse, os
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import *
import matplotlib.pyplot as plt
#from tensorflow.python.ops import init_ops
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell
import pandas as pd
import re

  
#THis needs to take in the data, then return the data in a list of
#The output should be in a np array form. Note that the y value doesn't have to be returned. 
def packageForRNN(data,seq_length,batch_size,num_epochs):
	#Deal with batch size >.<
	#print(data)
	#Numsteps is your sequence length. In this case the earlier formula. 
	def gen_epochs(n,numsteps, batch_size):
		for i in xrange(n):
			yield ptb_iterator(data,batch_size,numsteps)
	print("My sequence length: ", seq_length)
	myepochs = gen_epochs(num_epochs,seq_length,batch_size)
	return myepochs


def file_data(filename="snippet-25_D_downsampled_20_numpy.npy"):
	data = np.load(filename)
	df = pd.DataFrame(data)
	#print(df.head(5))
	#Now normalize and all
	normData = (df - df.mean())/df.std()

	retVal = normData.as_matrix().astype(float)


	return retVal

def main():
	# --- Set data params ----------------
	#Create Data
	max_len_data = 1000000000
	seq_len = 50
	batch_size = 50
	num_epochs = 200



	tempData = file_data()
	#Now prep it. 
	data = packageForRNN(tempData,seq_len,batch_size,num_epochs)

	n_input = 127
	print("number input: " , n_input)

	n_output = n_input
	n_hidden = 100
	learning_rate = 0.001
	decay = 0.9
	reuse = False
	
	#Structure of this will be [weekday,seconds*1000,intPrice,volume]

	X = tf.placeholder("float32",[None,seq_len,n_input])
	Y = tf.placeholder("float32",[None,seq_len,n_input])

	# Input to hidden layer
	cell = None
	h = None
	num_layers = 3
	#h_b = None
	sequence_length = [seq_len] * 1


	cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)

	cells = tf.contrib.rnn.MultiRNNCell([BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias =1) for _ in range(num_layers)],state_is_tuple=True)

	#cells = core_rnn_cell_impl.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
	if h == None:
		h = cells.zero_state(1,tf.float32)

	hidden_out, states = tf.nn.dynamic_rnn(cells, X, sequence_length=sequence_length, dtype=tf.float32,initial_state=h)


	# if h == None:
	# 	h = cell.zero_state(1,tf.int32)
	# hidden_out, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


	# Hidden Layer to Output
	V_init_val = np.sqrt(6.)/np.sqrt(n_output + n_input)

	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_output], \
			dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
	V_bias = tf.get_variable("V_bias", shape=[n_output], \
			dtype=tf.float32, initializer=tf.constant_initializer(0.01))

	hidden_out_list = tf.unstack(hidden_out, axis=1)
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias)


	# define evaluate process
	print("Output data: " , output_data)
	print("Labels: " , Y)

	cost = tf.reduce_mean(tf.square(Y-output_data))

	#correct_pred = tf.equal(tf.round(output_data*standardDev+meanVal), tf.round(Y*standardDev+meanVal))
	#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# --- Initialization ----------------------
	global_step = tf.Variable(0, trainable=False)

	learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                           5000, 0.9, staircase=True)
	optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(cost)
	init = tf.global_variables_initializer()

	for i in tf.global_variables():
		print(i.name)

	'''

	def do_validation(f2,data,curEpoch):
		maxIter = int((len(data)/seq_len)-1.0)
		val_losses = []
		training_state = None

		for i in xrange(1,maxIter):
			#Batch sizes of 30. 
			myTrain_x = data[seq_len*i:seq_len*(i+1),0:2].reshape((1,seq_len,2))
			myTrain_y = data[seq_len*i+1:seq_len*(i+1)+1,2:3].reshape((1,seq_len))
			myfeed_dict={X: myTrain_x, Y: myTrain_y}
			if training_state is not None:
				myfeed_dict[h] = training_state
			
			loss,training_state,output_data_2 = sess.run([cost, states,output_data], feed_dict = myfeed_dict)
			val_losses.append(loss)

		valLoss = sum(val_losses)/len(val_losses)

		print("Our File Validation Loss= " + \
				  "{:.6f}".format(valLoss))

		maxIter = int((len(data2)/seq_len)-1.0)
		val_losses = []
		training_state = None

		for i in xrange(1,maxIter):
			#Batch sizes of 30. 
			myTrain_x = data2[seq_len*i:seq_len*(i+1),0:2].reshape((1,seq_len,2))
			myTrain_y = data2[seq_len*i+1:seq_len*(i+1)+1,2:3].reshape((1,seq_len))
			myfeed_dict={X: myTrain_x, Y: myTrain_y}
			if training_state is not None:
				myfeed_dict[h] = training_state
			
			loss,training_state,output_data_2 = sess.run([cost, states,output_data], feed_dict = myfeed_dict)
			val_losses.append(loss)
			
		testLoss = sum(val_losses)/len(val_losses)

		print("Real Validation Loss= " + \
				  "{:.6f}".format(testLoss))
		f2.write("%d\t%f\t%f\n"%(curEpoch, valLoss,testLoss))
		f2.flush()
	'''


	with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:
		print("Session Created")
		sess.run(init)
		
		steps = []
		losses = []
		accs = []
		validation_losses = []
		curEpoch = 0

		
		training_state = None
		i = 0
		#print ("Number train: " , len(data))
		train_file_name = "loss.csv"
		train_loss_file = open(train_file_name,'w')

		for epoch in data:
			i += 1
			print("Epoch: " , i)
			
			for stepa, (X,Y) in enumerate(epoch):
				batch_x = X
				batch_y = Y
				myfeed_dict={X: myTrain_x, Y: myTrain_y}
				empty,loss,training_state,output_data_2 = sess.run([optimizer,  cost, states,output_data], feed_dict = myfeed_dict)
				#print("TrainX: ", myTrain_x)
				
				print("Epoch: " + str(curEpoch) + " Iter " + str(i) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss))

		train_loss_file.close()
			
		print("Optimization Finished!")
		


if __name__=="__main__":
	
	main()
