'''
Created on 17 Oct 2017

@author: Hannah_Pinson
'''

#from FFT_RNN import FTRNN
#import FTRNN
import keras
import tensorflow as tf
import random
import numpy as np
import math
from sklearn.utils import shuffle
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.merge import Dot
from keras.layers import Reshape
from keras.layers.recurrent import SimpleRNN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.layers import recurrent
from keras.engine import InputSpec
from keras.legacy import interfaces
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.callbacks import EarlyStopping

from keras.constraints import unitnorm

from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops






################################################################################
################################################################################
########################### FTRNN Class ########################################
################################################################################
################################################################################



class FTRNN(recurrent.Recurrent):
    
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(FTRNN, self).__init__(**kwargs)
        self.units = units*2  # both real and imaginary operations
        self.complex_cells = units
        self.state_spec = InputSpec(shape=(None, self.units))
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        
        ##cell = FFT_Cell.FTRNNCell(units)
        ##super(FTRNN, self).__init__(cell, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(FTRNN, self).call(inputs,
                                        mask=mask,
                                        training=training,
                                        initial_state=initial_state)

    #@property
    #def units(self):
    #    return self.units
    
    def get_config(self):
        config = {'units': self.units}
        base_config = super(FTRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))
    
    
    ##################
    
    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        #input weights
        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        
        
        #recurrent weights 
        
        #build a tensorarray of recurrent kernels: one kernel per complex cell consisting of a real and an imaginary unit (thus 2x2 matrix)
#         number_of_trainable_submatrices = self.complex_cells
#         self.recurrent_kernel =  tensor_array_ops.TensorArray(size=number_of_trainable_submatrices, dtype='float32', clear_after_read = False, name="recurrent_kernel")
#         
#         #fill the tensorarray with trainable complex cell tensors (2x2)
#         init_state = (0, self.recurrent_kernel)
#         condition = lambda i, _: i < number_of_trainable_submatrices
#         body = lambda i, ta: (i + 1, 
#                               ta.write(i, self.add_weight( #this makes the complex cell matrix trainable
#                                   shape=(2, 2), #complex cell matrix
#                                   name = "recurrent_kernel",
#                                   initializer=self.recurrent_initializer,
#                                   regularizer=self.recurrent_regularizer,
#                                   constraint=self.recurrent_constraint))) #constraint should be norm one 
#         _, self.recurrent_kernel = tf.while_loop(condition, body, init_state)
        
        self.frequencies = self.add_weight(shape=(1,self.complex_cells), #contains the frequencies (1 frequency per complex cell)
                                      name='frequencies',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        
        
        self.bias = None
        self.built = True
        
#     def update_cell(self, tf_states_tensor, python_states_list, tensorarray_kernels, i):
#         #update the cell (real and imaginary unit) by multiplying it with the corresponding recurrent kernel
#         #index corresponds to the index in the tensorarray of kernels (or, if considering cells, the index of the cell)
#         # the real state is at index i*2 in the tensor of states and imaginary state at index i*2+1 in the tensor of states
#     
#         states = tf_states_tensor[:, i*2:(i*2)+1]
# #         states = tf.gather(tf_states_tensor,
# #                            tf.concat( tf.Variable(2*i),tf.Variable(2*i + 1)),
# #                            axis=1)
#         python_states_list.append(K.dot( states, tensorarray_kernels.read(i))) 
#         #tf_states_tensor[:, i*2:(i*2)+1].assign(K.dot( states, tensorarray_kernels.read(i)))
#         return python_states_list

    def permuteStates(self, states):
        reshaped = tf.reshape(states, [self.complex_cells,2], name="firstReshape")
        transposed = tf.transpose(reshaped)
        indices = tf.constant([[1],[0]])
        shape = tf.constant([2, self.complex_cells])
        scatter1 = tf.scatter_nd(indices, transposed, shape)
        transposed2 = tf.transpose(scatter1)
        reshaped2 = tf.reshape(transposed2, [self.units])
        return reshaped2
        


    def step(self, inputs, states):
        
        #retrieve the previous states 
        prev_states = states[0]
        freq = tf.reshape(self.frequencies, [self.complex_cells])
        
        #states_list = tf.unstack(tf_state_tensor, axis=1)#tensor_array_ops.TensorArray(size=self.complex_cells, dtype='float32')
        #new_states_list = []
        
#         states_transposed =  tf.transpose(tf_state_tensor)
#         
#         states_ta = tf.TensorArray(size=self.units, dtype=tf.float32, name = "states_ta")
#         states_ta.unstack(states_transposed)
    
        
        #while loop updating the cells with the recurrent kernels 
        #the loop appends the updated states to the tensorarray, after which they can be stacked in a tensor with the same shape as the original one
        #(this is needed because a tensor slice is non-assignable)
#         while_body = lambda i, l : (i + 1, 
#                                     #new_states_list.append( K.dot( states_list[:, i*2:(i*2)+2], self.recurrent_kernel.read(i))) )
#                                     #states_ta.write(i, K.dot( tf_state_tensor[:, i*2:(i*2)+2], self.recurrent_kernel.read(i))))
#                                     l.write(i, tf.multiply( tf_state_tensor[:, i], (self.recurrent_kernel.read(i//2))[i%2, 0]) + tf.multiply( tf_state_tensor[:, i], (self.recurrent_kernel.read(i//2))[i%2, 1])))
#                                     #tf.concat(0, [l, K.dot( tf_state_tensor[:, i*2:(i*2)+1], self.recurrent_kernel.read(i)) ]))
#                                     # python_states_list.append())
#                                     #self.update_cell(tf_state_tensor, python_state_list, self.recurrent_kernel, i)) 
#         init_state = [0, states_ta]
#         condition = lambda i, _: i < self.units
#         #_, _ = tf.while_loop(condition, while_body, init_state) 
#         
#         _, states_ta  = tf.while_loop(condition, while_body, init_state) 
#         result = states_ta.stack()
#         
#         result = tf.transpose(result)
#         
#         print("result:")
#         print(result)
        
        #result = tf.concat(result, 0)
        
        # input kernel
        h = K.dot(inputs, self.kernel)
        
        #multiplication with block-diagonal orthogonal matrix
        
        omega = tf.scalar_mul(np.pi*2, freq)
        v1 = tf.cos((tf.reshape(tf.tile(tf.expand_dims(omega, -1),  [1, 2]), [-1])))
        v2 = tf.sin((tf.reshape(tf.tile(tf.expand_dims(omega, -1),  [1, 2]), [-1])))
        sign_mask = tf.reshape(tf.tile(tf.expand_dims([1.0, -1.0], 0),  [1, self.complex_cells]), [-1])
        v2 = tf.multiply(v2, sign_mask)
        
        prev_states_permuted = self.permuteStates(prev_states)
        
        result = tf.multiply(v1, prev_states) + tf.multiply(v2, prev_states_permuted)    
        output = h + result
       
        return output, [output]



timesteps = 500
scaling_factor =  1/500#1/timesteps #used to scale the input before use in first layer, to mitigate numerical errors


def createWeightsDuplicate(number_of_complex_units):
    weights = np.empty([number_of_complex_units*2, number_of_complex_units*4]) 
    for i in range(number_of_complex_units*2):
        for j in range(number_of_complex_units*4):
            if (j == 2*i or j == 2*i + 1):
                weights[i,j] = 1 
            else:
                weights[i,j] = 0
    return [weights]
    

def createWeightsSquare(number_of_complex_units, l, b):
    weights = np.empty([number_of_complex_units*4, number_of_complex_units*8])
    block_for_even_rows = np.array([l, -l, -l, l])
    block_for_odd_rows = np.array([l, -l, l, -l])
    for i in range(number_of_complex_units*4):
        for j in range(number_of_complex_units*8):
            if (i%2 == 0 and (j ==2*i or j ==2*i+1 or j ==2*i+2 or j ==2*i+ 3) ): #wow, this is ugly : )
                weights[i, j] = block_for_even_rows[j%4]
            elif(i%2 == 1 and (j == 2*i - 2 or j == 2*i - 1 or j == 2*i or j == 2*i +1)): 
                weights[i, j:j+4] = block_for_odd_rows[j%4]
            else: 
                weights[i,j] = 0
                
    bias = np.ones([number_of_complex_units*8]) * b
    return [weights, bias]

def createWeightsSquareResult(number_of_complex_units, m):
    weights = np.empty([number_of_complex_units*8, number_of_complex_units*2])
    for i in range(number_of_complex_units*8):
        for j in range(number_of_complex_units*2):
            if(i in range(j*4, j*4+2)):
                weights[i, j] = m
            elif(i in range(j*4+2, j*4+4)): 
                weights[i, j] = -m
            else: 
                weights[i, j] = 0
    return [weights]

def createWeightAbs(number_of_complex_units):
    weights = np.empty([number_of_complex_units*2, number_of_complex_units])
    for i in range(number_of_complex_units*2):
        for j in range(number_of_complex_units):
            if (i==j*2 or i == j*2+1):
                weights[i,j] = 1/(scaling_factor * scaling_factor)
            else: 
                weights[i,j] = 0
    return [weights]
    
                






################################################################################
################################################################################
########################### Create Toy Dataset #################################
################################################################################
################################################################################


extended_timesteps = 2 * timesteps #to slice a signal with #timesteps from
samples_per_signal = 30
training_samples = 60


 
signal0 = np.zeros(extended_timesteps)  
signal1 = np.zeros(extended_timesteps)
signal2 = np.zeros(extended_timesteps)

 
for i in range(extended_timesteps):
    signal0[i] =  np.sin(1/90 * i * 2 * np.pi) + np.sin(1/50 * i * 2 * np.pi) + np.sin(1/30 * i * 2 * np.pi)  # + np.cos(1/20 * i * 2 * np.pi) + np.cos(1/60 * i * 2 * np.pi)) * 0.5
    signal1[i] =  np.sin(1/90 * i * 2 * np.pi)  + np.sin(1/50 * i * 2 * np.pi) + np.sin(1/72 * i * 2 * np.pi)
    signal2[i] =  np.sin(1/100 * i * 2 * np.pi)  + np.sin(1/40 * i * 2 * np.pi) + np.sin(1/30 * i * 2 * np.pi)  #+ np.cos(1/40 * i * 2 * np.pi) + np.cos(1/60 * i * 2 * np.pi) ) * 0.5
     
total_samples = 3 * samples_per_signal
     
# create training set with (samples_per_signal) examples of (shifted) first signal, (samples_per_signal) examples of second (shifted) signal, s...     
X = np.empty((total_samples,timesteps))
Y = np.empty(total_samples)

gaussian_mask = np.zeros(timesteps)
sigma = timesteps/2
mu = timesteps/2
for t in range(timesteps):
    gaussian_mask[t] = np.exp(-1 * np.power((t-mu)/sigma,2))
    
signal_to_noise = 5


for i in range(samples_per_signal):
    random_begin_index = random.randint(0, extended_timesteps - timesteps) 
    X[i, :] = signal_to_noise * np.multiply(gaussian_mask, signal0[random_begin_index:random_begin_index+timesteps])  + np.random.randn(timesteps)
    Y[i] = 0
    random_begin_index = random.randint(0, extended_timesteps - timesteps)
    X[i+samples_per_signal, :] = signal_to_noise * np.multiply(gaussian_mask, signal1[random_begin_index:random_begin_index+timesteps])  + np.random.randn(timesteps)
    Y[i+samples_per_signal] = 1
    random_begin_index = random.randint(0, extended_timesteps - timesteps)
    X[i+2*samples_per_signal, :] = signal_to_noise * np.multiply(gaussian_mask, signal2[random_begin_index:random_begin_index+timesteps]) + np.random.randn(timesteps)
    Y[i+2*samples_per_signal] = 2
     
X, Y = shuffle(X, Y, random_state=0)


train_X = X[0:training_samples, ]
coloring = Y[0:training_samples, ]

#assign correct class labels
for y_index in  range(Y.shape[0]):
    if (Y[y_index, ]==0):
        Y[y_index, ] = 1
    elif(Y[y_index, ]==1):
        Y[y_index, ] = 0
    elif(Y[y_index, ]==2): 
        Y[y_index, ] = 0
    
train_Y = Y[0:training_samples, ]     
        
         
    
test_X = X[training_samples+1:total_samples, ]
test_Y = Y[training_samples+1:total_samples, ]


plotx= timesteps
plt.figure()
plt.title('Example input time series')
plt.xlabel('Timesteps')
plt.ylabel('Signal')
plt.plot(train_X[4, ][0:plotx], 'r')
plt.plot(train_X[1, ][0:plotx], 'b')
plt.plot(train_X[2, ][0:plotx], 'g')
plt.show()

#(number of samples, sequence size, features)
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1],  1))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))
train_Y = train_Y.reshape((train_X.shape[0], 1))
test_Y = test_Y.reshape((test_X.shape[0], 1))

train_X = train_X.astype("float64")
test_X = test_X.astype("float64")



################################################################################
################################################################################
########################### Build and Train model ##############################
################################################################################
################################################################################


init_period = 50#timesteps
init_freq_1 = 1/init_period
init_period = 30
init_freq_2 = 1/init_period

epochs = 1

print("starting from period: ")
print(init_period)
# cosweight = np.cos(init_freq * 2 * np.pi) 
# sineweight = np.sin( init_freq* 2 * np.pi)
# cosweight = cosweight.astype('float64')
# sineweight = sineweight.astype('float64')
# weights = [np.array([[ 1, 0]]), np.array([[cosweight],[sineweight]])]
# weights_simpleRNN = [np.float64([[ 1/2000, 0]]), np.float64([[cosweight, sineweight], [-sineweight, cosweight]])]
# print(weights_simpleRNN)

number_of_complex_units = 2
total_number_of_units = 2 * number_of_complex_units
weights = [np.array([[scaling_factor, 0, scaling_factor, 0]]), np.array([[init_freq_1, init_freq_2 ]])]
initial_states = np.array([0, 0, 0, 0])


l = 0.1
b = 1
m = 1 / ((4 * (-0.09)) / np.power(l, 2))

weights_duplicate = createWeightsDuplicate(number_of_complex_units)#[np.array([[ 1, 1, 0, 0], [ 0, 0, 1, 1]])]
weights_square = createWeightsSquare(number_of_complex_units, l, b)#[np.array([[l, -l, -l, l, 0, 0, 0, 0], [l, -l, l, -l, 0, 0, 0, 0], [ 0, 0, 0, 0, l, -l, -l, l], [ 0, 0, 0, 0, l, -l, l, -l]]), np.array([b, b, b, b, b, b, b, b])]
weights_square_result = createWeightsSquareResult(number_of_complex_units, m)#[np.array([ [m, 0], [m, 0], [-m, 0], [-m, 0],[0,m], [0,m], [0,-m], [0,-m]]) ]
weights_abs = createWeightAbs(number_of_complex_units)#[np.array([[1], [1]])]
doTrain = False

#initial_weights_last = [np.array([[3.4], [4.2]]), np.array([ -6.5])]

initial_weights_last = [np.array([[0.3], [0.3]]), np.array([ -9])]


model = Sequential()
FTRNNlayer = FTRNN(units=number_of_complex_units, input_shape=(timesteps, 1), return_sequences=False,  trainable=False, recurrent_constraint = unitnorm(), weights=weights)
#FTRNNlayer = SimpleRNN(units=total_number_of_units, input_shape=(timesteps, 1), use_bias=False,  return_sequences=False,  trainable=False,  weights=weights_simpleRNN) #recurrent_constraint = unitnorm(),
FTRNNlayer.states = initial_states
model.add(FTRNNlayer)

#model.add(Dot(axes=1)([FTRNNlayer, FTRNNlayer]))

duplicateLayer = Dense(total_number_of_units*2, activation=None, use_bias=False, trainable=doTrain, weights=weights_duplicate)
squareLayer =  Dense(total_number_of_units*4, activation='sigmoid', use_bias=True, trainable=doTrain, weights=weights_square)
squareResultLayer =  Dense(total_number_of_units, activation=None, use_bias=False,  trainable=doTrain,  weights=weights_square_result)
absLayer = Dense(number_of_complex_units, activation=None, use_bias=False, trainable=doTrain, weights=weights_abs )
lastLayer = Dense(1, activation='sigmoid', use_bias=True, trainable=True, weights = initial_weights_last)
model.add(duplicateLayer)
model.add(squareLayer)
model.add(squareResultLayer)
model.add(absLayer)
model.add(lastLayer)

# model.add(Reshape((number_of_complex_units*2, timesteps)))
# model.add(SimpleRNN(units=2, activation=None, use_bias=False, kernel_initializer='zeros' ))
# model.add(Dense(1, activation='sigmoid'))

#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer='RMSprop',
           loss='binary_crossentropy',
           metrics=['accuracy'])


#before = lastLayer.get_weights()

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
 
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
         
epochcounter = 0
class FreqHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.freqWeights = np.empty([number_of_complex_units, epochs])
 
    def on_epoch_end(self, batch, logs={}):
        print('period of first cell: ')
        print(1/model.layers[0].get_weights()[1][0][0] )
        print('period of second cell: ')
        print(1/model.layers[0].get_weights()[1][0][1] )
        self.freqWeights[0, epochcounter] = model.layers[0].get_weights()[1][0][0] 
        self.freqWeights[1, epochcounter] = model.layers[0].get_weights()[1][0][1] 
        
        
        
earlyStopping = EarlyStopping(monitor='loss', min_delta=1/100000, patience=10, verbose=0)
         
 
 
history = LossHistory()
freqHistory = FreqHistory()
model.fit(train_X, train_Y, epochs=epochs, batch_size=1, shuffle=False, callbacks=[history, freqHistory, earlyStopping], verbose=2) 


#model.fit(train_X, train_Y, epochs=1, batch_size=1, shuffle=True) 

################################################################################
################################################################################
########################### Analyse results ####################################
################################################################################
################################################################################

print("learned periods: ")
print(1/model.layers[0].get_weights()[1][0])  

print("learned sigmoid weights: ")
print( lastLayer.get_weights())  

#  
#  
# learned_freq_cos = np.arccos(model.layers[0].get_weights()[1][0][0]) / (2*np.pi) 
# learned_freq_sin = np.arcsin(model.layers[0].get_weights()[1][1][0]) / (2*np.pi) 
# print("learned frequencies: ")
# print(learned_freq_cos )
# print(learned_freq_sin )
# print("learned period: ")
# print(1/learned_freq_cos )
# print(1/learned_freq_sin )
# 
# 
# 
get_absLayer_output = K.function([FTRNNlayer.input],[absLayer.output])

#get the output of the abs layer (a somewhat awkward construction)
single_input = np.empty((1, train_X.shape[1], train_X.shape[2])) 
all_outputs = np.empty([train_X.shape[0], 2])
for seq_number in range(training_samples):
    FTRNNlayer.states = initial_states
    single_input[0, :, :] = train_X[seq_number, :, :]
    all_outputs[seq_number, 0] = get_absLayer_output([single_input])[0][0, 0]
    all_outputs[seq_number, 1] = get_absLayer_output([single_input])[0][0, 1]
    
    
print(all_outputs)
    

# 
# 
# scores = model.evaluate(test_X, test_Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# # 
# 


x_values = np.arange(0,2,0.05)
y_values = x_values
scaling_x = lastLayer.get_weights()[0][0]
scaling_y = lastLayer.get_weights()[0][1]
bias = lastLayer.get_weights()[1][0]
z_values = np.empty([x_values.shape[0], y_values.shape[0]])
for x_index in range(x_values.shape[0]):
    for y_index in range(y_values.shape[0]): 
        z_values[x_index, y_index] = 1 / (1 + math.exp(-scaling_x*x_values[x_index] -scaling_y*y_values[y_index] - bias ))
    

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(x_values, y_values, z_values)
x_values = np.sqrt(all_outputs[:,0])
y_values = np.sqrt(all_outputs[:,1])

plt.scatter(x_values, y_values, c=coloring)

plt.show()




# plt.figure()
# plt.scatter(np.sqrt(all_outputs), coloring, c=coloring,marker="o", cmap="Set1")
# plt.plot(x_values, y_values)
# plt.title('Discernability of classes')
# plt.xlabel('Absolute value of Fourier Coefficient (times some constant)')
# plt.ylabel('Class (=frequency present or not)')
 
#print(get_absLayer_output([train_X])[0])
 

# 
# 
# 
# get_layer_output = K.function([model.layers[0].input],
#                                   [model.layers[0].output])
# layer_output = get_layer_output([train_X])[0]
# # #print(layer_output)
# #print(lastLayer.get_weights())
# 
# 

figPeriod = plt.subplot(211)
plt.title('Learned period over time')
periods = 1/freqHistory.freqWeights[0,:] 
plt.plot(periods)
periods = 1/freqHistory.freqWeights[1,:] 
plt.plot(periods)
plt.xlabel('Number of epochs')
plt.ylabel('Period')
figPeriod.set_ylim([-100,100])
plt.draw()   


  
figLoss = plt.subplot(212)
plt.title('Losses')
plt.plot(history.losses)
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.draw()

plotx= timesteps
# 
# 
# get_duplicateLayer_output = K.function([model.layers[0].input],
#                                   [duplicateLayer.output])
# print(get_duplicateLayer_output([train_X])[0])
# 
# 
# get_squareLayer_output = K.function([model.layers[0].input],
#                                   [squareLayer.output])
# print(get_squareLayer_output([train_X])[0])
# 
# 
# get_resultLayer_output = K.function([model.layers[0].input],
#                                   [resultLayer.output])
# print(get_resultLayer_output([train_X])[0])
# 
# get_absLayer_output = K.function([model.layers[0].input],
#                                   [absLayer.output])
# print(get_absLayer_output([train_X])[0])
# 
# get_lastLayer_output = K.function([model.layers[0].input],
#                                   [lastLayer.output])
# print(get_lastLayer_output([train_X])[0])
# 
# print(lastLayer.get_weights())
#  
# print("before:")
# print(before)
# 
# print(train_Y)





#  
# print(layer_output.shape)
# plt.figure()
# plt.title('Output time series')
# plt.plot(layer_output[0,:, 0][0:plotx], 'b')
# plt.plot(layer_output[1,:, 0][0:plotx], 'r')
# plt.plot(layer_output[0,:, 1][0:plotx], 'b--')
# plt.plot(layer_output[1,:, 1][0:plotx], 'r--')
# plt.draw()
#     
#     
# print(layer_output.shape)
# plt.figure()
# plt.title('Absolute value of Output time series')
# plt.plot( np.power((layer_output[0,:, 0][0:plotx]),2)+ np.power((layer_output[0,:, 1][0:plotx]),2) , 'b')
# plt.plot( np.power((layer_output[1,:, 0][0:plotx]),2)+ np.power((layer_output[1,:, 1][0:plotx]),2) , 'r')
# #plt.plot( np.power((layer_output[2,:, 0][0:plotx]),2)+ np.power((layer_output[2,:, 1][0:plotx]),2) , 'r')
# #plt.plot( np.power((layer_output[3,:, 0][0:plotx]),2)+ np.power((layer_output[3,:, 1][0:plotx]),2) , 'b')
# plt.draw()





# 
# plt.show()
# 
# 
# 
# model_test = Sequential()
# model_test.add(Dense(1, activation='sigmoid', use_bias=True, trainable=True, input_dim=1))
# model_test.compile(optimizer='adam',
#           loss='binary_crossentropy',
#           metrics=['accuracy'])
# 
# 
# output_X = output_X.reshape((output_X.shape[0], output_X.shape[1],  1))
# 
# 
# model_test.fit(output_X, train_Y, epochs=5, batch_size=1, shuffle=True, verbose=2) 
# 
# print(model_test.layers[0].get_weights())


plt.show()


