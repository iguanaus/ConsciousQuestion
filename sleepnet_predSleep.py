'''
    This program trains a feed-forward neural network. It takes in a geometric design (the radi of concentric spheres), and outputs the scattering spectrum. It is meant to be the first program run, to first train the weights. 
'''

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import time
import argparse, os
from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell


RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

num_decay = 43200

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=.1)
    return tf.Variable(weights)

def init_bias(shape):
    """ Weight initialization """
    biases = tf.random_normal([shape], stddev=.1)
    return tf.Variable(biases)

def save_weights(weights,biases,output_folder,weight_name_save,num_layers):
    for i in xrange(0, num_layers+1):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+weight_name_save+"w_"+str(i)+".txt",weight_i,delimiter=',')
        bias_i = biases[i].eval()
        np.savetxt(output_folder+weight_name_save+"b_"+str(i)+".txt",bias_i,delimiter=',')
        print("Bias: " , i, " : ", bias_i)
    return

def load_weights(output_folder,weight_load_name,num_layers):
    weights = []
    biases = []
    for i in xrange(0, num_layers+1):
        weight_i = np.loadtxt(output_folder+weight_load_name+"w_"+str(i)+".txt",delimiter=',')
        w_i = tf.Variable(weight_i,dtype=tf.float32)
        weights.append(w_i)
        bias_i = np.loadtxt(output_folder+weight_load_name+"b_"+str(i)+".txt",delimiter=',')
        b_i = tf.Variable(bias_i,dtype=tf.float32)
        biases.append(b_i)
    return weights , biases

def forwardprop(X, weights, biases, num_layers,dropout=False):
    htemp = None
    for i in xrange(0, num_layers):
        if i ==0:
            htemp = tf.add(tf.nn.relu(tf.matmul(X,weights[i])),biases[i])    
        else:   
            htemp = tf.add(tf.nn.relu(tf.matmul(htemp,weights[i])),biases[i])
        print("Bias: " , i, " : ", biases[i])
    #drop_out = tf.nn.dropout(htemp,0.9)
    yval = tf.add(tf.matmul(htemp,weights[-1]),biases[-1])
    print("Last bias: " , biases[-1])
    return yval

#This method reads from the 'X' and 'Y' file and gives in the input as an array of arrays (aka if the input dim is 5 and there are 10 training sets, the input is a 10X 5 array)
#a a a a a       3 3 3 3 3 
#b b b b b       4 4 4 4 4
#c c c c c       5 5 5 5 5

def get_data(data,percentTest=.2,random_state=42,sampling_rate=100):
    #data = 'data/snip'
    x_file = data+"_data.csv"
    y_file = data+"_lables.csv"
    print("Train X: " , np.genfromtxt(x_file,delimiter='\t'))

    print(np.genfromtxt(x_file,delimiter='\t').shape)
    print(np.genfromtxt(y_file,delimiter='\t').shape)
    train_X = np.genfromtxt(x_file,delimiter='\t')#[0:20000,:]
    #print("TX:",train_X)
    #We don't need this right now. 
    #train_X = train_X[:,[4,5,6,8,10,12]]
    
    print("Train X means")
    print(train_X.mean(axis=0))
    print(train_X.mean(axis=0).shape)
    print("Train X Std")
    print(train_X.std(axis=0))
    train_X = np.subtract(train_X,train_X.mean(axis=0))
    print("Subtracted: " , train_X)
    train_X = np.divide(train_X,train_X.std(axis=0))
    print("Divided: " , train_X)



    #print("TX:",train_X[:,[4,5,6,8,10,12]])
    
    train_Y = np.genfromtxt(y_file,delimiter='\t')#[0:20000,:]
    #B = np.reshape(A, (-1, 2))
    print("Train Y 3: " , train_Y[0::sampling_rate])
    train_Y = train_Y#np.reshape(train_Y,(-1,1))
    print("Train Y 2: " , train_Y)

    lowBar = int(train_X.shape[0]*0.0)
    highBar = int(train_X.shape[0]*1.0)

    print(train_X)
    my_X = train_X[lowBar:highBar,:]
    newX = np.reshape(my_X,(-1,sampling_rate,train_X.shape[1]))
    my_Y = train_Y[lowBar:highBar]
    my_Y = my_Y[0::sampling_rate].astype('int64')
    #Fixes the 0 indexing
    my_Y = np.subtract(my_Y,1)

    #newY = np.reshape(my_Y,(-1,100))

    #newY = np.reshape(my_Y,(-1,,train_Y.shape[1])).astype(np.int64)
    print("My X: " , newX)
    print("My Y: " , my_Y)
    #Now normalize it. #Fine the mean value, subtract it. Find the std of all numbers, and divide by it.
    #Now, the X 
    #Now X should be split into groups of 100. We should ignore the first 25%, the last 25%, then take the middle and split it up.
    #for ele in train_Y:
    #    print len(ele)
    #    print ele
    X_train, X_val, y_train, y_val = train_test_split(newX,my_Y,test_size=percentTest,random_state=random_state)
    return X_train, y_train, X_val, y_val

def main(data,reuse_weights,output_folder,weight_name_save,weight_name_load,n_batch,numEpochs,lr_rate,lr_decay_rate,num_layers,n_hidden,percent_val,n_steps,n_iter):

    f = open("loss.csv",'w')
    n_steps = 100

    train_X, train_Y , val_X, val_Y = get_data(data,percentTest=percent_val,sampling_rate=n_steps)

    x_size = train_X.shape[2]
    print("X Size: " , x_size)
    #n_hidden = 100
    n_classes = 2
    #num_layers = 3
    #y_size = train_Y.shape[2]
    #lr_rate = 0.001
    #lr_rate_decay = .99
    #n_iter = 100000
    #n_batch = 8
    maxVal = train_X.shape[0]/n_batch

    # Symbols
    x = tf.placeholder("float", shape=[None, n_steps,x_size])
    y = tf.placeholder("int64", shape=[None])

    
    # Weight initializations

    cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)

    cells = tf.contrib.rnn.MultiRNNCell([BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias =1) for _ in range(num_layers)],state_is_tuple=True)

    hidden_out, states = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32)


    V_init_val = np.sqrt(6.)/np.sqrt(200)

    V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_classes], \
            dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
    V_bias = tf.get_variable("V_bias", shape=[n_classes], \
            dtype=tf.float32, initializer=tf.constant_initializer(0.01))


    hidden_out_list = tf.unstack(hidden_out, axis=1)[-1]
    final_hidden = tf.matmul(hidden_out_list, V_weights)
    output_data = tf.nn.bias_add(final_hidden, V_bias)


    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_data, labels=y))
    correct_pred = tf.equal(tf.argmax(output_data, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate, decay=lr_decay_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    step4 = time.time()
    steps = []
    losses = []
    accs = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    saver = tf.train.Saver()


    with tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=False)) as sess:
        print("Session Created")
    
        sess.run(init)
        if reuse_weights:
            new_saver = tf.train.import_meta_graph(output_folder+"modelFile"+'.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(output_folder))

        step5 = time.time()
        print("-- Initialization: " + str(step5 - step4))
        step = 0
        epoch_num = 0.0
        #n_batch = 4
        print ("Shape:", train_X.shape)
        cum_loss = 0

        while step < n_iter:
            
            batch_x = train_X[step * n_batch : (step+1) * n_batch]
            batch_y = train_Y[step * n_batch : (step+1) * n_batch]            

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            #epoch_num  = float(n_batch)/55000.0*step
            step += 1
            if step == maxVal:
                step = 0
                epoch_num += 1.0
                acc, val_loss = sess.run([accuracy,cost],feed_dict={x:val_X,y:val_Y})


                print("Epoch: " , epoch_num, " loss=", cum_loss, " val loss: " , val_loss, "accuracy", acc)
                f.write(str(cum_loss)+"," + str(val_loss)+"," + str(acc))
                f.write("\n")
                f.flush()
                cum_loss = 0
                file_name_file = saver.save(sess,os.path.join(output_folder+"modelFile"))
                print("Model saved in: " , file_name_file)
                
            if step % 5 == 0:
                loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
                acc = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
                cum_loss += loss
                print("Epoch: " + str(epoch_num) + " Iter: " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
                
                


    print "========Iterations completed in : " + str(time.time()-start_time) + " ========"
    sess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--data",type=str,default='data/snip')
    parser.add_argument("--reuse_weights",type=str,default='True')
    parser.add_argument("--output_folder",type=str,default='results/Project_1/')
        #Generate the loss file/val file name by looking to see if there is a previous one, then creating/running it.
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights
    parser.add_argument("--weight_name_save",type=str,default="")
    parser.add_argument("--n_batch",type=int,default=20)
    parser.add_argument("--numEpochs",type=int,default=2000)
    parser.add_argument("--lr_rate",default=.001)
    parser.add_argument("--lr_decay_rate",default=.99)
    parser.add_argument("--num_layers",default=2)
    parser.add_argument("--n_hidden",default=100)
    parser.add_argument("--percent_val",default=.2)
    parser.add_argument("--n_steps",default=100)
    parser.add_argument("--n_iter",default=100000)

    args = parser.parse_args()
    dict = vars(args)

    for i in dict:
        if (dict[i]=="False"):
            dict[i] = False
        elif dict[i]=="True":
            dict[i] = True
        
    kwargs = {  
            'data':dict['data'],
            'reuse_weights':dict['reuse_weights'],
            'output_folder':dict['output_folder'],
            'weight_name_save':dict['weight_name_save'],
            'weight_name_load':dict['weight_name_load'],
            'n_batch':dict['n_batch'],
            'numEpochs':dict['numEpochs'],
            'lr_rate':dict['lr_rate'],
            'lr_decay_rate':dict['lr_decay_rate'],
            'num_layers':dict['num_layers'],
            'n_hidden':dict['n_hidden'],
            'percent_val':dict['percent_val'],
            'n_steps':dict['n_steps'],
            'n_iter':dict['n_iter']
            }

    main(**kwargs)




