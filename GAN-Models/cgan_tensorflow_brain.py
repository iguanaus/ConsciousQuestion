import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import os
import pickle
import pandas as pd

#mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 8

n_sample = mb_size # This should be adjusted. keep this the same if you want them to look comparable. 

Z_dim = 100
#X_dim = mnist.train.images.shape[1]
#y_dim = mnist.train.labels.shape[1]
h_dim = 128
abs_total_iter = 100000

#Load special data

x_dat = np.load('data/norm_gan_data_30sec.npy')
y_dat = np.load('data/norm_gan-data_30sec_labels.npy')
p_dat = np.load('data/norm_gan-data_30sec_pat.npy')

print("X data: " , x_dat)
print("Y data: " , y_dat)
print("P data: " , p_dat)

#print("X dat: ", x_dat)
#print("Y dat: " , y_dat)
#print(y_dat.shape)
new_len_x = x_dat.shape[1]*x_dat.shape[2]
#new_len_y = y_dat.shape[1]*y_dat.shape[2]
#print("new length: " ,new_len)
#print("X data: " , x_dat)
train_X = x_dat.reshape([-1,new_len_x])
#Note that we need to one-hot this. 
#print("y dat last: " , y_dat)
#y_dat[-1][0]=2
#print("y dat last: " , y_dat)
train_Y = y_dat-1#.reshape([-1,new_len_y])
n_classes = 2
train_y_new = np.zeros((y_dat.shape[0],n_classes))
#b = np.zeros((3, 4))
print("Trainy : " , train_Y)
print("Train y new: " , train_y_new)
#print("y data shape: " , y_dat.shape[0])
train_y_one_d = train_Y.flatten()
#print("Train y one d: " , train_y_one_d)
train_y_new[np.arange(y_dat.shape[0]), train_y_one_d] = 1
#print("new train:", train_y_new)
#print("Train y: " , train_Y)
#os.exit()
train_Y = train_y_new

#print("X shape: " , train_X.shape)

X_dim = train_X.shape[1]
#print("X dim: " , X_dim)
y_dim = train_Y.shape[1]
#print("y diM: " , y_dim)
print("Train X Shape: " , train_X.shape)
print("Train x is: " , train_X)
print("Final trainy : ", train_Y)
#os.exit()

print("Train Y Shape: " , train_Y.shape)

total_iterations = int(train_X.shape[0]/mb_size)
print("Total iterations: " , total_iterations)

# This is first pass for data. 
if False:
    #We should populate train x/y
    train_X = []#np.array([total_iterations,mb_size,X_dim])
    train_Y = []#np.array([total_iterations,mb_size,y_dim])
    i = -1
    for it in range(total_iterations*mb_size):
        i = i + 1
        X_mb, y_mb = mnist.train.next_batch(1)
        #print("X: " , X_mb)
        train_X.append(X_mb[0])
        train_Y.append(y_mb[0])
        #train_X = train_X.vstack([train_X,X_mb])
        #train_Y = train_Y.vstack([train_Y,y_mb])
    print(train_X)
    train_X = np.vstack( train_X)
    train_Y = np.vstack(train_Y)
    print("X complete:")
    print(train_X)
    print(train_X.shape)
    print("Y complete:")
    print(train_Y)
    print(train_Y.shape)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z, y):
    print("Z value: " , z)
    print("Y value: " , y)
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_prob = tf.matmul(G_h1, G_W2) + G_b2
    #G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = -1
j = 0

for it in range(abs_total_iter):
    i += 1
    if (i >= total_iterations):
        #print("At total...")
        #break
        i = 0

    if it % 1000 == 0:

        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        print("Y sample: " , y_sample)
        #y_sample[7,:] = 1
        y_sample[:,0] = 1
        print("new y sample: " , y_sample)

        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})

        pickle.dump(samples,open('out/{}.dat'.format(str(j).zfill(3)),'wb'))
        

        #fig = plot(samples)
        #plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        #i = 0 
        #plt.close(fig)
    #print("I value" , i)

    X_mb = train_X[i * mb_size : (i+1) * mb_size]
    y_mb = train_Y[i * mb_size : (i+1) * mb_size]
    #print("Xmb: " , X_mb)
    #print("ymb: "  , y_mb)
    if it % 1000 == 0:
        print("J is: " , j)
        if j > 5:
            pass
        else:
            samples = X_mb
            pickle.dump(samples,open('out/ex{}.dat'.format(str(j).zfill(3)),'wb'))
            #fig = plot(samples)
            #plt.savefig('out/ex{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        j += 1

    #X_mb, y_mb = mnist.train.next_batch(mb_size)

    Z_sample = sample_Z(mb_size, Z_dim)
    #print("Z dim: " , Z_dim)
    #print("mb size: " , mb_size)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
