# -*- coding: utf-8 -*-
# DNN_DMG_1000_gage1_NoHighVal.ipynb
# Wilson VELOZ, Hao BAI
#
# Original file is located at
#    https://colab.research.google.com/drive/1ym7L1mEyG46ovT0VfNNgHjWgMXNQOKxH
#
# Deep Neural Network for prediting fatigue damage from wind conditions
#   - INPUT: standard deviation on X-direction over all height under 1 wind 
#            speed [dim=16]
#   - OUTPUT: damage on theta=0° on tower gage 1 [dim=1]
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
try:
    import seaborn as sns
except:
    pass
import sklearn.preprocessing
import sklearn.utils
import json
import random


#! General setting =============================================================
export_result = False # set True for save prediction values in .json
reuse_network = False # set True for loading pre-trained network
save_network = False # set True for saving current network after training
by_batch = False # set True for training network through mini-batches

speed = 3.0 # wind speed used in this study


#! Preprocessing Data ==========================================================
""" - Inputs are stored between columns 1 and 64. The first 16 columns are wind
    speed at different heigth of the wind turbine tower, from Hub height to 
    bottom, the following columns contain standard deviation u, v and w 
    components for each wind speed.
    
    - Outputs are from column 65 to 101, these are the damage computed at the 
    bottom of the wind turbine tower for differents angles theta = [0:10:360].
"""
# Data loading: Damage for 1000 simulation
Data = pd.read_csv('./DataNS1000_gage1.csv',)

#* data selection
# damage
threshold = 0.0000002 # Any damage greater than this value will be removed from 
                      # database
for col in Data.columns[65:]:
    Data = Data[Data[col] <= threshold]
# wind speed
for col in Data.columns[1:2]:
    Data = Data[Data[col] == speed]

Data = sklearn.utils.shuffle(Data, random_state=0)
        # permute data in random order
        # random_state: seed for generating pseudo random numbers

#* Train set
# Input data: standard deviation in X direction over all heights
indice = [17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62]
wind = Data.values[:, indice]
# Output data: 
#dmg = Data.values[:,65:] # All outputs for theta from 0 to 350
dmg = Data.values[:, 65] # output on theta = 0
XX = wind
m, n = XX.shape
if len(dmg.shape) == 1:
    yy = dmg.reshape(-1,1)
else:
    yy = dmg
p, q = yy.shape
# Scaling Data: normalize values to the range (0,1)
scaler = sklearn.preprocessing.MinMaxScaler()
scalery = sklearn.preprocessing.MinMaxScaler()
XX = scaler.fit_transform(XX)
yy = scalery.fit_transform(yy) 

# Split Train and Validation sets
N_train = 800 # length of train set
seed_train = Data.values[:N_train,0] # seed column
X_train = XX[:N_train].reshape(-1,n) # input
y_train = yy[:N_train].reshape(-1,q) # output
m, n = X_train.shape # update dimensions
p, q = y_train.shape

#* Valid set
seed_valid = Data.values[N_train:,0]
X_valid, y_valid = XX[N_train:].reshape(-1, n), yy[N_train:].reshape(-1, q)


#! Parametrage =================================================================
# Iteration Hyperparameters
n_epoch = 2000 # maximum allowable epoch
learning_rate = 6.480902580168296e-05 # optimized value from skopt
# Layers Hyperparameters
n_input = n
n_hidden1 = 251
n_hidden2 = 251
n_hidden3 = 200
n_hidden4 = 200
n_hidden5 = 150
n_hidden6 = 150
n_hidden7 = 150
n_output = q
# Batch Hyperpameter
batch_size = 100
n_batches = int((m/batch_size))
X = tf.placeholder(tf.float32, shape=[None, n_input], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_output], name='y')
# Dropout Hyperparameters
training = tf.placeholder_with_default(False, shape=(), name='training')
dropout_rate = 0.5
X_drop = tf.layers.dropout(X, dropout_rate, training=training)
# He initialization: used if layers are created by tensorflow tf.layers.dense
initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                mode='FAN_IN')


#! Deep Neural Network (DNN)====================================================
def fetch_batch(epoch, batch_index, batch_size):
    """ Transform train set into random mini-batches
        -epoch, batch_index, batch_size: integrer
    """
    np.random.seed(epoch * n_batches + batch_index)  # Different random seed
    indices = np.random.randint(m, size=batch_size)  
    indices_y = np.random.randint(p, size=batch_size) # Random indexes
    X_batch = X_train.reshape(-1,n_input)[indices] 
    y_batch = y_train.reshape(-1,n_output)[indices_y]
    return X_batch, y_batch

with tf.name_scope('DNN'):
    """ Creation of Deep Neural Network
    """
    #* Study case 1: with dropout layers
    #hidden1 = tf.layers.dense(X_drop, n_hidden1, name='hidden1', activation=leaky_relu, kernel_initializer=initializer)
    #hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    #hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name='hidden2', activation=leaky_relu, kernel_initializer=initializer)
    #hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    #hidden3 = tf.layers.dense(hidden2_drop, n_hidden3, name='hidden3', activation=leaky_relu, kernel_initializer=initializer)
    #hidden3_drop = tf.layers.dropout(hidden3, dropout_rate, training=training)
    #hidden4 = tf.layers.dense(hidden3_drop, n_hidden4, name='hidden4', activation=leaky_relu, kernel_initializer=initializer)
    #hidden4_drop = tf.layers.dropout(hidden4, dropout_rate, training=training)

    #* Study case 2: normal
    hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu, kernel_initializer=initializer)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu, kernel_initializer=initializer)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name='hidden3', activation=tf.nn.relu, kernel_initializer=initializer)
    hidden4 = tf.layers.dense(hidden3, n_hidden4, name='hidden4', activation=tf.nn.relu, kernel_initializer=initializer)
    hidden5 = tf.layers.dense(hidden4, n_hidden5, name='hidden5', activation=tf.nn.relu, kernel_initializer=initializer)
    hidden6 = tf.layers.dense(hidden5, n_hidden6, name='hidden6', activation=tf.nn.relu, kernel_initializer=initializer)
    hidden7 = tf.layers.dense(hidden6, n_hidden7, name='hidden7', activation=None, kernel_initializer=initializer) #! activation=None means
    #! using a linear activation function a(x)=x
    output = tf.layers.dense(hidden7, n_output, name='output')

with tf.name_scope('loss'):
    """ Loss function for Regression problem: MSE (Mean Square Error)
        -Optimizer: AdamOptimizer (can update the learning rate)
    """
    error = output - y
    mse = tf.reduce_mean(tf.square(error))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

#! Train model =================================================================
accuracy = np.abs(output - y) # Absolute Accuracy of the model
init = tf.global_variables_initializer()
saver = tf.train.Saver()
mse_train = [] # History of MSE for each training batch
mse_valid = [] # History of MSE for the validation set
i = 0
tol = 1e-10
n_iter = 2000 # set a break point before maximum epoch
mse_tol = -99999999.0

with tf.Session() as sess:
    sess.run(init)
    if reuse_network:
        saver.restore(sess, "./Damage1000.ckpt")
        print("[OK] Checkpoint loaded !")
    # training model
    for epoch in range(n_epoch):
        if by_batch:
            for batch_index in range(n_batches):
                X_b, y_b = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={X:X_b, y:y_b})
                mse_train.append(mse.eval(feed_dict={X:X_b, y:y_b}))
                mse_valid.append(mse.eval(feed_dict={X:X_valid, y:y_valid}))
        else:
            X_b, y_b = X_train, y_train
            sess.run(training_op, feed_dict={X:X_b, y:y_b})
            mse_train.append(mse.eval(feed_dict={X:X_b, y:y_b}))
            mse_valid.append(mse.eval(feed_dict={X:X_valid, y:y_valid}))
        # print information
        if epoch % 100 == 0:
            print('Epoch:', epoch,
                  'MSE Training:', mse.eval(feed_dict={X:X_b, y:y_b}),
                  'MSE Validation:', mse.eval(feed_dict={X:X_valid, y:y_valid}))
        # exit criteria
        if abs(mse_valid[i]-mse_tol) <= tol or epoch >= n_iter:
            save_path = saver.save(sess, "./Damage1000.ckpt")
            epoch_saved = epoch
            y_pred = sess.run(output, feed_dict={X:X_valid})
            acc_train = accuracy.eval(feed_dict={X:X_train, y:y_train})
            acc_valid = accuracy.eval(feed_dict={X:X_valid, y:y_valid})
            print("[OK] Forced Exit !")
            break
        else:
            mse_tol = mse_valid[i]
            i = i + 1
    if save_network:
        save_path = saver.save(sess, "./Damage1000.ckpt")
        print("[OK] Checkpoint saved !")
    epoch_saved = epoch
    y_pred = sess.run(output, feed_dict={X:X_valid})
    acc_train = accuracy.eval(feed_dict={X:X_train, y:y_train})
    acc_valid = accuracy.eval(feed_dict={X:X_valid, y:y_valid})


#! Postprocessing ==============================================================
# Plot of MSE train and validation vs Nb. of iterations
plt.plot(mse_train, label='MSE train')
plt.plot(mse_valid, label='MSE valid')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Plot of absolute error in the validation set
plt.scatter(range(len(acc_valid[:,0])),acc_valid[:,0])
plt.ylim([0,max(acc_valid[:,0])])
plt.xlabel("No. Sample - Validation Set")
plt.ylabel("Absolute Error: Output Prediction - Output")
plt.show()

tol = 1e-2
correct_valid = [1 for i in acc_valid[:,0] if i < tol]
total_correct = len(correct_valid)/len(acc_valid[:,0])
print("total_correct is {}".format(total_correct))

# Denormalization: keep prediction between 0 and 1
X_valid_dt = scaler.inverse_transform(X_valid)
#y_valid_dt = scalery.inverse_transform(y_valid.reshape(1, -1))
#y_pred_dt = scalery.inverse_transform(y_pred.reshape(1, -1))
#y_valid_dt = y_valid_dt.reshape(-1,1)
#y_pred_dt = y_pred_dt.reshape(-1,1)
y_valid_dt = y_valid
y_pred_dt = y_pred

# Histogram of Damage simulated in the validation set
plt.subplot(1,2,1)
plt.hist(y_valid_dt[:,0],bins=100)
plt.ylabel('Count')
plt.xlabel('Damage Simulated')
# Histogram of Damage predicted by DNN
plt.subplot(1,2,2)
plt.hist(y_pred_dt[:,0],bins=50)
plt.ylabel('Count')
plt.xlabel('Damage Predicted')
plt.show()

# Plot correlation between Damage predicted and simulated
theta = 0 # index of list of angle
plt.scatter(y_valid_dt[:,theta],y_pred_dt[:,theta],)
d = [0,1]
plt.plot(d,d,'r',label = 'Correlation')
plt.xlim(0,max(y_pred_dt[:,theta]))
plt.ylim(0,max(y_pred_dt[:,theta]))
plt.xlabel('Damage Simulated')
plt.ylabel('Damage Predicted')
plt.show()
print("y_pred_dt is {}".format(y_pred_dt[:,0]))
print("y_valid_dt is {}".format(y_valid_dt[:, 0]))

# Plot of Damage predicted and simulated vs Wind Speed for the validation set
input_var = 0
theta = 0
plt.scatter(X_valid_dt[:,input_var],y_valid_dt[:,theta], label='Simulation Output')
plt.scatter(X_valid_dt[:,input_var],y_pred_dt[:,theta],label='Predicted Output')
#plt.ylim([-1e-7,8e-7])
plt.ylabel('Damage')
#plt.xlabel('Wind Speed')
plt.legend()
plt.show()

#* Export damages to JSON File
if export_result:
    result = []
    for i in range(len(X_valid_dt)):
        for j in range(len(y_pred_dt[0,:])):
            result.append((seed_valid[i], X_valid_dt[i,0], j*10,
                        float(y_pred_dt[i,j]), float(y_valid_dt[i,j])))
    with open('Damage.json', 'w') as f:
        json.dump(result,f,indent=4)

