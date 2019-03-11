# -*- coding: utf-8 -*-
"""DNN_DMG_1000_gage1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bzuGmQfrgBcMYlO3bhAuXA-N_1DvWm5e
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# import seaborn as sns
from sklearn.preprocessing import StandardScaler


# HB : test modification

# Data loading: Damage for 100 simulation
Data = pd.read_csv('./DataNS1000_gage1.csv')
Data

# Data selection:
wind = Data.values[:,1:65]
dmg = Data.values[:,65]
XX = wind
m, n = XX.shape
yy = dmg

# Scaling Data
scaler = StandardScaler()
XX = scaler.fit_transform(XX)

# Train and Test set
X_train = XX[:10000].reshape(-1,n)
y_train = yy[:10000].reshape(-1,1)
X_test = XX[10000:].reshape(-1,n)
y_test = yy[10000:].reshape(-1,1)

m,n = X_train.shape
p = y_train.shape

Labels = ['W-14','W-13','Sgm-u-13','Sgm-v-13','Sgm-w-13','Sgm-u-14','Sgm-v-14','Sgm-w-14','Damage']
Dataf = pd.DataFrame({Labels[0]:XX[:,0],Labels[1]:XX[:,1],Labels[2]:XX[:,2],Labels[3]:XX[:,3],Labels[4]:XX[:,4],Labels[5]:XX[:,5],Labels[6]:XX[:,6],Labels[7]:XX[:,7],Labels[8]:yy})
# sns.pairplot(Dataf, diag_kind="kde")

# DNN Creation

def neuron_layers(X, n_neurones, name='layer', activation=None):
  with tf.name_scope(name):
    n_inputs = int(X.get_shape()[1])
    stddev = np.sqrt(2) * np.sqrt(2/(n_inputs + n_neurones))
    init = tf.truncated_normal((n_inputs, n_neurones), stddev=stddev)
    W = tf.Variable(init, name='kernel')
    b = tf.Variable(tf.zeros([n_neurones]), name='bias')
    z = tf.matmul(X,W) + b
    if activation:
      return activation(z)
    else:
      return z
    
def selu (z, scale=1.05070098735548, alpha=1.6732632423543772848):
  return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = X_train.reshape(-1,n_input)[indices] # not shown
    y_batch = y_train.reshape(-1,n_output)[indices] # not shown
    return X_batch, y_batch
  
n_epoch = 100
learning_rate = 0.001

n_input = n
n_hidden1 = 300
n_hidden2 = 300
n_hidden3 = 300
n_hidden4 = 200
n_hidden5 = 200
n_hidden6 = 100
n_hidden7 = 100
n_output = 1

X = tf.placeholder(tf.float32, shape=[None, n_input], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_output], name='y')


initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG')

# Dropout
training = tf.placeholder_with_default(False,shape=(),name='training')
dropout_rate=0.5
X_drop = tf.layers.dropout(X,dropout_rate, training=training)

with tf.name_scope('DNN'):
  #hidden1 = neuron_layers(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
  #hidden2 = neuron_layers(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
  #hidden3 = neuron_layers(hidden2, n_hidden3, name='hidden3', activation=tf.nn.relu)
  #hidden4 = neuron_layers(hidden3, n_hidden4, name='hidden4', activation=tf.nn.relu)
  #hidden5 = neuron_layers(hidden4, n_hidden5, name='hidden5', activation=tf.nn.relu)
  hidden1 = tf.layers.dense(X_drop, n_hidden1, name='hidden1', activation=tf.nn.relu, kernel_initializer=initializer)
  hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
  hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name='hidden2', activation=tf.nn.relu, kernel_initializer=initializer)
  hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
  hidden3 = tf.layers.dense(hidden2_drop, n_hidden3, name='hidden3', activation=tf.nn.relu, kernel_initializer=initializer)
  hidden3_drop = tf.layers.dropout(hidden3, dropout_rate, training=training)
  hidden4 = tf.layers.dense(hidden3_drop, n_hidden4, name='hidden4', activation=tf.nn.relu, kernel_initializer=initializer)
  hidden4_drop = tf.layers.dropout(hidden4, dropout_rate, training=training)
  #hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu, kernel_initializer=initializer)
  #hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu, kernel_initializer=initializer)
  #hidden3 = tf.layers.dense(hidden2, n_hidden3, name='hidden3', activation=tf.nn.relu, kernel_initializer=initializer)
  #hidden4 = tf.layers.dense(hidden3, n_hidden4, name='hidden4', activation=tf.nn.relu, kernel_initializer=initializer)
  #hidden5 = tf.layers.dense(hidden4, n_hidden5, name='hidden5', activation=tf.nn.relu, kernel_initializer=initializer)
  #hidden6 = tf.layers.dense(hidden5, n_hidden6, name='hidden6', activation=tf.nn.relu, kernel_initializer=initializer)
  #hidden7 = tf.layers.dense(hidden6, n_hidden7, name='hidden7', activation=tf.nn.relu, kernel_initializer=initializer)
  output = neuron_layers(hidden4_drop, n_output, name='output')
  
with tf.name_scope('loss'):
  error = output - y
  #mse = tf.metrics.mean_absolute_error(y,output,weights=None,metrics_collections=None,updates_collections=None,name=None)
  mse = tf.reduce_mean(tf.square(error))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
  #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9, decay=0.9, epsilon=1e-10)
  training_op = optimizer.minimize(mse)

# Accuracy
accuracy = np.abs(output - y)

with tf.name_scope('eval'):
  pass
  
init = tf.global_variables_initializer()
saver = tf.train.Saver()

batch_size = 500
n_batches = int((m/batch_size))

# DNN Training
mse_train = []
mse_test = []

with tf.Session() as sess:
#   sess.run(init)
  saver.restore(sess, "Damage100.ckpt")  # load best model
  
  for epoch in range(n_epoch):
    for batch_index in range(n_batches):
      X_b, y_b = fetch_batch(epoch, batch_index, batch_size)
      sess.run(training_op, feed_dict={X:X_b, y:y_b})
      mse_train.append(mse.eval(feed_dict={X:X_b, y:y_b}))
      mse_test.append(mse.eval(feed_dict={X:X_test, y:y_test}))
    if epoch % 10 == 0:
      print('Epoch:', epoch, 'MSE Training:',mse.eval(feed_dict={X:X_b, y:y_b}), 'MSE Validation:', mse.eval(feed_dict={X:X_test, y:y_test}))
      #print('Acc Train:', accuracy.eval(feed_dict={X:X_b, y:y_b}), ', Acc Test:', accuracy.eval(feed_dict={X:X_test, y:y_test}))
  y_pred = sess.run(output, feed_dict={X:X_test})
  acc_train = accuracy.eval(feed_dict={X:X_train, y:y_train})
  acc_test = accuracy.eval(feed_dict={X:X_test, y:y_test})
  save_path = saver.save(sess, "./Damage100_local.ckpt")

plt.plot(mse_train, label='MSE train')
plt.plot(mse_test, label='MSE test')
plt.legend()
plt.show()

plt.scatter(range(len(acc_test)),acc_test)

tol = 0.002
correct_test = [1 for i in acc_test if i < tol]
total_correct = len(correct_test)/len(acc_test)
total_correct

plt.hist(y_test,bins=50)
plt.show()

plt.hist(y_pred,bins=50)
plt.show()
