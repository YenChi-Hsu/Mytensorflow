import time
import tensorflow as tf
import load_data

sess = tf.InteractiveSession()

# Input images.
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
# Target output.
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# Build a Multilayer Convolutional Network
##################################
##################################
## Convolutional Neural Network ##
##################################
##################################

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#############################
# First Convolutional layer #
#############################

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

##############################
# Second Convolutional Layer #
##############################

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

###########################
# Densely Connected Layer #
###########################

W_fc1 = weight_variable([8 * 8 * 64, 2048])
b_fc1 = bias_variable([2048])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

###########
# Dropout #
###########

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer

W_fc2 = weight_variable([2048, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#########
# Train #
#########

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

# Backpropagation #
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# Backpropagation #

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

datasets = ['./data/data_batch_1', './data/data_batch_2', './data/data_batch_3', './data/data_batch_4','./data/data_batch_5']

epoch = 10
batch_size = 500
temp = 10000/batch_size

t1 = time.time()
for k in xrange(20):
	print '\nglobal epoch %d' %(k+1)
	for j in xrange(5):
		start_index = 0
		images, labels = load_data.load_data(datasets[j])
		print '\n%d data batch' %(j+1)
		for i in xrange(epoch * temp):
			batch_x, batch_y, start_index = load_data.next_batch(images, labels, batch_size, start_index)
			_, loss_val = sess.run([train_step, cross_entropy], feed_dict={x:batch_x, y_:batch_y, keep_prob: 0.8})
			if i%temp == 0:
				print 'epoch %d , loss = %s' % (i/temp, loss_val)

del images, labels

print '\nEnd training.\n'
t2 = time.time()

test_x, test_y = load_data.load_data('./data/test_batch')

print 'test accuracy %g' %accuracy.eval(feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
print 'Spends %f seconds.' % (t2 - t1)





































