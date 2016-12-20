import numpy as np
import cPickle
import gzip

#############
# load data #
#############

# cifar10
def extract_images(images, rows, cols, channels):
	num_images = images.shape[0]
	images = images.reshape(num_images, rows, cols, channels)
	images = images.astype(np.float32) / 255.0
	return images

# mnist
def extract_mnist(images, rows, cols, channels):
	num_images = images.shape[0]
	images = images.reshape(num_images, rows, cols, channels)
	return images	

def dense_to_one_hot(labels_dense, num_classes):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

# Load the dataset for mnist
def unpickle_mnist(file):
	f = gzip.open(file, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()
	return train_set, valid_set, test_set

def load_mnist(file):
	train_set, valid_set, test_set = unpickle_mnist(file)
	images = np.asarray(train_set[0], np.float32)
	labels = np.asarray(train_set[1], int)
	test_img = np.asarray(test_set[0], np.float32)
	test_lab = np.asarray(test_set[1], int)
	return images, labels, test_img, test_lab

def load_mnist_images(file):
	images, labels, test_img, test_lab = load_mnist(file)
	num_images = images.shape[0]
	images = images.reshape(num_images, 28, 28, 1)
	labels = dense_to_one_hot(labels, 10)
	return images, labels

def load_mnist_test(file):
	images, labels, test_img, test_lab = load_mnist(file)
	num_images = test_img.shape[0]
	test_img = test_img.reshape(num_images, 28, 28, 1)
	test_lab = dense_to_one_hot(test_lab, 10)
	return test_img, test_lab

# Load the dataset for cifar10
def unpickle(file):
	f = open(file, 'rb')
	dic = cPickle.load(f)
	f.close()
	return dic

def load_data(file):
	data1 = unpickle(file)
	'''
	print '\ndata_batch_1 keys:'
	for k in data1:
		print k
	'''
	images = np.asarray(data1['data'])
	labels = np.asarray(data1['labels'])

	images = extract_images(images, 32, 32, 3)
	labels = dense_to_one_hot(labels, 10)

	return images, labels

def load_images(file):
	data = unpickle(file)
	images = np.asarray(data['data'])
	images = extract_images(images, 32, 32, 3)
	return images

def load_labels(file):
        data = unpickle(file)
        labels = np.asarray(data['lables'])
        labels = dense_to_one_hot(labels, 10)
        return images


# start index
def next_batch(images, labels, batch_size, start):
	num = labels.shape[0]
	temp = start
	end = temp + batch_size
	start += batch_size
	start %= num
	return images[temp:end], labels[temp:end], start






















