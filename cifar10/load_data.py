import numpy as np
import cPickle

#############
# load data #
#############

def unpickle(file):
	f = open(file, 'rb')
	dic = cPickle.load(f)
	f.close()
	return dic

def extract_images(images, rows, cols, channels):
	num_images = images.shape[0]
	images = images.reshape(num_images, rows, cols, channels)
	images = images.astype(np.float32) / 255.0
	return images
	

def dense_to_one_hot(labels_dense, num_classes):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

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
        labels = np.asarray(data['labels'])
        labels = dense_to_one_hot(labels, 10)
        return labels

# images, labels = load_data('./data/data_batch_1')

# start index
def next_batch(images, labels, batch_size, start):
	num = labels.shape[0]
	temp = start
	end = temp + batch_size
	start += batch_size
	start %= num
	return images[temp:end], labels[temp:end], start






















