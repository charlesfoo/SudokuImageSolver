#!/usr/bin/env python3

"""
@author: Foo Zhi Yuan
Sudoku Image Solver uses image processing techniques to extract sudoku puzzle and Convolutional Neural Network for digit parsing, then solve parsed sudoku puzzle
using Linear Programming.
Requires Python 3, OpenCV (for image processing), TensorFlow (for building ConvolutionalNN), NumPy, PuLP (for Linear Programming) and Pillow(for GUI)
USAGE: python3 sudokuImageSolver.py to launch in GUI and python3 sudokuImageSolver_console.py to launch in console
"""

import tensorflow as tf
from settings import modelPath
from ConvolutionalNN.dataset import Dataset
import os

## Multilayer Convolutional Neural Network
#Reference to https://www.tensorflow.org/get_started/mnist/pros

class ConvolutionalNN:
	x=None
	y_label=None
	y_predict=None
	keep_prob=None

	modelPath=None

	def __init__(self):

		self.checkIfModelDirectoryExists()

		#################### Initialize Input Variable ####################
		#x and y_ are not specific values. They are each a placeholder,
		#which the value we will input when we ask TensorFlow to run computation

		#x consists of a 2d tensor of floating point numbers.
		#784 is the dimension of a single flattened 28x28 MNIST image
		#None means the first dimension (indicating the batch size), can be of any size.
		x=tf.placeholder(tf.float32,shape=[None,784])
		#stores the label of golden data
		y_=tf.placeholder(tf.float32,shape=[None,10])

		#################### First Convolutional Layer ####################

		#Convolutional Layer -> add bias -> RELU -> Max Pool
		#the convolution computes 32 features for each 5x5 patch
							   #patch size   num input channels    num output channels
		W_conv1=self.weight_variable([  5,5,            1,                         32])
		#bias vector with a component for each output channel
		b_conv1=self.bias_variable([32])

		#To apply the first convolutional layer to input x, we first reshape x to a 4d tensor
		#with second and third dimension be image width and height respectively, and final dimension be num of colour channels
		x_image=tf.reshape(x,[-1,28,28,1])

		#convolve x_image with weight tensor, add bias and apply ReLU
		h_conv1=tf.nn.relu(self.conv2d(x_image,W_conv1)+b_conv1)
		#apply max pool (size 2x2) which reduces the image size to 14x14
		h_pool1=self.max_pool_2x2(h_conv1)


		#################### Second Convolutional Layer ####################

		#stack another layer similar to our first layer
		#the second layer will have 64 features for each 5x5 patch
								#patch size   num input channels    num output channels
		W_conv2=self.weight_variable([5,5,               32,                    64])
		#bias vector with a component for each output channel
		b_conv2=self.bias_variable([64])

		#convolve h_pool1 with weight tensor, add bias and apply ReLU
		h_conv2=tf.nn.relu(self.conv2d(h_pool1,W_conv2)+b_conv2)
		#apply max pool (size 2x2) which reduces the image size to 7x7
		h_pool2=self.max_pool_2x2(h_conv2)


		#################### Fully Connected Layer ####################

		#Now our image size has been reduced to 7x7
		#We add a fully connected layer with 1024 neurons to process on the entire image.
		W_fc1=self.weight_variable([7*7*64,1024])
		b_fc1=self.bias_variable([1024])

		#reshape tensor from pooling layer into a batch of vectors
		h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
		#multiply flatten pooling layer (now vector) with weight matrix, add a bias and apply ReLU
		h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)


		#################### Dropout ####################

		#To reduce overfitting, we apply dropout before the readout layer.
		#create a placeholder for the probability that a neuron's output is kept during dropout
		keep_prob=tf.placeholder(tf.float32)
		h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)


		#################### Readout Layer ####################

		#add a layer like the the one for softmax regression
		W_fc2=self.weight_variable([1024,10])
		b_fc2=self.bias_variable([10])

		
		"""
		Regression model
		W2,1 means weight of x1 to b2 

		[y1]			[W1,1*x1 + W1,2*x2 + W1,3*x3 + b1 ]
		[y2]  = softmax [W2,1*x1 + W2,2*x2 + W2,3*x3 + b2 ]
		[y3]			[W3,1*x1 + W3,2*x2 + W3,3*x3 + b3 ]

		    			  [W1,1   W1,2   W1,3]	 [x1]	 [b1]
		      = softmax ( [W2,1   W2,2   W2,3] * [x2]  + [b2] )
						  [W3,1   W3,2   W3,3]	 [x3]	 [b3]

		More compactly, y=softmax(Wx + b)
		"""
		y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2

		self.x=x
		self.y_label=y_
		self.y_predict=y_conv
		self.keep_prob=keep_prob


	#Weight initialization
	def weight_variable(self,shape):
		#initialize weights with a small amount of noise for symmetry breaking and to prevent 0 gradients
		initial=tf.truncated_normal(shape,stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		#initialize ReLU neurons with a slightly positive  initial bias to avoid dead neurons
		initial=tf.constant(0.1,shape=shape)
		return tf.Variable(initial)

	#Convolution and pooling
	def conv2d(self,x,W):
		#convolves with stride of one and are zero padded so that the output is the same size as input
		return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

	def max_pool_2x2(self,x):
		#classic max pooling over 2x2 blocks (strides same length as filter)
		return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	def compute_accuracy(self,y_predict,y_label):
		### Evaluating the model
		#tf.argmax(input, axis=None, name=None, dimension=None) returns index with largest value across axis of a tensor
		#tf.argmax(y,1) is the label our model think is most likely for each input
		#tf.argmax(y_,1) is the true label. The reason is because y_ is a one hot vector.
		correct_prediction=tf.equal(tf.argmax(y_predict,1),tf.argmax(y_label,1))
		#correct_prediction gives us a list of booleans. If we have 100 images in dataset,
		#we will have an array of 100 elements.
		#eg:[True,False,True,True]
		#to determine what fraction are correct, we cast to floating points,which would become [1,0,1,1]
		#and take the mean, which in this case, will become 0.75
		accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

		return accuracy

	#this method checks if the directories in model path exists, and recursively create the missing directory according to path if it doesn't
	def checkIfModelDirectoryExists(self):
		if(not os.path.exists(modelPath)):
			os.makedirs(modelPath)
		self.modelPath=os.path.join(modelPath,"model.ckpt")

	"""
	train the model for ConvolutionalNN using trainingSet, and save the model to modelPath.
	@param trainingSet: type Dataset storing training set's sudoku digits and labels
	@param steps: type int specifying number of iteration for training
	@param batchSize: type int specifying number of training set to be trained per iteration
	"""
	def train(self,trainingSet,steps,batchSize=50):
		#reduce loss function
		cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_label,logits=self.y_predict))
		#we replace our gradient descent optimizer in softmax regression with more sophisticated ADAM optimizer
		train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

		accuracy=self.compute_accuracy(self.y_predict,self.y_label)

		#Add ops to save and restore all variables
		saver=tf.train.Saver()

		with tf.Session() as session:
			session.run(tf.global_variables_initializer())

			try:
				saver.restore(session,self.modelPath)
				print("Model restored.")
			except:
				pass

			for i in range(steps):
				batch=trainingSet.next_batch(batchSize)

				#for every 100th iteration, we print the accuracy of the batch
				if(i%100==0):
					train_accuracy=accuracy.eval(feed_dict={self.x:batch[0],self.y_label:batch[1],self.keep_prob:1.0})
					print("step %d, training accuracy %g" %(i,train_accuracy))

				#load "batchSize" numbers of training examples in each iteration, then run train_step operation, which applies ADAM update to the parameter.
				#We use feed_dict to replace placeholder tensors x and y_label
				train_step.run(feed_dict={self.x:batch[0],self.y_label:batch[1],self.keep_prob:0.5})

			save_path=saver.save(session,self.modelPath)
			print("Model saved in path: %s" %(save_path))

	"""
	Test the ConvolutionalNN model trained on testSet and compute the accuracy
	@param testSet: type Dataset storing test set's sudoku digits and labels
	"""
	def test(self,testSet):
		accuracy=self.compute_accuracy(self.y_predict,self.y_label)

		saver=tf.train.Saver()

		with tf.Session() as session:
			session.run(tf.global_variables_initializer())

			try:
				saver.restore(session,self.modelPath)
				print("Model restored.")
			except:
				raise FileNotFoundError("Error in test: No tensorflow model is found in modelPath. Cannot test accuracy on test set.")

			testSetAccuracy=accuracy.eval(feed_dict={self.x:testSet.digits,self.y_label:testSet.labels,self.keep_prob:1.0})
			print("test accuracy %g" %(testSetAccuracy))