#!/usr/bin/env python3

"""
@author: Foo Zhi Yuan
Sudoku Image Solver uses image processing techniques to extract sudoku puzzle and Convolutional Neural Network for digit parsing, then solve parsed sudoku puzzle
using Linear Programming.
Requires Python 3, OpenCV (for image processing), TensorFlow (for building ConvolutionalNN), NumPy, PuLP (for Linear Programming) and Pillow(for GUI)
USAGE: python3 sudokuImageSolver.py to launch in GUI and python3 sudokuImageSolver_console.py to launch in console
"""

import tensorflow as tf

from ConvolutionalNN.dataset import *
from ConvolutionalNN.convolutionalNN import ConvolutionalNN
from settings import modelPath,DIGIT_SIZE
from copy import deepcopy
import os

class DigitRecognition:
	def predict(self,sudokuDigits,normalize=True,flatten=True,digitSize=DIGIT_SIZE,threshold=0,trainedModelPath=None):
		if(threshold<0):
			raise ValueError("Error in predict: threshold needs to be >= 0")
		if(trainedModelPath is None):
			trainedModelPath=os.path.join(modelPath,"model.ckpt")

		sudokuDigits=deepcopy(sudokuDigits)
		digits=preprocessSudokuDigitsAndLabels(digits=sudokuDigits,labels=None,normalize=normalize,flatten=flatten,digitSize=DIGIT_SIZE,readFromPath=False)
		tf.reset_default_graph()

		convolutionalNN=ConvolutionalNN()
		saver=tf.train.Saver()

		with tf.Session() as session:
			session.run(tf.global_variables_initializer())

			try:
				saver.restore(session,trainedModelPath)
				print("Model restored.")
			except:
				raise FileNotFoundError("Error in predict: No tensorflow model is found in modelPath. Cannot predict sudoku digits. Try training the ConvolutionalNN, save the final model in modelPath and recall this method again.")
			
			if(threshold>0):
				#probability of the sudoku digit being [None,1,2,3,4,5,6,7,8,9] for each sudoku digits
				probabilities=(convolutionalNN.y_predict).eval(feed_dict={convolutionalNN.x:digits, convolutionalNN.keep_prob:1.0},session=session)
				#An array storing index of element having highest probabilities for each sudoku digits
				predictions=np.argmax(probabilities,1)

				#filter prediction by checking if it's larger than threshold.
				#if a prediction is smaller than threshold, treat it as None instead
				filtered_predictions=[]

				for i, prediction in enumerate(predictions):
					#i-th prediction=[0.85, 0.1, 1, 0.1, 0, 0, 0, 0, 0, 0]
					#					^                      ^
					#					|					   |
					#				probability for None	probability for 6

					if(probabilities[i][prediction]>threshold):
						filtered_predictions.append(prediction)
					else:
						filtered_predictions.append(0)

				result=np.array(filtered_predictions)

			else:
				prediction=tf.argmax(convolutionalNN.y_predict,1)
				result=prediction.eval(feed_dict={convolutionalNN.x:digits, convolutionalNN.keep_prob:1.0},session=session)

		return result




