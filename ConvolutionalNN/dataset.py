#!/usr/bin/env python3

"""
@author: Foo Zhi Yuan
Sudoku Image Solver uses image processing techniques to extract sudoku puzzle and Convolutional Neural Network for digit parsing, then solve parsed sudoku puzzle
using Linear Programming.
Requires Python 3, OpenCV (for image processing), TensorFlow (for building ConvolutionalNN), NumPy, PuLP (for Linear Programming) and Pillow(for GUI)
USAGE: python3 sudokuImageSolver.py to launch in GUI and python3 sudokuImageSolver_console.py to launch in console
"""

import cv2
import numpy as np
import os
import Helper.helper as Helper
from settings import sudokuDigitFolder,DIGIT_SIZE


																						#if readFromPath is true, digits must be an array of paths
																						#that this method will open up (using OpenCV).
																						#Otherwise, digits must be an array of images that have already been
																						#openup using OpenCV's imread.
def preprocessSudokuDigitsAndLabels(digits,labels,normalize,flatten,digitSize=DIGIT_SIZE,readFromPath=True):
	if(labels is not None):
		if(len(digits)!=len(labels)):
			raise AssertionError("Error in preprocessSudokuDigitsAndLabels: length of sudoku digits and labels need to be the same.")

	preprocessedDigits=[]
	preprocessedLabels=[]

	for digit in digits:

		if(readFromPath):
			currentDigit=cv2.imread(digit,cv2.IMREAD_GRAYSCALE)
		else:
			currentDigit=digit

		if(len(currentDigit.flatten())!=digitSize**2):
			currentDigit=cv2.resize(currentDigit,(digitSize,digitSize))
		if(normalize):
			cv2.normalize(currentDigit,currentDigit,0,255,cv2.NORM_MINMAX)
		if(flatten):
			currentDigit=currentDigit.flatten()

		preprocessedDigits.append(np.float32(currentDigit))
	del digits

	if(type(preprocessedDigits) is list):
		preprocessedDigits=np.array(preprocessedDigits)

	#if argument label is None (This method is being called by DigitRecognition class to predict sudoku digit), just return the preprocessed sudoku digits
	if(labels is None):
		return preprocessedDigits

	#else return preprocessed sudoku digits and labels
	def convertLabelToOneHotVector(label):
		oneHotVector=[0.]*10
		if(label=="None"):
			oneHotVector[0]=1.
		elif(int(label)>0 and int(label)<10):
			oneHotVector[int(label)]=1.
		else:
			raise ValueError("Error in convertLabelToOneHotVector: Argument label needs to be string 'None' or string of value between 1 and 9")
		return oneHotVector


	preprocessedLabels=[convertLabelToOneHotVector(label) for label in labels]
	del labels

	if(len(preprocessedDigits)!=len(preprocessedLabels)):
		raise AssertionError("Error in preprocessSudokuDigitsAndLabels: length of digits and labels need to be the same.")
	if(type(preprocessedLabels) is list):
		preprocessedLabels=np.array(preprocessedLabels)

	return preprocessedDigits,preprocessedLabels


class Dataset:
	digits=None
	labels=None
	permutatedDigits=None
	permutatedLabels=None
	datasetSize=0
	currentCounter=0

	"""
	This method reads in all the sudoku digits in training/testing directory (it assumes that there is None,1,2,3,4,5,6,7,8,9 folders inside training/testing directory),
	create label for each image and randomize the entire dataset
	@param trainingDirectory: boolean indicating if we are creating dataset for training or testing directory
	@param normalize: boolean indicating if the sudoku digits need to be normalize
	@param flatten: boolean indicating if the sudoku digits need to be flatten
	@param digitSize: int indicating the size of sudoku digits
	"""
	def __init__(self,trainingDirectory,normalize=True,flatten=True,digitSize=DIGIT_SIZE):

		Helper.checkIfDirectoryExists(trainingDirectory=True)
		Helper.checkIfDirectoryExists(trainingDirectory=False)

		if(trainingDirectory):
			dataPath=os.path.join(sudokuDigitFolder,"training")
		else:
			dataPath=os.path.join(sudokuDigitFolder,"testing")

		folders=["None","1","2","3","4","5","6","7","8","9"]
		digits=[]
		labels=[]

		for folder in folders:
			currentPath=os.path.join(dataPath,folder)
			currentPathFiles=Helper.getFilesInDirectory(currentPath)
			currentLabels=[folder]*len(currentPathFiles)

			digits.extend(currentPathFiles)
			labels.extend(currentLabels)

		digits,labels=preprocessSudokuDigitsAndLabels(digits=digits,labels=labels,normalize=normalize,flatten=flatten,digitSize=digitSize,readFromPath=True)

		self.datasetSize=len(digits)
		self.digits=digits
		self.labels=labels

		randomized=np.random.permutation(self.datasetSize)
		self.permutatedDigits=digits[randomized]
		self.permutatedLabels=labels[randomized]
		self.currentCounter=0


	"""
	perform similarly to tensorflow's next_batch method, in which the method will fetch "num" amount of (unused,randomized) digits and the corresponding labels
	If all the datasets have been used, the dataset will be randomized again and the first "num" amount of digits and labels will be returned
	"""
	def next_batch(self,num):

		if(self.currentCounter+num>self.datasetSize):
			difference=self.currentCounter+num-self.datasetSize
			temp1_digits=self.permutatedDigits[self.currentCounter:]
			temp1_labels=self.permutatedLabels[self.currentCounter:]

			randomized=np.random.permutation(self.datasetSize)
			self.permutatedDigits=self.digits[randomized]
			self.permutatedLabels=self.labels[randomized]
			
			temp2_digits=self.permutatedDigits[:difference]
			temp2_labels=self.permutatedLabels[:difference]

			nextBatch_digits=np.concatenate((temp1_digits,temp2_digits))
			nextBatch_labels=np.concatenate((temp1_labels,temp2_labels))

			self.currentCounter=difference

			return nextBatch_digits,nextBatch_labels


		endingIndex=self.currentCounter+num

		nextBatch_digits=self.permutatedDigits[self.currentCounter:endingIndex]
		nextBatch_labels=self.permutatedLabels[self.currentCounter:endingIndex]

		self.currentCounter=endingIndex

		return nextBatch_digits,nextBatch_labels




