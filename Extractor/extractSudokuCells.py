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
from copy import deepcopy
import Helper.helper as Helper
from settings import *

class ExtractSudokuCells:
	preprocessedExtracted=None
	postProcessedExtracted=None
	findDigitViaLargestFeature=True
	displayImage=True


	def __init__(self,preprocessedExtracted,postProcessedExtracted,display):
		self.preprocessedExtracted=preprocessedExtracted
		self.postProcessedExtracted=postProcessedExtracted
		self.display=display

	def run(self,findDigitViaLargestFeature=True,displayImage=True):
		self.findDigitViaLargestFeature=findDigitViaLargestFeature
		self.displayImage=displayImage

		cellPositions=self.computePositionOfSudokuPuzzleCells(self.preprocessedExtracted,self.postProcessedExtracted)
		sudokuDigits=self.extractSudokuPuzzleCells(cellPositions,self.postProcessedExtracted)

		return sudokuDigits


	def __displaySudokuPuzzleCellsGrid(self,cellPositions,sudokuPuzzle):
		for position in cellPositions:

			startX,endX,startY,endY=position
			cv2.rectangle(sudokuPuzzle,(startX,startY),(endX,endY),(255,0,0),3)

		if(self.displayImage):
			self.display.displayImage(sudokuPuzzle)


	"""
	Compute the position of each cell in sudoku puzzle in the form of [startX,endX,startY,endY], and store it in an array of array 
	This method then calls to private method displaySudokuPuzzlecellsGrid to display all the positions of the cell computed
	"""	
	def computePositionOfSudokuPuzzleCells(self,preprocessedExtracted,postProcessedExtracted):
		preprocessedExtracted=deepcopy(preprocessedExtracted)

		#cell position takes the form of [startX,endX,startY,endY]
		cellPositions=[]

		sudokuPuzzleWidth=postProcessedExtracted.shape[1]
		sudokuPuzzleHeight=postProcessedExtracted.shape[0]

		cellWidth=sudokuPuzzleWidth//9
		cellHeight=sudokuPuzzleHeight//9

		startX=0;endX=0;startY=0;endY=0

		for row in range(9):
			endY=startY+cellHeight
			startX=0

			for column in range(9):
				endX=startX+cellWidth
				currentCellPosition=[startX,endX,startY,endY]
				cellPositions.append(currentCellPosition)
				startX=endX

			startY=endY

		self.__displaySudokuPuzzleCellsGrid(cellPositions,preprocessedExtracted)

		return cellPositions

	"""
	1. perform gaussian filter to remove noise 
	2. perform erosion, then dilation to further remove noises in the image
	3. perform adaptive threshold to turn the image into black and white (binary image)
	"""
	def preprocessSudokuPuzzleCell(self,sudokuCell):

		denoised=cv2.fastNlMeansDenoising(src=sudokuCell,h=10,templateWindowSize=9,searchWindowSize=13)
		#Perform guassian blur to remove noise 
		# blurred=cv2.GaussianBlur(denoised,(5,5),3)
		blurred=cv2.bilateralFilter(src=denoised,d=15,sigmaColor=40,sigmaSpace=40) 

		threshold=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,7,3) 
		kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
		#perform dilation then erosion to removes small holes and smoothen the contour in the image
		#(can set iterations for dilate to be 2 to increase the size of digit for better recognition)
		dilate=cv2.dilate(threshold,kernel,iterations=1)
		erode=cv2.erode(dilate,kernel,iterations=1)

		return erode
		

	"""
	This method takes in sudoku digit, resize it to intended size and pad it with black at the sides
	"""
	def centerAndResizeSudokuDigit(self,sudokuCell,intendedSize,padding):
		if(intendedSize%2!=0):
			raise ValueError("Error in centerAndResizeSudokuDigit: Argument intendedSize must be even.")
		if(2*padding>=intendedSize):
			raise ValueError("Error in centerAndResizeSudokuDigit: Padding cannot be larger or equals to intended size of image.")


		def normalizePadding(length, targetLength):
			if(length%2==0):
				padding_1 = padding_2 = (intendedSize-length)//2
			else:
				padding=(intendedSize-length)//2
				if(length+padding*2+1 > targetLength):
					padding_1=padding
					padding_2=padding
				else:
					padding_1=padding
					padding_2=padding+1

			return padding_1, padding_2


		height,width=sudokuCell.shape[:2]
		if(width>height):
			leftPadding=padding; rightPadding=padding
			ratio=(intendedSize-2*padding)/width
			width, height=int(ratio*width), int(ratio*height)
			if(height==0):
				height=1
			sudokuCell=cv2.resize(sudokuCell,(width,height))
			topPadding, bottomPadding=normalizePadding(height,width+2*padding)
		#if height>width
		else:
			topPadding=padding; bottomPadding=padding
			ratio=(intendedSize-2*padding)/height
			width, height=int(ratio*width), int(ratio*height)
			if(width==0):
				width=1
			sudokuCell=cv2.resize(sudokuCell,(width,height))
			leftPadding, rightPadding=normalizePadding(width,height+2*padding)
																					  									#set padding colour to be black
		sudokuCell=cv2.copyMakeBorder(sudokuCell, topPadding, bottomPadding, leftPadding, rightPadding, cv2.BORDER_CONSTANT, None, 0)
		sudokuCell=cv2.resize(sudokuCell,(intendedSize,intendedSize))
		return sudokuCell



	def findAndExtractSudokuDigit(self,sudokuCell, re_extract=False):
		#we start looking at the middle of the cell as this is where the sudoku digit should be at
		height, width=sudokuCell.shape[:2]
		index=int(np.mean([height,width])/2.5)

		if(not re_extract):
			_, boundingBox, seed=Helper.findLargestFeatureInImage(sudokuCell,[index-2,index-2],[width-index,height-index])
		else:
			_, boundingBox, seed=Helper.findLargestFeatureInImage(sudokuCell,[index,index],[width-index,height-index])
			
		feature, cornerPoints=Helper.computeBoundingBoxOfFeature(sudokuCell,seed,boundingBox=True)
		topLeft, bottomRight=cornerPoints
		startX, endX, startY, endY=topLeft[0], bottomRight[0], topLeft[1], bottomRight[1]
		width=endX-startX;  height=endY-startY

		#if we can't detect any feature in the given cell, dilate the features in the cell and detect again
		if(width<0 and height<0 and not re_extract):
			height, width=sudokuCell.shape[:2]
			kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
			dilate=cv2.dilate(sudokuCell[6:height,6:width],kernel,iterations=1)
			return self.findAndExtractSudokuDigit(dilate,re_extract=True)

																		#to detect digit 1
		if( (width>0 and height>0 and (width*height)>100) or (width>1 and height>13 and (width*height)>45 and not re_extract) or 
																(width>2 and height>17 and (width*height)>75 and re_extract) ):

			feature=feature[int(startY):int(endY),int(startX):int(endX)]
			return feature
		else:
			return None


	def extractSudokuPuzzleCells(self,cellPositions,extractedSudokuPuzzle):
		sudokuDigits=[]
		#iterates from left to right, then only top to bottom, eg:
		# ------------------------->
		#             |
		#             |
		#			  ↓

		#conforms to our post processed sudoku cell format, which after being converted to grayscale, only have 2 channels.
		blank=np.zeros((DIGIT_SIZE,DIGIT_SIZE),np.uint8)

		for position in cellPositions:
			startX,endX,startY,endY=position
			currentCell=extractedSudokuPuzzle[startY:endY,startX:endX]

			currentCell=self.preprocessSudokuPuzzleCell(currentCell)

			#find and extract sudoku digit
			if(self.findDigitViaLargestFeature):
				currentCell=self.findAndExtractSudokuDigit(currentCell)
			else:
				#find digit based on largest bounding box of contour and domain knowledge
				currentCell=Helper.findDigitInSudokuCell(currentCell)

			if(currentCell is not None):
				currentCell=self.centerAndResizeSudokuDigit(currentCell,DIGIT_SIZE,2) #default size is 28 (MNIST)
				# self.display.displayImage(currentCell)
				sudokuDigits.append(currentCell)
			else:
				#an alternative will be to just include None. however, sometimes even though there is no digit in cell,
				#our digit extraction method might mistakenly think that there is digit inside due to unfiltered noises.
				#so it's better for us to create and append an empty black square whenever there is None and train our CNN to recognize it as None
				sudokuDigits.append(blank)

		return sudokuDigits
			