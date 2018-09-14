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
from settings import sudokuDigitFolder, CONSOLE_DISPLAY_IMAGE
import os

################################################# GENERAL
class Display:
	images=None
	graphicalUserInterface=None

	def __init__(self,graphicalUserInterface=False):
		self.graphicalUserInterface=graphicalUserInterface
		self.images=[]

	def displayImage(self,image,title="Sudoku Puzzle"):
		if(not CONSOLE_DISPLAY_IMAGE and not self.graphicalUserInterface):
			return
		if(self.graphicalUserInterface):
			self.images.append(image)
		else:
			cv2.imshow(title,image)
			cv2.waitKey(0)

def getFilesInDirectory(directory):
	return [os.path.join(directory,f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]


#this method checks if the training/testing directory exists, and recursively create the missing directory according to path if it doesn't
def checkIfDirectoryExists(trainingDirectory):
	if(trainingDirectory):
		dataPath=os.path.join(sudokuDigitFolder,"training")
	else:
		dataPath=os.path.join(sudokuDigitFolder,"testing")

	folders=["unsorted","None","1","2","3","4","5","6","7","8","9"]
	for folder in folders:
		currentPath=os.path.join(dataPath,folder)
		if(not os.path.exists(currentPath)):
			os.makedirs(currentPath)

#save sudoku digits to directory for training/testing
def save(digits,sudokuImageFilename,findDigitViaLargestFeature,trainingDirectory=True):
	checkIfDirectoryExists(trainingDirectory=True)
	if(trainingDirectory):
		datasetPath=os.path.join(sudokuDigitFolder,"training/unsorted")
	else:
		datasetPath=os.path.join(sudokuDigitFolder,"testing/unsorted")

	counter=0

	sudokuImageFilename=sudokuImageFilename.split(".")
	sudokuImageFilename=sudokuImageFilename[0]


	for digit in digits:
		if(findDigitViaLargestFeature):
			filename=sudokuImageFilename+"_"+str(counter)+".jpg"
		else:
			filename=sudokuImageFilename+"_"+str(counter)+"_lc"+".jpg"
		cv2.imwrite(os.path.join(datasetPath,filename),digit)
		counter+=1

################################################# ExtractSudokuPuzzle and ExtractSudokuCells

"""
This method finds the largest connected pixel structure in image and returns the seed of it
"""
def findLargestFeatureInImage(image,topLeft=None,bottomRight=None):
	img=deepcopy(image)

	height,width=image.shape[:2]

	if(topLeft is None):
		        #(x,y)
		topLeft=(0,0)
	if(bottomRight is None):
		             #(x,y)
		bottomRight=(width,height)


	if(bottomRight[0]-topLeft[0]>width or bottomRight[1]-topLeft[1]>height):
		raise ValueError("Error in findLargestFeatureInImage: coordinate of topLeft and bottomRight cannot be larger than the image it")


	maximumArea=0
	seed=None

	for y in range(topLeft[1],bottomRight[1]):
		for x in range(topLeft[0],bottomRight[0]):
			if(img.item(y,x)==255 and x<width and y<height):
				#flood fill current feature with grey
				featureArea=cv2.floodFill(img,None,(x,y),64)
				if(featureArea[0]>maximumArea):
					maximumArea=featureArea[0]
					seed=(x,y)

	feature, cornerPoints=computeBoundingBoxOfFeature(image,seed,boundingBox=False)

	return feature,cornerPoints,seed

"""
This extraction method for target feature in image is an upgrade from my original method of findDigitInSudokuCell (below) which relies heavily on my domain knowledge.
This extraction method significantly improves the precision of convolutional neural network to recognize digit in image,
but can sometimes work terribly bad, especially when the digit in sudoku cell is not connected after applying filter.
In times like these, we will resort to findDigitInSudokuCell method which doesn't have problem like this.

Reference to https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2

This method returns a box (or corner points if includeBoundingBox is False) bounding the connected pixels on (x,y)
1. We will first flood fill all features in the given image with grey
2. We then flood fill all the zero valued pixels in the same connected component as seed with white
3. We then flood fill all those grey pixels with black (Recall that we just want the target feature in our image)
4. By loop through all the pixels, we can examine the whites and compute the corner points or bounding box of it
"""
def computeBoundingBoxOfFeature(image,seed,boundingBox=True):
	sudokuImage=deepcopy(image)
	height,width=image.shape[:2]
	#initialize mask as zero arrays that have 2 pixels wider and taller than input image, as from OpenCV documentation
	#https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill#floodfill
	mask=np.zeros((height+2,width+2),np.uint8)

	#flood fill all features with grey
	for y in range(height):
		for x in range(width):
			if(sudokuImage.item(y,x)==255 and x<width and y<height):
				cv2.floodFill(sudokuImage,None,(x,y),64)

	#flood fill all zero valued pixels in the same connected component as seed with white
	#after this step only the target feature will be white in the input image, everything else will be black/grey
	if(seed is not None):
		if(seed[0] is not None and seed[1] is not None):
			cv2.floodFill(sudokuImage,mask,seed,255)

	#we initialize our corner points to be the opposite of their points, 
	#for example, the coordinates for top left will be coordinates of bottom right, coordinates of top right will be coordinates of bottom left and so on
	#this allows us to make the safe assumption that if the area is negative or if the area is small, there is no digit in the image
	topLine=height; bottomLine=0; leftLine=width; rightLine=0
	topLeft=(width,height); topRight=(0,height); bottomLeft=(width,0); bottomRight=(0,0)

	for y in range(height):
		for x in range(width):
			if(sudokuImage.item(y,x)==64):
				#colour everything that is not our target feature with black colour
				cv2.floodFill(sudokuImage,mask,(x,y),0)
			#if this is our target feature, we compute the corner point/bounding box of it
			if(sudokuImage.item(y,x)==255):
				if(boundingBox):
					if(x<leftLine):
						leftLine=x

					if(x>rightLine):
						rightLine=x

					if(y<topLine):
						topLine=y

					if(y>bottomLine):
						bottomLine=y
				else:
					##Idea:
					#our initial topLeft (width,height) has the max sum of the image. topLeft should be coordinates with minimum sum
					if(x+y<sum(topLeft)):
						topLeft=(x,y)
					#our initial bottomRight (0,0) has the minimum sum of the image. bottomRight should be coordinates with maximum sum
					if(x+y>sum(bottomRight)):
						bottomRight=(x,y)
					#our initial topRight (0,height) has the minimum difference between x and y of the image. topRight should have maximum difference between x and y
					if(x-y>topRight[0]-topRight[1]):
						topRight=(x,y)
					#our initial bottomLeft (width,0) has the maximum difference between x and y of the image. bottomLeft should have minimum difference between x and y
					if(x-y<bottomLeft[0]-bottomLeft[1]):
						bottomLeft=(x,y)


	if boundingBox:
		topLeft=(leftLine,topLine)
		bottomRight=(rightLine,bottomLine)
		cornerPoints=np.array([topLeft,bottomRight],dtype="float32")
	else:
		cornerPoints=np.array([topLeft,topRight,bottomRight,bottomLeft],dtype="float32")

	return sudokuImage,cornerPoints

"""
This method finds digit in sudoku cell based on the largest bounding box of countour and domain knowledge.
"""
def findDigitInSudokuCell(sudokuCell,re_extract=False):
	_,contours,hierarchy=cv2.findContours(sudokuCell,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	maxArea=0
	cellX=None;cellY=None;featureWidth=None;featureHeight=None
	for contour in contours:
		x,y,w,h=cv2.boundingRect(contour)
		area=w*h 						 
		#these magic numbers are being obtained through hundreds of trial and error
		if(x>5 and y>1 and area>maxArea and area>65 and ((True) if not re_extract else (x<38 and y<38)) ):
			maxArea=area
			cellX=x; cellY=y; featureWidth=w; featureHeight=h

	if(maxArea>0 and not re_extract):
		if(cellX+featureWidth>=49 or cellY+featureHeight>=49):
			height, width=sudokuCell.shape[:2]
			return findDigitInSudokuCell(sudokuCell[6:height,6:width],re_extract=True)

	if(maxArea==0):
		return None
	else:
		return sudokuCell[cellY:cellY+featureHeight,cellX:cellX+featureWidth]

################################################# Backtracking Solver and Integer Programming Solver

def convert1DSudokuPuzzleTo2D(arr):
	sudokuPuzzle=[]
	currentRow=[]
	emptyEntries=[]
	for i in range(81):
		if(i>0 and i%9==0):
			sudokuPuzzle.append(currentRow)
			currentRow=[]

		if(arr[i]==0):
			currentRow.append(None)
		else:
			currentRow.append(arr[i])

	sudokuPuzzle.append(currentRow)
	return sudokuPuzzle

def findEmptyEntries(arr):
	emptyEntries=[]

	for i in range(9):
		for j in range(9):
			if(arr[i][j]==None):
				emptyEntries.append((i,j))

	return emptyEntries

def printSudokuPuzzle(arr,emptyEntries):

	print("|――――――――――――――|――――――――――――――|―――――――――――――|")
	for i in range(9):
		for j in range(9):
			if(j%3==0):
				if((i,j) not in emptyEntries):
					print("|"+"  ",end="")
					currentEntryEmpty=False
				else:
					print("|"+" [",end="")
					currentEntryEmpty=True

			if(not currentEntryEmpty):
				print(str(arr[i][j])+"  ",end="")
			else:
				print(str(arr[i][j])+"] ",end="")

			if(j<8):
				if( (j+1)%3!=0 and (i,j+1) in emptyEntries ):
					print("[",end="")
					currentEntryEmpty=True
				else:
					print(" ",end="")
					currentEntryEmpty=False
			#else if j==8
			else:
				print("|",end="")

		print("")

		if( (i+1)%3 ==0 ):
			print("|――――――――――――――|――――――――――――――|―――――――――――――|")
