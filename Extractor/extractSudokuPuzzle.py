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
import Helper.helper as Helper
from copy import deepcopy
from skimage import exposure


class ExtractSudokuPuzzle:
	preprocessedExtracted=None
	postProcessedExtracted=None

	def __init__(self,colouredSudokuImage,display):
		self.display=display

		grayscale=cv2.cvtColor(colouredSudokuImage,cv2.COLOR_BGR2GRAY)
		preprocessed=self.preprocessImage(grayscale)
		
		#then find sudoku puzzle via largest contour and largest feature 
		quadrangle=self.findSudokuPuzzleGrid(preprocessed,colouredSudokuImage)

		#compute the maximum height and width of sudoku puzzle based of the 4 corners
		maxWidth,maxHeight=self.computeMaxWidthAndHeightOfSudokuPuzzle(quadrangle)

		#apply transformation to obtain bird eye(top down) view of sudoku puzzle
		warpedSudokuPuzzle=self.extractSudokuPuzzleAndWarpPerspective(quadrangle,maxWidth,maxHeight,colouredSudokuImage)

		#resize the extracted sudoku puzzle, convert it to grayscale and rescale intensity of it
		postProcessed=self.postProcessExtractedSudokuPuzzle(warpedSudokuPuzzle)

		self.preprocessedExtracted=self.postProcessExtractedSudokuPuzzle(warpedSudokuPuzzle,postProcess=False)
		self.postProcessedExtracted=postProcessed
		

	"""
	1. apply bilateral filter to removes noise while keeping the edges sharp
	2. perform dilation, then erosion to close the small holes in the object
	3. perform adaptive threshold to turn the image into black and white (binary image)
	"""
	def preprocessImage(self,sudokuImage):
		#Smooths image
		#bilateral Filter removes noise while keeping the edges sharp
		#Gaussian blur is a function of space alone, eg: nearby pixels are considered while filtering, regardless of if the pixels have almost the same intensity. 
		#Because of that, it doesn't consider whether the pixel is edge pixel or not, hence blurring the edges along the way. We don't want that!
		#Bilateral filter uses one more additional gaussian filter, called function of pixel difference.
		#This ensures only pixels with similar intensity to central pixel is considered for blurring, hence keeping the edges, since pixels at edge have large intensity variation.
		blurred=cv2.bilateralFilter(sudokuImage,5,75,75)

		kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
		#removes small holes, smoothen the contour in the image
		closed=cv2.morphologyEx(blurred,cv2.MORPH_CLOSE,kernel)
		
		div=np.float32(blurred)/(closed)
		normalized=np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
		#perform adaptive threshold to turn the image into binary image (anything that's larger than threshold get turned into different colour)
		threshold=cv2.adaptiveThreshold(normalized,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
		#invert it such that the object in white is now the lines
		threshold=cv2.bitwise_not(threshold)

		if((threshold==0).all()):
			threshold=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)

		self.display.displayImage(threshold)

		return threshold



	"""
	We assume that our sudoku puzzle has four points and has the largest grid in the image.
	Hence, by finding the largest contour which has 4 points, we can safely say that it is the sudoku puzzle that we are finding.

	This method find the largest contour in the preprocessedSudokuImage that has 4 points and return the contour
	"""
	def findLargestContour(self,preprocessedSudokuImage):
		# originalSudokuImage=deepcopy(originalSudokuImage)

		#find contours in the binary image of sudokuImage and obtain the end points of them in a list (without hierarchical relationships)
		_,contours,hierarchy=cv2.findContours(preprocessedSudokuImage,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		#the minimum area that is required in order to be considered as a potential contour for sudoku image is 300 (subject to change)
		minArea=300
		maxArea=0
		largestContour=None
		for contour in contours:
			currentArea=cv2.contourArea(contour)
			if(currentArea>minArea):
				#find the perimeter of the closed contour
				currentPerimeter=cv2.arcLength(contour,True)
				#obtain an approximate to the current contour
				currentApproximate=cv2.approxPolyDP(contour,0.02*currentPerimeter,True)
				#if current area is the largest area and it has four points (a square grid), this is the grid of sudoku image puzzle we are searching for
				if(currentArea>maxArea and len(currentApproximate)==4):
					maxArea=currentArea
					largestContour=currentApproximate

		# cv2.drawContours(originalSudokuImage,[largestContour],-1,(0,255,0),3)
		# self.display.displayImage(originalSudokuImage)

		return largestContour,maxArea


	"""
	We use two image processing method to find the grid of sudoku puzzle in the sudoku image, namely find largest contour and find largest feature.

	We first find the largest contour in the sudoku image and a bounding box bounding all children of the largest contour.
	By default, we will use the bounding box (bounding all chidlren of largest contour), however, if the area of bounding box/area of largest contour 
	is smaller than threshold, this means we have some other information in our largest contour, we will use the largest contour instead. 
	
	In addition to this, we will also find the grid of sudoku puzzle via finding the largest feature. 
	The area of sudoku puzzle found via largest countour and largest feature should be similar, if they are not, eg: if largest contour is much 
	larger than largest feature, we will use the sudoku puzzle found via largest feature.
	The reason is because find sudoku puzzle using the method of finding largest contour is susceptible to error whenever there is a box larger
	than sudoku puzzle that is bounding the sudoku puzzle. In cases like this, we will need to find sudoku puzzle by finding the largest feature in the
	image.

	With this, one might ask why can't we just find sudoku puzzle via finding largest feature. This is because find largest feature could not capture the 
	grid of sudoku puzzle perfectly, and this is better done via find largest contour.
	"""
	def findSudokuPuzzleGrid(self,preprocessedSudokuImage,originalSudokuImage):
		originalSudokuImage=deepcopy(originalSudokuImage)

		height,width=preprocessedSudokuImage.shape[:2]
		sudokuImageArea=height*width


		############################################# Find sudoku puzzle via largest contour
		largestContour,largestContourArea=self.findLargestContour(preprocessedSudokuImage)


		############################################# Find sudoku puzzle via largest feature
		feature,cornerPoints,seed=Helper.findLargestFeatureInImage(preprocessedSudokuImage)

		#cornerPoints is numpy array in float32. We want to convert it to int tuple in order to draw the rectangle box bounding the feature
		featureCornerPoints=cornerPoints.astype(int)
		featureCornerPoints=featureCornerPoints.tolist()
		topLeft,topRight,bottomRight,bottomLeft=featureCornerPoints
		topLeft=tuple(topLeft); topRight=tuple(topRight); bottomRight=tuple(bottomRight); bottomLeft=tuple(bottomLeft)

		largestFeatureArea=cv2.contourArea(cornerPoints)

		try:
			ratio=largestFeatureArea/largestContourArea
		except(ZeroDivisionError):
			if(largestFeatureArea==0):
				print("Error in findSudokuPuzzleGrid: Unable to extract sudoku puzzle from image.")
				exit()
			else:	
				ratio=0 #jumps to if block, which we will then use largestFeatureArea

		if(ratio<0.95 or ratio>1.5):
			#uses largest feature
			cv2.line(originalSudokuImage,topLeft,topRight,(0,255,0),3)
			cv2.line(originalSudokuImage,bottomLeft,bottomRight,(0,255,0),3)
			cv2.line(originalSudokuImage,topLeft,bottomLeft,(0,255,0),3)
			cv2.line(originalSudokuImage,topRight,bottomRight,(0,255,0),3)
			self.display.displayImage(originalSudokuImage)
			return cornerPoints
		else:
			#uses largest contour
			cv2.drawContours(originalSudokuImage,[largestContour],-1,(0,255,0),3)
			self.display.displayImage(originalSudokuImage)
			return self.getQuadrangleVertices(largestContour)



	"""
	Obtain the corner points of sudoku puzzle grid and store it in quadrangle
	Reference to https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/ (A great and interesting article on how to build a pokedex from scratch)
	"""
	def getQuadrangleVertices(self,sudokuGrid):
		########################convert the corners of the contour from an array of array of array to an array of array.
		#eg: from 
		#	[									[
		#	[[140  93]]							[140 93] 	
 		# 	[[126 338]]                  to     [126 338]  
 		# 	[[388 337]]							[388 337]
 		# 	[[370  98]]]						[370 98]]

		if(len(sudokuGrid)==0):
 			return None

		corners=sudokuGrid.reshape(len(sudokuGrid),2)

		# Create a rectangle of zeros in the form of 
		# [[0. 0.]     <-top left
 		# [0. 0.]	   <-top right
 		# [0. 0.]      <-bottom right
 		# [0. 0.]]     <-bottom left
		quadrangle=np.zeros((4,2),dtype="float32")

		#turn corners into a single dimension by summing the two values in each array up
		s=corners.sum(axis=1)
		#initialize rectangle to be in the order of top left, top right, bottom right, bottom left

		#top left has smallest sum, bottom right has largest sum
		quadrangle[0]=corners[np.argmin(s)]
		quadrangle[2]=corners[np.argmax(s)]
		
		difference=np.diff(corners,axis=1)
		#top right have minimum difference, bottom left have maximum difference
		quadrangle[1]=corners[np.argmin(difference)]
		quadrangle[3]=corners[np.argmax(difference)]

		return quadrangle

	def computeMaxWidthAndHeightOfSudokuPuzzle(self,quadrangle):
		topLeft,topRight,bottomRight,bottomLeft=quadrangle
		#recall that to calculate distance between (x1,y1) and (x2,y2)
		#we need to do sqrt( (x2-x1)^2 + (y2-y1)^2 )
		upperWidth=np.sqrt( ((topRight[0]-topLeft[0])**2) + ((topRight[1]-topLeft[1])**2) )
		bottomWidth=np.sqrt( ((bottomRight[0]-bottomLeft[0])**2) + ((bottomRight[1]-bottomLeft[1])**2) )

		leftHeight=np.sqrt( ((topLeft[0]-bottomLeft[0])**2) + ((topLeft[1]-bottomLeft[1])**2) )
		rightHeight=np.sqrt( ((topRight[0]-bottomRight[0])**2) + ((topRight[1]-bottomRight[1])**2) )

		maximumWidth=max(int(upperWidth),int(bottomWidth))
		maximumHeight=max(int(leftHeight),int(rightHeight))

		return maximumWidth,maximumHeight


	"""
	Extract out the grid and apply transformation to obtain top down perspective (bird eye view) on the sudoku puzzle. 
	Reference to https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html (OpenCV tutorial)
	"""
	def extractSudokuPuzzleAndWarpPerspective(self,quadrangle,maximumWidth,maximumHeight,originalSudokuImage):
		originalSudokuImage=deepcopy(originalSudokuImage)
		
		#map the screen to a top down, bird eye view
		destinationPoints=np.array([ [0,0],[maximumWidth-1,0],[maximumWidth-1,maximumHeight-1],[0,maximumHeight-1] ],dtype="float32")
		#compute perspective transform
		M=cv2.getPerspectiveTransform(quadrangle,destinationPoints)
		#apply transformation                           #width and height of output image
		warp=cv2.warpPerspective(originalSudokuImage,M,(maximumWidth,maximumHeight))

		return warp

	def postProcessExtractedSudokuPuzzle(self,warpedSudokuPuzzle,postProcess=True):
		if(postProcess):
			#convert warped image to grayscale
			postProcessed= cv2.cvtColor(warpedSudokuPuzzle, cv2.COLOR_BGR2GRAY)
			#adjust intensity of pixels to have min and max value of 0 and 255
			postProcessed=exposure.rescale_intensity(postProcessed,out_range=(0,255))
			postProcessed = cv2.resize(postProcessed,(450, 450),interpolation=cv2.INTER_AREA)
			self.display.displayImage(postProcessed)
		else:
			postProcessed = cv2.resize(warpedSudokuPuzzle,(450, 450),interpolation=cv2.INTER_AREA)
		return postProcessed