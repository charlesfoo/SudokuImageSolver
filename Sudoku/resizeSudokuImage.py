#!/usr/bin/env python3

"""
@author: Foo Zhi Yuan
Sudoku Image Solver uses image processing techniques to extract sudoku puzzle and Convolutional Neural Network for digit parsing, then solve parsed sudoku puzzle
using Linear Programming.
Requires Python 3, OpenCV (for image processing), TensorFlow (for building ConvolutionalNN), NumPy, PuLP (for Linear Programming) and Pillow(for GUI)
USAGE: python3 sudokuImageSolver.py to launch in GUI and python3 sudokuImageSolver_console.py to launch in console
"""

import cv2
import Helper.helper as Helper

class ResizeSudokuImage:
	sudokuImage=None
	imageHeight=None
	imageWidth=None

	def __init__(self,filename,MAXIMUM_WIDTH,MAXIMUM_HEIGHT,display):
		self.sudokuImage=self.loadSudokuImage(filename)

		self.imageHeight, self.imageWidth = self.sudokuImage.shape[:2]
		
		if(self.imageWidth>MAXIMUM_WIDTH):
			self.sudokuImage=self.resizeImage(self.sudokuImage,intendedWidth=MAXIMUM_WIDTH)

		self.imageHeight, self.imageWidth = self.sudokuImage.shape[:2]
		if(self.imageHeight>MAXIMUM_HEIGHT):
			self.sudokuImage=self.resizeImage(self.sudokuImage,intendedHeight=MAXIMUM_HEIGHT)

		display.displayImage(self.sudokuImage)


	def loadSudokuImage(self,filename):
		#read in image in grayscale
		sudokuImage=cv2.imread(filename,cv2.IMREAD_COLOR)
		# sudokuImage=cv2.resize(sudokuImage,(900,900),interpolation=cv2.INTER_AREA)
		if(sudokuImage is None):
			print("Error: Sudoku image cannot be read. Please check the filename and ensure that the image is in the correct path before trying again.")
			exit()

		return sudokuImage

	"""
	Resize image without distortion
	Reference to https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv

	"""
	def resizeImage(self,sudokuImage,intendedWidth=None,intendedHeight=None,inter=cv2.INTER_AREA):
		dimension=None

		if(intendedHeight is None and intendedWidth is None):
			return sudokuImage

		if(intendedWidth is None):
			ratio=intendedHeight/float(self.imageHeight)
			dimension=(int(self.imageWidth*ratio),intendedHeight)
		elif(intendedHeight is None):
			ratio=intendedWidth/float(self.imageWidth)
			dimension=(intendedWidth,int(self.imageHeight*ratio))

		resized=cv2.resize(sudokuImage,dimension,interpolation=inter)

		return resized