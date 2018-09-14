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
from settings import *
import Helper.helper as Helper
from Sudoku.resizeSudokuImage import ResizeSudokuImage
from Extractor.extractSudokuPuzzle import ExtractSudokuPuzzle
from Extractor.extractSudokuCells import ExtractSudokuCells
from ConvolutionalNN.dataset import *
from ConvolutionalNN.convolutionalNN import ConvolutionalNN
from ConvolutionalNN.digitRecognition import DigitRecognition
from Solver.linearProgrammingSolver import LinearProgrammingSolver
import os

class SudokuImageSolver_Console:

	def trainConvolutionalNNForDigitRecognition(self):
		trainingSet=Dataset(trainingDirectory=True)
		testSet=Dataset(trainingDirectory=False)
		#generally, setting number of iteration as 20 epoch seems to train model well
		numIteration=(len(trainingSet)*20)//50

		convolutionalNN=ConvolutionalNN()
		convolutionalNN.train(trainingSet,steps=numIteration,batchSize=50)
		convolutionalNN.test(testSet)

	def main(self):
		display=Helper.Display()

		#resize sudoku image
		resized=ResizeSudokuImage(sudokuImagePath,MAXIMUM_WIDTH,MAXIMUM_HEIGHT,display)

		#extract sudoku puzzle from sudoku image
		extractedSudokuPuzzle=ExtractSudokuPuzzle(resized.sudokuImage,display)

		#extract sudoku cells from sudoku puzzle via largest feature
		cellsExtractor=ExtractSudokuCells(extractedSudokuPuzzle.preprocessedExtracted,extractedSudokuPuzzle.postProcessedExtracted,display)
		sudokuDigits_largestFeature=cellsExtractor.run(findDigitViaLargestFeature=True,displayImage=True)

		#find digit in sudoku cells via largest contour and domain knowledge
		sudokuDigits_largestContour=cellsExtractor.run(findDigitViaLargestFeature=False,displayImage=False)

		#save sudoku digits in training directory if the createTrainingSet variable is True
		if(createTrainingSetForDigitRecognition):
			Helper.save(sudokuDigits_largestFeature, filename, findDigitViaLargestFeature=True, trainingDirectory=True)
			Helper.save(sudokuDigits_largestContour, filename, findDigitViaLargestFeature=False, trainingDirectory=True)

		#For training convolutionalNN for digit recognition
		if(trainConvolutionalNeuralNetwork):
			trainConvolutionalNNForDigitRecognition()

		digitRecognition=DigitRecognition()
		sudokuPuzzle=digitRecognition.predict(sudokuDigits_largestFeature)
		
		linearProgrammingSolver=LinearProgrammingSolver(sudokuPuzzle)
		solved,solvedSudokuPuzzle,emptyEntries=linearProgrammingSolver.run(printSolvedPuzzle=True)
		
		if(not solved):
			sudokuPuzzle=digitRecognition.predict(sudokuDigits_largestContour)
			linearProgrammingSolver=LinearProgrammingSolver(sudokuPuzzle)
			solved,solvedSudokuPuzzle,emptyEntries=linearProgrammingSolver.run(printSolvedPuzzle=True)
			if(not solved):
				print("Error: Unable to solve sudoku puzzle.")

		if(CONSOLE_DISPLAY_IMAGE):
			cv2.destroyAllWindows()


sudokuImageSolver_console=SudokuImageSolver_Console()
sudokuImageSolver_console.main()