#!/usr/bin/env python3

"""
@author: Foo Zhi Yuan
Sudoku Image Solver uses image processing techniques to extract sudoku puzzle and Convolutional Neural Network for digit parsing, then solve parsed sudoku puzzle
using Linear Programming.
Requires Python 3, OpenCV (for image processing), TensorFlow (for building ConvolutionalNN), NumPy, PuLP (for Linear Programming) and Pillow(for GUI)
USAGE: python3 sudokuImageSolver.py to launch in GUI and python3 sudokuImageSolver_console.py to launch in console
"""

import os
from random import randint, shuffle

############################# INPUT FILE
sudokuImageFolder="dataset/sudokuImage"
filename="image1.jpg"
sudokuImagePath=os.path.join(sudokuImageFolder,filename)


############################# USER SETTINGS
#Indicates whether image should be displayed when in console mode. If set to false, only the solution will be printed to console
CONSOLE_DISPLAY_IMAGE=True

MAXIMUM_HEIGHT=900
MAXIMUM_WIDTH=900

#conforms to the size of MNIST datasets
DIGIT_SIZE=28

############################# SETTINGS OF MODEL TRAINING FOR DIGIT RECOGNITION
#create training set in ${sudokuDigitFolder}/training/unsorted
createTrainingSetForDigitRecognition=False

#if createTrainingSetForDigitRecognition is true, a folder named training will be created if it doesn't exist, and the model will be trained on the digit images inside the folder
sudokuDigitFolder=os.path.join(os.getcwd(),"dataset/sudokuDigit")

#path where tensorflow model for digit recognition is stored
modelPath=os.path.join(os.getcwd(),"ConvolutionalNN/model")

# WARNING: ***DO NOT*** turn this on unless you want to train the Convolutional NN used for digit recognition.
# Please refer to README.md and ensure that ***you know what you're doing*** before turning this on
trainConvolutionalNeuralNetwork=False
#############################
