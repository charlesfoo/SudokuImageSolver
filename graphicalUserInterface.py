#!/usr/bin/env python3

"""
@author: Foo Zhi Yuan
Sudoku Image Solver uses image processing techniques to extract sudoku puzzle and Convolutional Neural Network for digit parsing, then solve parsed sudoku puzzle
using Linear Programming.
Requires Python 3, OpenCV (for image processing), TensorFlow (for building ConvolutionalNN), NumPy, PuLP (for Linear Programming) and Pillow(for GUI)
USAGE: python3 sudokuImageSolver.py to launch in GUI and python3 sudokuImageSolver_console.py to launch in console
"""

import cv2
import os
import numpy as np
from settings import *
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk
from Sudoku.resizeSudokuImage import ResizeSudokuImage
from Extractor.extractSudokuPuzzle import ExtractSudokuPuzzle
from Extractor.extractSudokuCells import ExtractSudokuCells
from ConvolutionalNN.dataset import *
from ConvolutionalNN.convolutionalNN import ConvolutionalNN
from ConvolutionalNN.digitRecognition import DigitRecognition
from Solver.linearProgrammingSolver import LinearProgrammingSolver

#current state of sudoku image solver
DISPLAY_ORIGINAL_SUDOKU_IMAGE=0
PREPROCESS_SUDOKU_IMAGE=1
EXTRACT_SUDOKU_PUZZLE=2
FLATTEN_SUDOKU_PUZZLE=3
EXTRACT_SUDOKU_CELLS=4
PARSE_SUDOKU_PUZZLE=5
SOLVE_SUDOKU_PUZZLE=6
RESCAN_SUDOKU_PUZZLE=7
RESOLVE_SUDOKU_PUZZLE=8


def trainConvolutionalNNForDigitRecognition():
	trainingSet=Dataset(trainingDirectory=True)
	testSet=Dataset(trainingDirectory=False)
	#generally, setting number of iteration as 20 epoch seems to train model well
	numIteration=(len(trainingSet)*20)//50

	convolutionalNN=ConvolutionalNN()
	convolutionalNN.train(trainingSet,steps=numIteration,batchSize=50)
	convolutionalNN.test(testSet)


class GraphicalUserInterface():

	def __init__(self):
		self.master=tk.Tk()
		self.master.title("Sudoku Image Solver")

		self.squareTileSize=100

		#initialize all the rendering images (sudoku logo and digit images)
		self.digitImages=[]
		for i in range(1,10):
			currentPath=os.path.join("renderingImage",str(i)+".png")
			self.digitImages.append(ImageTk.PhotoImage(file=currentPath,width=self.squareTileSize,height=self.squareTileSize))
		currentPath=os.path.join("renderingImage","sudokuLogo.jpg")
		self.sudokuLogo=ImageTk.PhotoImage(file=currentPath,width=100,height=100)


		self.selectedFile=None
		self.filePath=None

		self.master.withdraw()
		self.initializeVariables()
		self.buildHome()
		self.master.mainloop()

	def initializeVariables(self):
		#size of home page
		self.canvasWidth=900
		self.canvasHeight=900
		self.sudokuImagePath=""

		self.bottomFrameElements=[]

		if(self.selectedFile is not None):
			self.selectedFile.config(text="")
		if(self.filePath is not None):
			self.filePath.config(text="")


	def buildHome(self):
		self.master.deiconify()

		#home page consists of top frame and bottom frame
		## top frame contains sudoku logo, file selection frame, selected file frame and start button
		self.homeTopFrame=tk.Frame(self.master,bg="medium sea green")
		self.homeTopFrame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
		
		#sudoku logo
		logo=tk.Label(self.homeTopFrame,image=self.sudokuLogo)
		logo.pack(side=tk.TOP,anchor=tk.CENTER,padx=50,pady=70)


		#file selection frame contains label for file selection and file selection button, all centered horizontally in the same row
		fileSelectionFrame=tk.Frame(self.homeTopFrame,bg="medium sea green")
		fileSelectionFrame.pack(side=tk.TOP,fill='x')

		prompt=tk.Label(fileSelectionFrame,text="Select a sudoku image",fg="black",bg="medium sea green",font=("Verdana",18,"bold"))
		prompt.grid(row=0,column=1,padx=5,pady=10)
		browseFile_button=tk.Button(fileSelectionFrame,text="Browse",fg="black",bg="MediumOrchid1",font=("Verdana",18,"bold"),command=self.browseFile)
		browseFile_button.grid(row=0,column=2,padx=5,pady=10)

		fileSelectionFrame.columnconfigure(0,weight=1)
		fileSelectionFrame.columnconfigure(3,weight=1)

		#selected file label contains the text label for selected file.
		selectedFileFrame=tk.Frame(self.homeTopFrame,bg="medium sea green")
		selectedFileFrame.pack(side=tk.TOP,fill='x')

		self.selectedFile=tk.Label(selectedFileFrame,fg="black",bg="medium sea green",font=("Verdana",9,"bold","italic"))
		self.selectedFile.grid(row=1,column=1,padx=5,pady=10)
		self.filePath=tk.Label(selectedFileFrame,fg="black",bg="medium sea green",font=("Verdana",9,"italic"))
		self.filePath.grid(row=1,column=2,padx=5,pady=10)

		selectedFileFrame.columnconfigure(0,weight=1)
		selectedFileFrame.columnconfigure(3,weight=1)

		#start button
		start_button=tk.Button(self.homeTopFrame,text="Start",fg="black",bg="red3",font=("Verdana",18,"bold"),command=self.startSudokuImageSolver)
		start_button.pack(side=tk.TOP,anchor=tk.CENTER,padx=70,pady=20)

		## bottom frame contains program name and watermark
		self.homeBottomFrame=tk.Frame(self.master)
		self.homeBottomFrame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
		programName=tk.Label(self.homeBottomFrame,text="Sudoku Image Solver",fg="black",font=("Times",50,"bold"))
		programName.pack(side=tk.TOP,anchor=tk.CENTER,padx=100,pady=10)

		watermark=tk.Label(self.homeBottomFrame,text="A product of Foo Zhi Yuan",fg="black",font=("Verdana",13,"bold","italic"))
		watermark.pack(side=tk.BOTTOM,anchor=tk.S,pady=15)

		self.master.update_idletasks()

	def hideHome(self):
		self.homeTopFrame.pack_forget()
		self.homeBottomFrame.pack_forget()

	def showHome(self):
		self.homeTopFrame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
		self.homeBottomFrame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

	def browseFile(self):
		self.sudokuImagePath=tk.filedialog.askopenfilename( parent=self.homeTopFrame,initialdir=os.getcwd(),filetypes = (("jpg files","*.jpg"),("JPG files","*.JPG"),("png files","*.png"),
			("PNG files","*.PNG"),("All files","*.*")), title="Select a sudoku image" )
		
		if(self.sudokuImagePath!=()):
			self.selectedFile.config(text="Selected File:")
			self.filePath.config(text=self.sudokuImagePath)
		else:
			self.selectedFile.config(text="")
			self.filePath.config(text="")

		return self.sudokuImagePath


	def startSudokuImageSolver(self):
		if(os.path.isfile(self.sudokuImagePath)):
			self.hideHome()
			self.buildCanvas()
		else:
			# raise FileNotFoundError("Error: File selected does not exist.")
			print("Error: File selected does not exist.")
	
	def returnHome(self,popup=None):
		if(popup is not None):
			popup.destroy()

		self.canvas.delete("canvasImage")
		self.canvas.delete("squareTile")
		self.canvas.delete("line")
		self.canvas.delete("sudokuDigits")
		for item in self.bottomFrameElements:
			item.destroy()
		self.topFrame.destroy()
		self.bottomFrame.destroy()
		self.initializeVariables()
		self.showHome()



	#this method computes the (x,y) coordinate in which the image should be displayed on canvas
	def computeCoordinateOfImage(self,image):
		height,width=image.shape[:2]
		x=(self.canvasWidth-width)//2
		y=(self.canvasHeight-height)//2
		return x,y

	def getCurrentParsedImage(self):
		if(len(self.display.images)==0):
			raise IndexError("Error in getCurrentParsedImage: popping from an empty list")
		return self.display.images.pop(0)


	def displayCurrentImageToCanvas(self):
		x,y=self.computeCoordinateOfImage(self.currentImage)
		self.currentImage=ImageTk.PhotoImage(Image.fromarray(self.currentImage))
		self.canvas.create_image(x,y,anchor=tk.NW,image=self.currentImage,tags="canvasImage")

	def addNextButtonToBottomFrame(self,currentState):
		if(currentState==DISPLAY_ORIGINAL_SUDOKU_IMAGE):
			text="Preprocess Sudoku Image"
			nextState=PREPROCESS_SUDOKU_IMAGE
		elif(currentState==PREPROCESS_SUDOKU_IMAGE):
			text="Extract Sudoku Puzzle"
			nextState=EXTRACT_SUDOKU_PUZZLE
		elif(currentState==EXTRACT_SUDOKU_PUZZLE):
			text="Flatten Sudoku Puzzle"
			nextState=FLATTEN_SUDOKU_PUZZLE
		elif(currentState==FLATTEN_SUDOKU_PUZZLE):
			text="Extract Sudoku Cells"
			nextState=EXTRACT_SUDOKU_CELLS
		elif(currentState==EXTRACT_SUDOKU_CELLS):
			text="Parse Sudoku Puzzle"
			nextState=PARSE_SUDOKU_PUZZLE
		elif(currentState==PARSE_SUDOKU_PUZZLE):
			text="Solve"
			nextState=SOLVE_SUDOKU_PUZZLE
		elif(currentState==RESCAN_SUDOKU_PUZZLE):
			text="Solve"
			nextState=RESOLVE_SUDOKU_PUZZLE
		else:
			raise ValueError("Error in addNextButtonToBottomFrame: currentState invalid.")

		button=tk.Button(self.bottomFrame,text=text,fg="black",bg="RoyalBlue1",font=("Verdana",18,"bold"),command=lambda: self.switchState(nextState))
		button.pack(side=tk.TOP,anchor=tk.CENTER,padx=70,pady=20)
		self.bottomFrameElements.append(button)

	def addWatermarkToBottomFrame(self):
		watermark=tk.Label(self.bottomFrame,text="A product of Foo Zhi Yuan",fg="black",font=("Verdana",13,"bold","italic"))
		watermark.pack(side=tk.BOTTOM,anchor=tk.S,pady=15)
		self.bottomFrameElements.append(watermark)



	def displayParsedSudokuPuzzleToCanvas(self,sudokuPuzzle,emptyEntries):
		self.drawTiles(emptyEntries)
		self.drawDigits(sudokuPuzzle)

	def drawTiles(self,emptyEntries):
		self.canvas.delete("squareTile")
		self.canvas.delete("line")

		for i in range(9):
			for j in range(9):
				#top left corner coordinate of square tile
				topLeft_x=j*self.squareTileSize
				topLeft_y=i*self.squareTileSize
				#bottom right corner coordinate of square tile
				bottomRight_x=topLeft_x+self.squareTileSize
				bottomRight_y=topLeft_y+self.squareTileSize
				if( (i,j) in emptyEntries ):
					self.canvas.create_rectangle(topLeft_x,topLeft_y,bottomRight_x,bottomRight_y,fill="spring green",tags="squareTile")
				else:
					self.canvas.create_rectangle(topLeft_x,topLeft_y,bottomRight_x,bottomRight_y,fill="white",tags="squareTile")

		#draw vertical outline
		for i in range(10):
			if(i%3==0):
				currentColour="black"
				width=10
			else:
				currentColour="gray"
				width=1
			start_x=i*self.squareTileSize
			start_y=0
			end_x=i*self.squareTileSize
			end_y=self.squareTileSize*9
			self.canvas.create_line(start_x,start_y,end_x,end_y,fill=currentColour,width=width,tags="line")

		#draw horizontal line
		for i in range(10):
			if(i%3==0):
				currentColour="black"
				width=10
			else:
				currentColour="gray"
				width=1
			start_x=0
			start_y=i*self.squareTileSize
			end_x=self.squareTileSize*9
			end_y=i*self.squareTileSize
			self.canvas.create_line(start_x,start_y,end_x,end_y,fill=currentColour,width=width,tags="line")



	def drawDigits(self,sudokuPuzzle):
		self.canvas.delete("sudokuDigits")
		for i in range(9):
			for j in range(9):
				currentDigit=sudokuPuzzle[i][j]
				if(currentDigit is not None):
					#place new piece at the center of coordinate
					x=j*self.squareTileSize+self.squareTileSize//2
					y=i*self.squareTileSize+self.squareTileSize//2
					self.canvas.create_image(x,y,image=self.digitImages[currentDigit-1],anchor="c",tags="sudokuDigits")


	#display image to canvas and build buttons frame according to current state
	def switchState(self,currentState):
		for item in self.bottomFrameElements:
			item.destroy()
		self.bottomFrameElements=[]
		self.canvas.delete("canvasImage")

		if(currentState==DISPLAY_ORIGINAL_SUDOKU_IMAGE):
			#1. Initialize display
			self.display=Helper.Display(graphicalUserInterface=True)
			#2. open sudoku image and resize it
			self.resized=ResizeSudokuImage(self.sudokuImagePath,MAXIMUM_WIDTH,MAXIMUM_HEIGHT,self.display)
			
			#display the original sudoku image
			self.currentImage=self.getCurrentParsedImage()
			self.displayCurrentImageToCanvas()	
			
			#add next button and watermark to bottomFrame
			self.addNextButtonToBottomFrame(currentState)
			self.addWatermarkToBottomFrame()

		elif(currentState==PREPROCESS_SUDOKU_IMAGE):
			#3. extract sudoku puzzle from sudoku image
			self.extractedSudokuPuzzle=ExtractSudokuPuzzle(self.resized.sudokuImage,self.display)

			#display preprocessed sudoku puzzle
			self.currentImage=self.getCurrentParsedImage()
			self.displayCurrentImageToCanvas()	

			#add next button and watermark to bottomFrame
			self.addNextButtonToBottomFrame(currentState)
			self.addWatermarkToBottomFrame()
		elif(currentState==EXTRACT_SUDOKU_PUZZLE):
			#display extracted sudoku puzzle
			self.currentImage=self.getCurrentParsedImage()
			self.displayCurrentImageToCanvas()

			#add next button and watermark to bottomFrame
			self.addNextButtonToBottomFrame(currentState)
			self.addWatermarkToBottomFrame()
		elif(currentState==FLATTEN_SUDOKU_PUZZLE):
			#display flattened sudoku puzzle (flatten using warp perspective)
			self.currentImage=self.getCurrentParsedImage()
			self.displayCurrentImageToCanvas()

			#add next button and watermark to bottomFrame
			self.addNextButtonToBottomFrame(currentState)
			self.addWatermarkToBottomFrame()
		elif(currentState==EXTRACT_SUDOKU_CELLS):
			#4a. extract sudoku cells from sudoku puzzle via largest feature
			self.cellsExtractor=ExtractSudokuCells(self.extractedSudokuPuzzle.preprocessedExtracted,self.extractedSudokuPuzzle.postProcessedExtracted,self.display)
			self.sudokuDigits_largestFeature=self.cellsExtractor.run(findDigitViaLargestFeature=True,displayImage=True)
			#4b. find digit in sudoku cells via largest contour and domain knowledge
			self.sudokuDigits_largestContour=self.cellsExtractor.run(findDigitViaLargestFeature=False,displayImage=False)

			#5. save sudoku digits in training directory if the createTrainingSet variable is True
			if(createTrainingSetForDigitRecognition):
				filename=self.sudokuImagePath.split("/")
				filename=filename[-1]
				Helper.save(self.sudokuDigits_largestFeature, filename, findDigitViaLargestFeature=True, trainingDirectory=True)
				Helper.save(self.sudokuDigits_largestContour, filename, findDigitViaLargestFeature=False, trainingDirectory=True)

			#display extracted sudoku cells (flatten using warp perspective)
			self.currentImage=self.getCurrentParsedImage()
			self.displayCurrentImageToCanvas()

			#add next button and watermark to bottomFrame
			self.addNextButtonToBottomFrame(currentState)
			self.addWatermarkToBottomFrame()
		elif(currentState==PARSE_SUDOKU_PUZZLE):
			#5.5 For training convolutionalNN for digit recognition
			# trainConvolutionalNNForDigitRecognition()
			#6. perform digit recognition on the preprocessed extracted sudoku cells
			self.digitRecognition=DigitRecognition()
			sudokuPuzzle=self.digitRecognition.predict(self.sudokuDigits_largestFeature)
			#7. Solve sudoku puzzle using linear programming
			self.linearProgrammingSolver=LinearProgrammingSolver(sudokuPuzzle)
			self.displayParsedSudokuPuzzleToCanvas(self.linearProgrammingSolver.sudokuPuzzle,self.linearProgrammingSolver.emptyEntries)

			#add next button and watermark to bottomFrame
			self.addNextButtonToBottomFrame(currentState)
			self.addWatermarkToBottomFrame()
		elif(currentState==SOLVE_SUDOKU_PUZZLE):
			solved,solvedSudokuPuzzle,emptyEntries=self.linearProgrammingSolver.run(printSolvedPuzzle=False)
			if(not solved):
				self.switchState(RESCAN_SUDOKU_PUZZLE)
			self.drawDigits(solvedSudokuPuzzle)
			#add rescan button, return to home button and watermark to bottomFrame
			notice=tk.Label(self.bottomFrame,text="Sudoku puzzle being parsed wrongly? Try rescanning again (using another image processing technique).",fg="black",font=("Verdana",11,"bold","italic"))
			notice.grid(row=0,column=0,padx=5,pady=5,columnspan=5)	

			rescan_button=tk.Button(self.bottomFrame,text="Rescan",fg="black",bg="RoyalBlue1",font=("Verdana",18,"bold"),command=lambda: self.switchState(RESCAN_SUDOKU_PUZZLE))
			rescan_button.grid(row=1,column=1,padx=20,pady=5)

			returnHome_button=tk.Button(self.bottomFrame,text="Return Home",fg="black",bg="red3",font=("Verdana",18,"bold"),command=self.returnHome)
			returnHome_button.grid(row=1,column=3,padx=20,pady=5)
			self.bottomFrame.columnconfigure(0,weight=1)
			self.bottomFrame.columnconfigure(2,weight=1)
			self.bottomFrame.columnconfigure(4,weight=1)

			watermark=tk.Label(self.bottomFrame,text="A product of Foo Zhi Yuan",fg="black",font=("Verdana",13,"bold","italic"))
			watermark.grid(row=2,pady=5,columnspan=5)

			self.bottomFrameElements.append(notice)
			self.bottomFrameElements.append(rescan_button)
			self.bottomFrameElements.append(returnHome_button)
			self.bottomFrameElements.append(watermark)
		elif(currentState==RESCAN_SUDOKU_PUZZLE):
			#7.5 If sudoku puzzle can't be solved, use sudoku cells that is extracted via largest contour
			sudokuPuzzle=self.digitRecognition.predict(self.sudokuDigits_largestContour)
			self.linearProgrammingSolver=LinearProgrammingSolver(sudokuPuzzle)
			self.displayParsedSudokuPuzzleToCanvas(self.linearProgrammingSolver.sudokuPuzzle,self.linearProgrammingSolver.emptyEntries)

			#add next button and watermark to bottomFrame
			self.addNextButtonToBottomFrame(currentState)
			self.addWatermarkToBottomFrame()
		elif(currentState==RESOLVE_SUDOKU_PUZZLE):
			solved,solvedSudokuPuzzle,emptyEntries=self.linearProgrammingSolver.run(printSolvedPuzzle=False)
			if(not solved):
				self.popup_error()
				return
			self.drawDigits(solvedSudokuPuzzle)

			returnHome_button=tk.Button(self.bottomFrame,text="Return Home",fg="black",bg="red3",font=("Verdana",18,"bold"),command=self.returnHome)
			returnHome_button.pack(side=tk.TOP,anchor=tk.CENTER,padx=70,pady=20)
			self.bottomFrameElements.append(returnHome_button)

			self.addWatermarkToBottomFrame()




	#create and display a pop up if sudoku puzzle can't be solved
	def popup_error(self):
		popup=tk.Toplevel()
		popup.title("ChessAI")

		frame1=tk.Frame(popup)
		frame1.pack()
		error_message=tk.Label(frame1,text="Unable to solve puzzle. There might be error in the parsed sudoku puzzle.",fg="black",font=("Times",18))
		error_message.pack(padx=30,pady=15)

		#create a one beautiful line separator to separate between question and the 2 buttons
		separator=tk.Frame(frame1,height=2,bd=1,relief=tk.SUNKEN)
		separator.pack(fill=tk.X,padx=5,pady=5)

		frame2=tk.Frame(popup)
		frame2.pack(side=tk.BOTTOM)

		returnHome_button=tk.Button(frame1,text="Return Home",fg="black",bg="red3",font=("Verdana",18,"bold"),command=lambda: self.returnHome(popup))
		returnHome_button.pack(side=tk.TOP,anchor=tk.CENTER,padx=15,pady=10)

		watermark=tk.Label(frame2,text="A product of Foo Zhi Yuan",fg="black",font=("Verdana",10,"bold","italic"))
		watermark.pack(side=tk.BOTTOM,pady=10)

		self.master.update_idletasks()


	def buildCanvas(self):
		self.topFrame=tk.Frame(self.master)
		self.topFrame.pack(side=tk.TOP,expand=True,fill=tk.BOTH)
		self.canvas=tk.Canvas(self.topFrame,width=self.canvasWidth,height=self.canvasHeight,background="gray27")
		self.canvas.pack(side=tk.TOP,fill=tk.BOTH,anchor=tk.CENTER,expand=True)

		#draw parsed sudoku image to canvas
		self.bottomFrame=tk.Frame(self.master)
		self.bottomFrame.pack(side=tk.TOP,fill='x')
		
		self.switchState(DISPLAY_ORIGINAL_SUDOKU_IMAGE)





