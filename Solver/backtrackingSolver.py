#!/usr/bin/env python3

"""
@author: Foo Zhi Yuan
Sudoku Image Solver uses image processing techniques to extract sudoku puzzle and Convolutional Neural Network for digit parsing, then solve parsed sudoku puzzle
using Linear Programming.
Requires Python 3, OpenCV (for image processing), TensorFlow (for building ConvolutionalNN), NumPy, PuLP (for Linear Programming) and Pillow(for GUI)
USAGE: python3 sudokuImageSolver.py to launch in GUI and python3 sudokuImageSolver_console.py to launch in console
"""

from copy import deepcopy
import Helper.helper as Helper

#Idea: Given a sudoku puzzle in array of array form, I want to be able to solve the sudoku puzzle via backtracking.

#Given a sudoku puzzle, mapped to an arr of 9x9, solver() should return an arr that solves the puzzle

"""
Brute force backtracking Algorithm
1. Find an empty entry (row, column), if none, return true
2. for i=1..9
	If it's legal to assign i to (row,column),assign it and recursively call solver to solve for the remaining entries.
	If recursion succeeds, return true
	If recursion fails, try with another number at (row,column)
If all current legal integers have been tried and all fails, return False. (Backtracking! This allows us to go to a level before and assign with another number)
"""

class BacktrackingSolver:
	sudokuPuzzle=None
	emptyEntries=None


	def __init__(self,sudokuPuzzle):
		self.sudokuPuzzle=Helper.convert1DSudokuPuzzleTo2D(sudokuPuzzle)
		self.emptyEntries=Helper.findEmptyEntries(self.sudokuPuzzle)


	def run(self,printSolvedPuzzle):
		solvedSudokuPuzzle=deepcopy(self.sudokuPuzzle)
		solved=self.solver(solvedSudokuPuzzle)

		if(not solved):
			return False,self.sudokuPuzzle,self.emptyEntries

		if(printSolvedPuzzle):
			Helper.printSudokuPuzzle(solvedSudokuPuzzle,self.emptyEntries)

		return True, solvedSudokuPuzzle, self.emptyEntries


	def searchForEmptyEntry(self,arr):
		for i in range(9):
			for j in range(9):
				if(arr[i][j]==None):
					return i,j

		return None,None

	def checkIfNumIsValidInRow(self,arr,row,num):
		if(num<1 or num>9):
			return False
		for j in range(9):
			if(arr[row][j]==num):
				return False
		return True

	def checkIfNumIsValidInColumn(self,arr,column,num):
		if(num<1 or num>9):
			return False
		for i in range(9):
			if(arr[i][column]==num):
				return False
		return True

	def checkIfNumIsValidInBox(self,arr,row,column,num):
		"""
		    0   1   2    3   4   5    6   7   8
		   ___________  ___________  ___________
		0 |___|___|___||___|___|___||___|___|___|
		1 |___|___|___||___|___|___||___|___|___|		  
		2 |___|___|___||___|___|___||___|___|___|
		   ___________  ___________  ___________
		3 |___|___|___||___|___|___||___|___|___|
		4 |___|___|___||___|___|___||___|___|___|		  
		5 |___|___|___||___|___|___||___|___|___|
		   ___________  ___________  ___________
		6 |___|___|___||___|___|___||___|___|___|
		7 |___|___|___||___|___|___||___|___|___|		  
		8 |___|___|___||___|___|___||___|___|___|

		if (row,column) is (4,5), we want to check the box with left corner of (3,3)
		This can be done by normalising (row,column) to the box margin, which is 3 (eg: row,column needs to ALWAYS be multiple of 3)
		In order to achieve this, we first compute row%3, then use row-row%3. This will always yield row that is a multiple of 3.
		The same goes with column
		"""
		if(num<1 or num>9):
			return False

		row=row-(row%3)
		column=column-(column%3)

		for i in range(3):
			for j in range(3):
				if(arr[row+i][column+j]==num):
					return False
		return True


	def checkIfNumIsValidToAssign(self,arr,row,column,num):
		result=self.checkIfNumIsValidInRow(arr,row,num) and self.checkIfNumIsValidInColumn(arr,column,num) and self.checkIfNumIsValidInBox(arr,row,column,num)
		return result

	def solver(self,arr):
		row,column=self.searchForEmptyEntry(arr)
		if(row==None and column==None):
			return True

		for i in range(1,10):
			if(self.checkIfNumIsValidToAssign(arr,row,column,i)==True):
				arr[row][column]=i
				if(self.solver(arr)==True):
					return True
				arr[row][column]=None

		#if at this level, all of the valid numbers cannot solve the sudoku puzzle, we need to backtrack one level and change with another number
		return False