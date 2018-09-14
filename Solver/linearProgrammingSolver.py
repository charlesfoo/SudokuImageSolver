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
import pulp
import os


class LinearProgrammingSolver:
	sudokuPuzzle=None
	emptyEntries=None


	def __init__(self,sudokuPuzzle):
		self.sudokuPuzzle=Helper.convert1DSudokuPuzzleTo2D(sudokuPuzzle)
		self.emptyEntries=Helper.findEmptyEntries(self.sudokuPuzzle)

		"""
		Sudoku Boxes

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
		"""

		#stores all the coordinates for each box
		#for example, for the first box, the coordinates are
		#(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1)(3,2),(3,3)
		self.box=[]
		for i in range(3):
			for j in range(3):
										#row 	  #column
				self.box.extend([[ ((i*3)+k+1, (j*3)+l+1) for k in range(3) for l in range(3) ]])

		#create 729 variables for sudoku[val][row][col]
		#sudoku[3][2][9]=1 means value 3 is present at row 2 column 9
		self.values=self.rows=self.columns=[i for i in range(1,10)]



	def run(self,printSolvedPuzzle):
		self.initializeVariables()
		self.initializeConstraints()
		self.initializeConstraintsBasedOnSudokuPuzzle(self.sudokuPuzzle)

		#write problem data to .lp file
		self.writeProblemDataToLP_File()

		#The problem is solved using PuLP's choice of solver
		self.problem.solve()

		#there are 5 status code, eg: Not Solved, Optimal, Infeasible, Unbounded and Undefined
		if(pulp.LpStatus[self.problem.status]!="Optimal"):
			return False,self.sudokuPuzzle,self.emptyEntries

		result=self.convertSolutionTo2DArray()

		if(printSolvedPuzzle):
			Helper.printSudokuPuzzle(result,self.emptyEntries)

		self.writeSolutionToFile(result)

		return True, result, self.emptyEntries



	def initializeVariables(self):
		#problem variable is created to contain the problem data
		self.problem=pulp.LpProblem("Sudoku Problem",pulp.LpMinimize)
		#define the variable to only take binary value, eg: 0 or 1
		self.sudoku=pulp.LpVariable.dicts("Sudoku",(self.values,self.rows,self.columns),0,1,pulp.LpInteger) 
		

	def initializeConstraints(self):
		#initialize objective function to be 0, since we don't care about the optimal solution.
		#we just want a solution that satisfies all the constraint
		self.problem+= 0,"Objective Function"

		#Add constraint to ensure that given a row and column, there can only be one value in it
		for r in self.rows:
			for c in self.columns:
				self.problem+= pulp.lpSum(self.sudoku[v][r][c] for v in self.values)==1


		for v in self.values:
			for r in self.rows:
				#given a fixed value and row, there can only be one digit across all the column for that row
				#
				#   ---------->
				#  ____________________________________
				# |   |   |   |   |   |   |   |   |   |  
				# |___|___|___|___|___|___|___|___|___|
				# 
				self.problem+= pulp.lpSum([self.sudoku[v][r][c] for c in self.columns])==1

			for c in self.columns:
				#given a fixed value and column, there can only be one digit across all the rows
				#	___
				#  |   |
				#  |___|
				#  |   |
				#  |___|  |
				#  |   |  |
				#  |___|  |
				#  |   |  ↓
				#  |___|
				#  |   |
				#  |___|
				#  
				self.problem+= pulp.lpSum([self.sudoku[v][r][c] for r in self.rows])==1

			#coordinates contains a list of all the coordinates in a single box
			for coordinates in self.box:
				#given a fixed value, it can only exists once across all the coordinates in the box
				self.problem+= pulp.lpSum([self.sudoku[v][r][c] for(r,c) in coordinates])==1

	#initial values in sudoku puzzle are being added as constraint
	def initializeConstraintsBasedOnSudokuPuzzle(self,sudokuPuzzle):
		for i in range(9):
			for j in range(9):
				value=sudokuPuzzle[i][j]
				if(value is not None):
					self.problem+= self.sudoku[value][i+1][j+1]==1


	def writeProblemDataToLP_File(self,filename="Sudoku.lp"):
		path=os.path.join("Solver",filename)
		self.problem.writeLP(path)


	def convertSolutionTo2DArray(self):
		result=[]
		for r in self.rows:
			currentRow=[]

			for c in self.columns:
				for v in self.values:
					if(pulp.value(self.sudoku[v][r][c])==1):
						currentRow.append(v)
			result.append(currentRow)
		return result


	def writeSolutionToFile(self,result,filename="sudoku_out.txt"):
		# A file called sudoku_out.txt is created/overwritten for writing to
		writePath=os.path.join("Solver",filename)

		with open(writePath,'w') as file:
			for i in range(9):
				for j in range(9):
					file.write(str(result[i][j])+" ")
				file.write("\n")

		print("Solution written to "+writePath)



