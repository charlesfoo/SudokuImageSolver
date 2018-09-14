# Sudoku Image Solver

##### *Implemented in Python 3*

#### Author: **Foo Zhi Yuan**

Sudoku Image Solver is a Python program that takes in an image, scan for Sudoku puzzle, and solve it using Linear Programming method.

Sudoku Image Solver uses image processing techniques to extract Sudoku Puzzle from an image, Convolutional Neural Network to parse digits and Linear Programming to solve parsed puzzle.

1. When an image is being input into the solver, the solver will first preprocess the input image using bilateral filter and thresholding. 

2. Sudoku puzzle is being located by finding the largest feature/ largest contour in the input image. 

3. The puzzle is then being flattened (warp perspective) to obtain a top down view of the puzzle. 

4. The flattened puzzle is being extracted and preprocessed.

5. The extracted puzzle is being chopped evenly into 81 pieces for cells extraction.

6. For each cell in the puzzle, the digit (located by finding the largest feature in the cell) is being centered and resized (28x28) to conform to the format of MNIST dataset.

7. The digits are then being passed into an 8 layer Convolutional Neural Network (built using TensorFlow) for digit recognition.

8. The parsed puzzle is being solved using Linear Programming model (implemented and built using PuLP).


### Dependencies
1. Python 3

2. OpenCV

3. TensorFlow

4. PuLP

5. NumPy

6. Pillow

*Note: (For Ubuntu/ Debian machines) If you encounter "ImportError: cannot import name ImageTk" when running the program*

  *Trying doing:*

  `sudo apt-get install python-imaging python-imaging-tk`

  *For Python 3, do:*

  `sudo apt-get install python3-pil python3-pil.imagetk`


### Run (with Tkinter GUI):
`./sudokuImageSolver.py`

#### Home
![home](https://github.com/fzy1995/SudokuImageSolver/blob/master/renderingImage/ProgramScreenshots/1_home.png)

#### Input Image 
![input_image](https://github.com/fzy1995/SudokuImageSolver/blob/master/renderingImage/ProgramScreenshots/2_inputImage.png)

#### Preprocess Image
![preprocessed](https://github.com/fzy1995/SudokuImageSolver/blob/master/renderingImage/ProgramScreenshots/3_preprocessed.png)

#### Extract Puzzle
![extract_puzzle](https://github.com/fzy1995/SudokuImageSolver/blob/master/renderingImage/ProgramScreenshots/4_extractPuzzle.png)

#### Extract Cells
![extract_cells](https://github.com/fzy1995/SudokuImageSolver/blob/master/renderingImage/ProgramScreenshots/5_extractCells.png)

#### Parse Puzzle
![parsed_puzzle](https://github.com/fzy1995/SudokuImageSolver/blob/master/renderingImage/ProgramScreenshots/6_parsedPuzzle.png)

#### Solve Puzzle
![solved_puzzle](https://github.com/fzy1995/SudokuImageSolver/blob/master/renderingImage/ProgramScreenshots/7_solvedPuzzle.png)


### Run (on console):
`./sudokuImageSolver_console.py`

![console_solver](https://github.com/fzy1995/SudokuImageSolver/blob/master/renderingImage/ProgramScreenshots/8_console_solver.png)
