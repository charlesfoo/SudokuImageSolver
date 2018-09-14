\* Sudoku Problem *\
Minimize
Objective_Function: __dummy
Subject To
_C1: Sudoku_1_1_1 + Sudoku_2_1_1 + Sudoku_3_1_1 + Sudoku_4_1_1 + Sudoku_5_1_1
 + Sudoku_6_1_1 + Sudoku_7_1_1 + Sudoku_8_1_1 + Sudoku_9_1_1 = 1
_C10: Sudoku_1_2_1 + Sudoku_2_2_1 + Sudoku_3_2_1 + Sudoku_4_2_1 + Sudoku_5_2_1
 + Sudoku_6_2_1 + Sudoku_7_2_1 + Sudoku_8_2_1 + Sudoku_9_2_1 = 1
_C100: Sudoku_1_1_1 + Sudoku_1_1_2 + Sudoku_1_1_3 + Sudoku_1_2_1
 + Sudoku_1_2_2 + Sudoku_1_2_3 + Sudoku_1_3_1 + Sudoku_1_3_2 + Sudoku_1_3_3
 = 1
_C101: Sudoku_1_1_4 + Sudoku_1_1_5 + Sudoku_1_1_6 + Sudoku_1_2_4
 + Sudoku_1_2_5 + Sudoku_1_2_6 + Sudoku_1_3_4 + Sudoku_1_3_5 + Sudoku_1_3_6
 = 1
_C102: Sudoku_1_1_7 + Sudoku_1_1_8 + Sudoku_1_1_9 + Sudoku_1_2_7
 + Sudoku_1_2_8 + Sudoku_1_2_9 + Sudoku_1_3_7 + Sudoku_1_3_8 + Sudoku_1_3_9
 = 1
_C103: Sudoku_1_4_1 + Sudoku_1_4_2 + Sudoku_1_4_3 + Sudoku_1_5_1
 + Sudoku_1_5_2 + Sudoku_1_5_3 + Sudoku_1_6_1 + Sudoku_1_6_2 + Sudoku_1_6_3
 = 1
_C104: Sudoku_1_4_4 + Sudoku_1_4_5 + Sudoku_1_4_6 + Sudoku_1_5_4
 + Sudoku_1_5_5 + Sudoku_1_5_6 + Sudoku_1_6_4 + Sudoku_1_6_5 + Sudoku_1_6_6
 = 1
_C105: Sudoku_1_4_7 + Sudoku_1_4_8 + Sudoku_1_4_9 + Sudoku_1_5_7
 + Sudoku_1_5_8 + Sudoku_1_5_9 + Sudoku_1_6_7 + Sudoku_1_6_8 + Sudoku_1_6_9
 = 1
_C106: Sudoku_1_7_1 + Sudoku_1_7_2 + Sudoku_1_7_3 + Sudoku_1_8_1
 + Sudoku_1_8_2 + Sudoku_1_8_3 + Sudoku_1_9_1 + Sudoku_1_9_2 + Sudoku_1_9_3
 = 1
_C107: Sudoku_1_7_4 + Sudoku_1_7_5 + Sudoku_1_7_6 + Sudoku_1_8_4
 + Sudoku_1_8_5 + Sudoku_1_8_6 + Sudoku_1_9_4 + Sudoku_1_9_5 + Sudoku_1_9_6
 = 1
_C108: Sudoku_1_7_7 + Sudoku_1_7_8 + Sudoku_1_7_9 + Sudoku_1_8_7
 + Sudoku_1_8_8 + Sudoku_1_8_9 + Sudoku_1_9_7 + Sudoku_1_9_8 + Sudoku_1_9_9
 = 1
_C109: Sudoku_2_1_1 + Sudoku_2_1_2 + Sudoku_2_1_3 + Sudoku_2_1_4
 + Sudoku_2_1_5 + Sudoku_2_1_6 + Sudoku_2_1_7 + Sudoku_2_1_8 + Sudoku_2_1_9
 = 1
_C11: Sudoku_1_2_2 + Sudoku_2_2_2 + Sudoku_3_2_2 + Sudoku_4_2_2 + Sudoku_5_2_2
 + Sudoku_6_2_2 + Sudoku_7_2_2 + Sudoku_8_2_2 + Sudoku_9_2_2 = 1
_C110: Sudoku_2_2_1 + Sudoku_2_2_2 + Sudoku_2_2_3 + Sudoku_2_2_4
 + Sudoku_2_2_5 + Sudoku_2_2_6 + Sudoku_2_2_7 + Sudoku_2_2_8 + Sudoku_2_2_9
 = 1
_C111: Sudoku_2_3_1 + Sudoku_2_3_2 + Sudoku_2_3_3 + Sudoku_2_3_4
 + Sudoku_2_3_5 + Sudoku_2_3_6 + Sudoku_2_3_7 + Sudoku_2_3_8 + Sudoku_2_3_9
 = 1
_C112: Sudoku_2_4_1 + Sudoku_2_4_2 + Sudoku_2_4_3 + Sudoku_2_4_4
 + Sudoku_2_4_5 + Sudoku_2_4_6 + Sudoku_2_4_7 + Sudoku_2_4_8 + Sudoku_2_4_9
 = 1
_C113: Sudoku_2_5_1 + Sudoku_2_5_2 + Sudoku_2_5_3 + Sudoku_2_5_4
 + Sudoku_2_5_5 + Sudoku_2_5_6 + Sudoku_2_5_7 + Sudoku_2_5_8 + Sudoku_2_5_9
 = 1
_C114: Sudoku_2_6_1 + Sudoku_2_6_2 + Sudoku_2_6_3 + Sudoku_2_6_4
 + Sudoku_2_6_5 + Sudoku_2_6_6 + Sudoku_2_6_7 + Sudoku_2_6_8 + Sudoku_2_6_9
 = 1
_C115: Sudoku_2_7_1 + Sudoku_2_7_2 + Sudoku_2_7_3 + Sudoku_2_7_4
 + Sudoku_2_7_5 + Sudoku_2_7_6 + Sudoku_2_7_7 + Sudoku_2_7_8 + Sudoku_2_7_9
 = 1
_C116: Sudoku_2_8_1 + Sudoku_2_8_2 + Sudoku_2_8_3 + Sudoku_2_8_4
 + Sudoku_2_8_5 + Sudoku_2_8_6 + Sudoku_2_8_7 + Sudoku_2_8_8 + Sudoku_2_8_9
 = 1
_C117: Sudoku_2_9_1 + Sudoku_2_9_2 + Sudoku_2_9_3 + Sudoku_2_9_4
 + Sudoku_2_9_5 + Sudoku_2_9_6 + Sudoku_2_9_7 + Sudoku_2_9_8 + Sudoku_2_9_9
 = 1
_C118: Sudoku_2_1_1 + Sudoku_2_2_1 + Sudoku_2_3_1 + Sudoku_2_4_1
 + Sudoku_2_5_1 + Sudoku_2_6_1 + Sudoku_2_7_1 + Sudoku_2_8_1 + Sudoku_2_9_1
 = 1
_C119: Sudoku_2_1_2 + Sudoku_2_2_2 + Sudoku_2_3_2 + Sudoku_2_4_2
 + Sudoku_2_5_2 + Sudoku_2_6_2 + Sudoku_2_7_2 + Sudoku_2_8_2 + Sudoku_2_9_2
 = 1
_C12: Sudoku_1_2_3 + Sudoku_2_2_3 + Sudoku_3_2_3 + Sudoku_4_2_3 + Sudoku_5_2_3
 + Sudoku_6_2_3 + Sudoku_7_2_3 + Sudoku_8_2_3 + Sudoku_9_2_3 = 1
_C120: Sudoku_2_1_3 + Sudoku_2_2_3 + Sudoku_2_3_3 + Sudoku_2_4_3
 + Sudoku_2_5_3 + Sudoku_2_6_3 + Sudoku_2_7_3 + Sudoku_2_8_3 + Sudoku_2_9_3
 = 1
_C121: Sudoku_2_1_4 + Sudoku_2_2_4 + Sudoku_2_3_4 + Sudoku_2_4_4
 + Sudoku_2_5_4 + Sudoku_2_6_4 + Sudoku_2_7_4 + Sudoku_2_8_4 + Sudoku_2_9_4
 = 1
_C122: Sudoku_2_1_5 + Sudoku_2_2_5 + Sudoku_2_3_5 + Sudoku_2_4_5
 + Sudoku_2_5_5 + Sudoku_2_6_5 + Sudoku_2_7_5 + Sudoku_2_8_5 + Sudoku_2_9_5
 = 1
_C123: Sudoku_2_1_6 + Sudoku_2_2_6 + Sudoku_2_3_6 + Sudoku_2_4_6
 + Sudoku_2_5_6 + Sudoku_2_6_6 + Sudoku_2_7_6 + Sudoku_2_8_6 + Sudoku_2_9_6
 = 1
_C124: Sudoku_2_1_7 + Sudoku_2_2_7 + Sudoku_2_3_7 + Sudoku_2_4_7
 + Sudoku_2_5_7 + Sudoku_2_6_7 + Sudoku_2_7_7 + Sudoku_2_8_7 + Sudoku_2_9_7
 = 1
_C125: Sudoku_2_1_8 + Sudoku_2_2_8 + Sudoku_2_3_8 + Sudoku_2_4_8
 + Sudoku_2_5_8 + Sudoku_2_6_8 + Sudoku_2_7_8 + Sudoku_2_8_8 + Sudoku_2_9_8
 = 1
_C126: Sudoku_2_1_9 + Sudoku_2_2_9 + Sudoku_2_3_9 + Sudoku_2_4_9
 + Sudoku_2_5_9 + Sudoku_2_6_9 + Sudoku_2_7_9 + Sudoku_2_8_9 + Sudoku_2_9_9
 = 1
_C127: Sudoku_2_1_1 + Sudoku_2_1_2 + Sudoku_2_1_3 + Sudoku_2_2_1
 + Sudoku_2_2_2 + Sudoku_2_2_3 + Sudoku_2_3_1 + Sudoku_2_3_2 + Sudoku_2_3_3
 = 1
_C128: Sudoku_2_1_4 + Sudoku_2_1_5 + Sudoku_2_1_6 + Sudoku_2_2_4
 + Sudoku_2_2_5 + Sudoku_2_2_6 + Sudoku_2_3_4 + Sudoku_2_3_5 + Sudoku_2_3_6
 = 1
_C129: Sudoku_2_1_7 + Sudoku_2_1_8 + Sudoku_2_1_9 + Sudoku_2_2_7
 + Sudoku_2_2_8 + Sudoku_2_2_9 + Sudoku_2_3_7 + Sudoku_2_3_8 + Sudoku_2_3_9
 = 1
_C13: Sudoku_1_2_4 + Sudoku_2_2_4 + Sudoku_3_2_4 + Sudoku_4_2_4 + Sudoku_5_2_4
 + Sudoku_6_2_4 + Sudoku_7_2_4 + Sudoku_8_2_4 + Sudoku_9_2_4 = 1
_C130: Sudoku_2_4_1 + Sudoku_2_4_2 + Sudoku_2_4_3 + Sudoku_2_5_1
 + Sudoku_2_5_2 + Sudoku_2_5_3 + Sudoku_2_6_1 + Sudoku_2_6_2 + Sudoku_2_6_3
 = 1
_C131: Sudoku_2_4_4 + Sudoku_2_4_5 + Sudoku_2_4_6 + Sudoku_2_5_4
 + Sudoku_2_5_5 + Sudoku_2_5_6 + Sudoku_2_6_4 + Sudoku_2_6_5 + Sudoku_2_6_6
 = 1
_C132: Sudoku_2_4_7 + Sudoku_2_4_8 + Sudoku_2_4_9 + Sudoku_2_5_7
 + Sudoku_2_5_8 + Sudoku_2_5_9 + Sudoku_2_6_7 + Sudoku_2_6_8 + Sudoku_2_6_9
 = 1
_C133: Sudoku_2_7_1 + Sudoku_2_7_2 + Sudoku_2_7_3 + Sudoku_2_8_1
 + Sudoku_2_8_2 + Sudoku_2_8_3 + Sudoku_2_9_1 + Sudoku_2_9_2 + Sudoku_2_9_3
 = 1
_C134: Sudoku_2_7_4 + Sudoku_2_7_5 + Sudoku_2_7_6 + Sudoku_2_8_4
 + Sudoku_2_8_5 + Sudoku_2_8_6 + Sudoku_2_9_4 + Sudoku_2_9_5 + Sudoku_2_9_6
 = 1
_C135: Sudoku_2_7_7 + Sudoku_2_7_8 + Sudoku_2_7_9 + Sudoku_2_8_7
 + Sudoku_2_8_8 + Sudoku_2_8_9 + Sudoku_2_9_7 + Sudoku_2_9_8 + Sudoku_2_9_9
 = 1
_C136: Sudoku_3_1_1 + Sudoku_3_1_2 + Sudoku_3_1_3 + Sudoku_3_1_4
 + Sudoku_3_1_5 + Sudoku_3_1_6 + Sudoku_3_1_7 + Sudoku_3_1_8 + Sudoku_3_1_9
 = 1
_C137: Sudoku_3_2_1 + Sudoku_3_2_2 + Sudoku_3_2_3 + Sudoku_3_2_4
 + Sudoku_3_2_5 + Sudoku_3_2_6 + Sudoku_3_2_7 + Sudoku_3_2_8 + Sudoku_3_2_9
 = 1
_C138: Sudoku_3_3_1 + Sudoku_3_3_2 + Sudoku_3_3_3 + Sudoku_3_3_4
 + Sudoku_3_3_5 + Sudoku_3_3_6 + Sudoku_3_3_7 + Sudoku_3_3_8 + Sudoku_3_3_9
 = 1
_C139: Sudoku_3_4_1 + Sudoku_3_4_2 + Sudoku_3_4_3 + Sudoku_3_4_4
 + Sudoku_3_4_5 + Sudoku_3_4_6 + Sudoku_3_4_7 + Sudoku_3_4_8 + Sudoku_3_4_9
 = 1
_C14: Sudoku_1_2_5 + Sudoku_2_2_5 + Sudoku_3_2_5 + Sudoku_4_2_5 + Sudoku_5_2_5
 + Sudoku_6_2_5 + Sudoku_7_2_5 + Sudoku_8_2_5 + Sudoku_9_2_5 = 1
_C140: Sudoku_3_5_1 + Sudoku_3_5_2 + Sudoku_3_5_3 + Sudoku_3_5_4
 + Sudoku_3_5_5 + Sudoku_3_5_6 + Sudoku_3_5_7 + Sudoku_3_5_8 + Sudoku_3_5_9
 = 1
_C141: Sudoku_3_6_1 + Sudoku_3_6_2 + Sudoku_3_6_3 + Sudoku_3_6_4
 + Sudoku_3_6_5 + Sudoku_3_6_6 + Sudoku_3_6_7 + Sudoku_3_6_8 + Sudoku_3_6_9
 = 1
_C142: Sudoku_3_7_1 + Sudoku_3_7_2 + Sudoku_3_7_3 + Sudoku_3_7_4
 + Sudoku_3_7_5 + Sudoku_3_7_6 + Sudoku_3_7_7 + Sudoku_3_7_8 + Sudoku_3_7_9
 = 1
_C143: Sudoku_3_8_1 + Sudoku_3_8_2 + Sudoku_3_8_3 + Sudoku_3_8_4
 + Sudoku_3_8_5 + Sudoku_3_8_6 + Sudoku_3_8_7 + Sudoku_3_8_8 + Sudoku_3_8_9
 = 1
_C144: Sudoku_3_9_1 + Sudoku_3_9_2 + Sudoku_3_9_3 + Sudoku_3_9_4
 + Sudoku_3_9_5 + Sudoku_3_9_6 + Sudoku_3_9_7 + Sudoku_3_9_8 + Sudoku_3_9_9
 = 1
_C145: Sudoku_3_1_1 + Sudoku_3_2_1 + Sudoku_3_3_1 + Sudoku_3_4_1
 + Sudoku_3_5_1 + Sudoku_3_6_1 + Sudoku_3_7_1 + Sudoku_3_8_1 + Sudoku_3_9_1
 = 1
_C146: Sudoku_3_1_2 + Sudoku_3_2_2 + Sudoku_3_3_2 + Sudoku_3_4_2
 + Sudoku_3_5_2 + Sudoku_3_6_2 + Sudoku_3_7_2 + Sudoku_3_8_2 + Sudoku_3_9_2
 = 1
_C147: Sudoku_3_1_3 + Sudoku_3_2_3 + Sudoku_3_3_3 + Sudoku_3_4_3
 + Sudoku_3_5_3 + Sudoku_3_6_3 + Sudoku_3_7_3 + Sudoku_3_8_3 + Sudoku_3_9_3
 = 1
_C148: Sudoku_3_1_4 + Sudoku_3_2_4 + Sudoku_3_3_4 + Sudoku_3_4_4
 + Sudoku_3_5_4 + Sudoku_3_6_4 + Sudoku_3_7_4 + Sudoku_3_8_4 + Sudoku_3_9_4
 = 1
_C149: Sudoku_3_1_5 + Sudoku_3_2_5 + Sudoku_3_3_5 + Sudoku_3_4_5
 + Sudoku_3_5_5 + Sudoku_3_6_5 + Sudoku_3_7_5 + Sudoku_3_8_5 + Sudoku_3_9_5
 = 1
_C15: Sudoku_1_2_6 + Sudoku_2_2_6 + Sudoku_3_2_6 + Sudoku_4_2_6 + Sudoku_5_2_6
 + Sudoku_6_2_6 + Sudoku_7_2_6 + Sudoku_8_2_6 + Sudoku_9_2_6 = 1
_C150: Sudoku_3_1_6 + Sudoku_3_2_6 + Sudoku_3_3_6 + Sudoku_3_4_6
 + Sudoku_3_5_6 + Sudoku_3_6_6 + Sudoku_3_7_6 + Sudoku_3_8_6 + Sudoku_3_9_6
 = 1
_C151: Sudoku_3_1_7 + Sudoku_3_2_7 + Sudoku_3_3_7 + Sudoku_3_4_7
 + Sudoku_3_5_7 + Sudoku_3_6_7 + Sudoku_3_7_7 + Sudoku_3_8_7 + Sudoku_3_9_7
 = 1
_C152: Sudoku_3_1_8 + Sudoku_3_2_8 + Sudoku_3_3_8 + Sudoku_3_4_8
 + Sudoku_3_5_8 + Sudoku_3_6_8 + Sudoku_3_7_8 + Sudoku_3_8_8 + Sudoku_3_9_8
 = 1
_C153: Sudoku_3_1_9 + Sudoku_3_2_9 + Sudoku_3_3_9 + Sudoku_3_4_9
 + Sudoku_3_5_9 + Sudoku_3_6_9 + Sudoku_3_7_9 + Sudoku_3_8_9 + Sudoku_3_9_9
 = 1
_C154: Sudoku_3_1_1 + Sudoku_3_1_2 + Sudoku_3_1_3 + Sudoku_3_2_1
 + Sudoku_3_2_2 + Sudoku_3_2_3 + Sudoku_3_3_1 + Sudoku_3_3_2 + Sudoku_3_3_3
 = 1
_C155: Sudoku_3_1_4 + Sudoku_3_1_5 + Sudoku_3_1_6 + Sudoku_3_2_4
 + Sudoku_3_2_5 + Sudoku_3_2_6 + Sudoku_3_3_4 + Sudoku_3_3_5 + Sudoku_3_3_6
 = 1
_C156: Sudoku_3_1_7 + Sudoku_3_1_8 + Sudoku_3_1_9 + Sudoku_3_2_7
 + Sudoku_3_2_8 + Sudoku_3_2_9 + Sudoku_3_3_7 + Sudoku_3_3_8 + Sudoku_3_3_9
 = 1
_C157: Sudoku_3_4_1 + Sudoku_3_4_2 + Sudoku_3_4_3 + Sudoku_3_5_1
 + Sudoku_3_5_2 + Sudoku_3_5_3 + Sudoku_3_6_1 + Sudoku_3_6_2 + Sudoku_3_6_3
 = 1
_C158: Sudoku_3_4_4 + Sudoku_3_4_5 + Sudoku_3_4_6 + Sudoku_3_5_4
 + Sudoku_3_5_5 + Sudoku_3_5_6 + Sudoku_3_6_4 + Sudoku_3_6_5 + Sudoku_3_6_6
 = 1
_C159: Sudoku_3_4_7 + Sudoku_3_4_8 + Sudoku_3_4_9 + Sudoku_3_5_7
 + Sudoku_3_5_8 + Sudoku_3_5_9 + Sudoku_3_6_7 + Sudoku_3_6_8 + Sudoku_3_6_9
 = 1
_C16: Sudoku_1_2_7 + Sudoku_2_2_7 + Sudoku_3_2_7 + Sudoku_4_2_7 + Sudoku_5_2_7
 + Sudoku_6_2_7 + Sudoku_7_2_7 + Sudoku_8_2_7 + Sudoku_9_2_7 = 1
_C160: Sudoku_3_7_1 + Sudoku_3_7_2 + Sudoku_3_7_3 + Sudoku_3_8_1
 + Sudoku_3_8_2 + Sudoku_3_8_3 + Sudoku_3_9_1 + Sudoku_3_9_2 + Sudoku_3_9_3
 = 1
_C161: Sudoku_3_7_4 + Sudoku_3_7_5 + Sudoku_3_7_6 + Sudoku_3_8_4
 + Sudoku_3_8_5 + Sudoku_3_8_6 + Sudoku_3_9_4 + Sudoku_3_9_5 + Sudoku_3_9_6
 = 1
_C162: Sudoku_3_7_7 + Sudoku_3_7_8 + Sudoku_3_7_9 + Sudoku_3_8_7
 + Sudoku_3_8_8 + Sudoku_3_8_9 + Sudoku_3_9_7 + Sudoku_3_9_8 + Sudoku_3_9_9
 = 1
_C163: Sudoku_4_1_1 + Sudoku_4_1_2 + Sudoku_4_1_3 + Sudoku_4_1_4
 + Sudoku_4_1_5 + Sudoku_4_1_6 + Sudoku_4_1_7 + Sudoku_4_1_8 + Sudoku_4_1_9
 = 1
_C164: Sudoku_4_2_1 + Sudoku_4_2_2 + Sudoku_4_2_3 + Sudoku_4_2_4
 + Sudoku_4_2_5 + Sudoku_4_2_6 + Sudoku_4_2_7 + Sudoku_4_2_8 + Sudoku_4_2_9
 = 1
_C165: Sudoku_4_3_1 + Sudoku_4_3_2 + Sudoku_4_3_3 + Sudoku_4_3_4
 + Sudoku_4_3_5 + Sudoku_4_3_6 + Sudoku_4_3_7 + Sudoku_4_3_8 + Sudoku_4_3_9
 = 1
_C166: Sudoku_4_4_1 + Sudoku_4_4_2 + Sudoku_4_4_3 + Sudoku_4_4_4
 + Sudoku_4_4_5 + Sudoku_4_4_6 + Sudoku_4_4_7 + Sudoku_4_4_8 + Sudoku_4_4_9
 = 1
_C167: Sudoku_4_5_1 + Sudoku_4_5_2 + Sudoku_4_5_3 + Sudoku_4_5_4
 + Sudoku_4_5_5 + Sudoku_4_5_6 + Sudoku_4_5_7 + Sudoku_4_5_8 + Sudoku_4_5_9
 = 1
_C168: Sudoku_4_6_1 + Sudoku_4_6_2 + Sudoku_4_6_3 + Sudoku_4_6_4
 + Sudoku_4_6_5 + Sudoku_4_6_6 + Sudoku_4_6_7 + Sudoku_4_6_8 + Sudoku_4_6_9
 = 1
_C169: Sudoku_4_7_1 + Sudoku_4_7_2 + Sudoku_4_7_3 + Sudoku_4_7_4
 + Sudoku_4_7_5 + Sudoku_4_7_6 + Sudoku_4_7_7 + Sudoku_4_7_8 + Sudoku_4_7_9
 = 1
_C17: Sudoku_1_2_8 + Sudoku_2_2_8 + Sudoku_3_2_8 + Sudoku_4_2_8 + Sudoku_5_2_8
 + Sudoku_6_2_8 + Sudoku_7_2_8 + Sudoku_8_2_8 + Sudoku_9_2_8 = 1
_C170: Sudoku_4_8_1 + Sudoku_4_8_2 + Sudoku_4_8_3 + Sudoku_4_8_4
 + Sudoku_4_8_5 + Sudoku_4_8_6 + Sudoku_4_8_7 + Sudoku_4_8_8 + Sudoku_4_8_9
 = 1
_C171: Sudoku_4_9_1 + Sudoku_4_9_2 + Sudoku_4_9_3 + Sudoku_4_9_4
 + Sudoku_4_9_5 + Sudoku_4_9_6 + Sudoku_4_9_7 + Sudoku_4_9_8 + Sudoku_4_9_9
 = 1
_C172: Sudoku_4_1_1 + Sudoku_4_2_1 + Sudoku_4_3_1 + Sudoku_4_4_1
 + Sudoku_4_5_1 + Sudoku_4_6_1 + Sudoku_4_7_1 + Sudoku_4_8_1 + Sudoku_4_9_1
 = 1
_C173: Sudoku_4_1_2 + Sudoku_4_2_2 + Sudoku_4_3_2 + Sudoku_4_4_2
 + Sudoku_4_5_2 + Sudoku_4_6_2 + Sudoku_4_7_2 + Sudoku_4_8_2 + Sudoku_4_9_2
 = 1
_C174: Sudoku_4_1_3 + Sudoku_4_2_3 + Sudoku_4_3_3 + Sudoku_4_4_3
 + Sudoku_4_5_3 + Sudoku_4_6_3 + Sudoku_4_7_3 + Sudoku_4_8_3 + Sudoku_4_9_3
 = 1
_C175: Sudoku_4_1_4 + Sudoku_4_2_4 + Sudoku_4_3_4 + Sudoku_4_4_4
 + Sudoku_4_5_4 + Sudoku_4_6_4 + Sudoku_4_7_4 + Sudoku_4_8_4 + Sudoku_4_9_4
 = 1
_C176: Sudoku_4_1_5 + Sudoku_4_2_5 + Sudoku_4_3_5 + Sudoku_4_4_5
 + Sudoku_4_5_5 + Sudoku_4_6_5 + Sudoku_4_7_5 + Sudoku_4_8_5 + Sudoku_4_9_5
 = 1
_C177: Sudoku_4_1_6 + Sudoku_4_2_6 + Sudoku_4_3_6 + Sudoku_4_4_6
 + Sudoku_4_5_6 + Sudoku_4_6_6 + Sudoku_4_7_6 + Sudoku_4_8_6 + Sudoku_4_9_6
 = 1
_C178: Sudoku_4_1_7 + Sudoku_4_2_7 + Sudoku_4_3_7 + Sudoku_4_4_7
 + Sudoku_4_5_7 + Sudoku_4_6_7 + Sudoku_4_7_7 + Sudoku_4_8_7 + Sudoku_4_9_7
 = 1
_C179: Sudoku_4_1_8 + Sudoku_4_2_8 + Sudoku_4_3_8 + Sudoku_4_4_8
 + Sudoku_4_5_8 + Sudoku_4_6_8 + Sudoku_4_7_8 + Sudoku_4_8_8 + Sudoku_4_9_8
 = 1
_C18: Sudoku_1_2_9 + Sudoku_2_2_9 + Sudoku_3_2_9 + Sudoku_4_2_9 + Sudoku_5_2_9
 + Sudoku_6_2_9 + Sudoku_7_2_9 + Sudoku_8_2_9 + Sudoku_9_2_9 = 1
_C180: Sudoku_4_1_9 + Sudoku_4_2_9 + Sudoku_4_3_9 + Sudoku_4_4_9
 + Sudoku_4_5_9 + Sudoku_4_6_9 + Sudoku_4_7_9 + Sudoku_4_8_9 + Sudoku_4_9_9
 = 1
_C181: Sudoku_4_1_1 + Sudoku_4_1_2 + Sudoku_4_1_3 + Sudoku_4_2_1
 + Sudoku_4_2_2 + Sudoku_4_2_3 + Sudoku_4_3_1 + Sudoku_4_3_2 + Sudoku_4_3_3
 = 1
_C182: Sudoku_4_1_4 + Sudoku_4_1_5 + Sudoku_4_1_6 + Sudoku_4_2_4
 + Sudoku_4_2_5 + Sudoku_4_2_6 + Sudoku_4_3_4 + Sudoku_4_3_5 + Sudoku_4_3_6
 = 1
_C183: Sudoku_4_1_7 + Sudoku_4_1_8 + Sudoku_4_1_9 + Sudoku_4_2_7
 + Sudoku_4_2_8 + Sudoku_4_2_9 + Sudoku_4_3_7 + Sudoku_4_3_8 + Sudoku_4_3_9
 = 1
_C184: Sudoku_4_4_1 + Sudoku_4_4_2 + Sudoku_4_4_3 + Sudoku_4_5_1
 + Sudoku_4_5_2 + Sudoku_4_5_3 + Sudoku_4_6_1 + Sudoku_4_6_2 + Sudoku_4_6_3
 = 1
_C185: Sudoku_4_4_4 + Sudoku_4_4_5 + Sudoku_4_4_6 + Sudoku_4_5_4
 + Sudoku_4_5_5 + Sudoku_4_5_6 + Sudoku_4_6_4 + Sudoku_4_6_5 + Sudoku_4_6_6
 = 1
_C186: Sudoku_4_4_7 + Sudoku_4_4_8 + Sudoku_4_4_9 + Sudoku_4_5_7
 + Sudoku_4_5_8 + Sudoku_4_5_9 + Sudoku_4_6_7 + Sudoku_4_6_8 + Sudoku_4_6_9
 = 1
_C187: Sudoku_4_7_1 + Sudoku_4_7_2 + Sudoku_4_7_3 + Sudoku_4_8_1
 + Sudoku_4_8_2 + Sudoku_4_8_3 + Sudoku_4_9_1 + Sudoku_4_9_2 + Sudoku_4_9_3
 = 1
_C188: Sudoku_4_7_4 + Sudoku_4_7_5 + Sudoku_4_7_6 + Sudoku_4_8_4
 + Sudoku_4_8_5 + Sudoku_4_8_6 + Sudoku_4_9_4 + Sudoku_4_9_5 + Sudoku_4_9_6
 = 1
_C189: Sudoku_4_7_7 + Sudoku_4_7_8 + Sudoku_4_7_9 + Sudoku_4_8_7
 + Sudoku_4_8_8 + Sudoku_4_8_9 + Sudoku_4_9_7 + Sudoku_4_9_8 + Sudoku_4_9_9
 = 1
_C19: Sudoku_1_3_1 + Sudoku_2_3_1 + Sudoku_3_3_1 + Sudoku_4_3_1 + Sudoku_5_3_1
 + Sudoku_6_3_1 + Sudoku_7_3_1 + Sudoku_8_3_1 + Sudoku_9_3_1 = 1
_C190: Sudoku_5_1_1 + Sudoku_5_1_2 + Sudoku_5_1_3 + Sudoku_5_1_4
 + Sudoku_5_1_5 + Sudoku_5_1_6 + Sudoku_5_1_7 + Sudoku_5_1_8 + Sudoku_5_1_9
 = 1
_C191: Sudoku_5_2_1 + Sudoku_5_2_2 + Sudoku_5_2_3 + Sudoku_5_2_4
 + Sudoku_5_2_5 + Sudoku_5_2_6 + Sudoku_5_2_7 + Sudoku_5_2_8 + Sudoku_5_2_9
 = 1
_C192: Sudoku_5_3_1 + Sudoku_5_3_2 + Sudoku_5_3_3 + Sudoku_5_3_4
 + Sudoku_5_3_5 + Sudoku_5_3_6 + Sudoku_5_3_7 + Sudoku_5_3_8 + Sudoku_5_3_9
 = 1
_C193: Sudoku_5_4_1 + Sudoku_5_4_2 + Sudoku_5_4_3 + Sudoku_5_4_4
 + Sudoku_5_4_5 + Sudoku_5_4_6 + Sudoku_5_4_7 + Sudoku_5_4_8 + Sudoku_5_4_9
 = 1
_C194: Sudoku_5_5_1 + Sudoku_5_5_2 + Sudoku_5_5_3 + Sudoku_5_5_4
 + Sudoku_5_5_5 + Sudoku_5_5_6 + Sudoku_5_5_7 + Sudoku_5_5_8 + Sudoku_5_5_9
 = 1
_C195: Sudoku_5_6_1 + Sudoku_5_6_2 + Sudoku_5_6_3 + Sudoku_5_6_4
 + Sudoku_5_6_5 + Sudoku_5_6_6 + Sudoku_5_6_7 + Sudoku_5_6_8 + Sudoku_5_6_9
 = 1
_C196: Sudoku_5_7_1 + Sudoku_5_7_2 + Sudoku_5_7_3 + Sudoku_5_7_4
 + Sudoku_5_7_5 + Sudoku_5_7_6 + Sudoku_5_7_7 + Sudoku_5_7_8 + Sudoku_5_7_9
 = 1
_C197: Sudoku_5_8_1 + Sudoku_5_8_2 + Sudoku_5_8_3 + Sudoku_5_8_4
 + Sudoku_5_8_5 + Sudoku_5_8_6 + Sudoku_5_8_7 + Sudoku_5_8_8 + Sudoku_5_8_9
 = 1
_C198: Sudoku_5_9_1 + Sudoku_5_9_2 + Sudoku_5_9_3 + Sudoku_5_9_4
 + Sudoku_5_9_5 + Sudoku_5_9_6 + Sudoku_5_9_7 + Sudoku_5_9_8 + Sudoku_5_9_9
 = 1
_C199: Sudoku_5_1_1 + Sudoku_5_2_1 + Sudoku_5_3_1 + Sudoku_5_4_1
 + Sudoku_5_5_1 + Sudoku_5_6_1 + Sudoku_5_7_1 + Sudoku_5_8_1 + Sudoku_5_9_1
 = 1
_C2: Sudoku_1_1_2 + Sudoku_2_1_2 + Sudoku_3_1_2 + Sudoku_4_1_2 + Sudoku_5_1_2
 + Sudoku_6_1_2 + Sudoku_7_1_2 + Sudoku_8_1_2 + Sudoku_9_1_2 = 1
_C20: Sudoku_1_3_2 + Sudoku_2_3_2 + Sudoku_3_3_2 + Sudoku_4_3_2 + Sudoku_5_3_2
 + Sudoku_6_3_2 + Sudoku_7_3_2 + Sudoku_8_3_2 + Sudoku_9_3_2 = 1
_C200: Sudoku_5_1_2 + Sudoku_5_2_2 + Sudoku_5_3_2 + Sudoku_5_4_2
 + Sudoku_5_5_2 + Sudoku_5_6_2 + Sudoku_5_7_2 + Sudoku_5_8_2 + Sudoku_5_9_2
 = 1
_C201: Sudoku_5_1_3 + Sudoku_5_2_3 + Sudoku_5_3_3 + Sudoku_5_4_3
 + Sudoku_5_5_3 + Sudoku_5_6_3 + Sudoku_5_7_3 + Sudoku_5_8_3 + Sudoku_5_9_3
 = 1
_C202: Sudoku_5_1_4 + Sudoku_5_2_4 + Sudoku_5_3_4 + Sudoku_5_4_4
 + Sudoku_5_5_4 + Sudoku_5_6_4 + Sudoku_5_7_4 + Sudoku_5_8_4 + Sudoku_5_9_4
 = 1
_C203: Sudoku_5_1_5 + Sudoku_5_2_5 + Sudoku_5_3_5 + Sudoku_5_4_5
 + Sudoku_5_5_5 + Sudoku_5_6_5 + Sudoku_5_7_5 + Sudoku_5_8_5 + Sudoku_5_9_5
 = 1
_C204: Sudoku_5_1_6 + Sudoku_5_2_6 + Sudoku_5_3_6 + Sudoku_5_4_6
 + Sudoku_5_5_6 + Sudoku_5_6_6 + Sudoku_5_7_6 + Sudoku_5_8_6 + Sudoku_5_9_6
 = 1
_C205: Sudoku_5_1_7 + Sudoku_5_2_7 + Sudoku_5_3_7 + Sudoku_5_4_7
 + Sudoku_5_5_7 + Sudoku_5_6_7 + Sudoku_5_7_7 + Sudoku_5_8_7 + Sudoku_5_9_7
 = 1
_C206: Sudoku_5_1_8 + Sudoku_5_2_8 + Sudoku_5_3_8 + Sudoku_5_4_8
 + Sudoku_5_5_8 + Sudoku_5_6_8 + Sudoku_5_7_8 + Sudoku_5_8_8 + Sudoku_5_9_8
 = 1
_C207: Sudoku_5_1_9 + Sudoku_5_2_9 + Sudoku_5_3_9 + Sudoku_5_4_9
 + Sudoku_5_5_9 + Sudoku_5_6_9 + Sudoku_5_7_9 + Sudoku_5_8_9 + Sudoku_5_9_9
 = 1
_C208: Sudoku_5_1_1 + Sudoku_5_1_2 + Sudoku_5_1_3 + Sudoku_5_2_1
 + Sudoku_5_2_2 + Sudoku_5_2_3 + Sudoku_5_3_1 + Sudoku_5_3_2 + Sudoku_5_3_3
 = 1
_C209: Sudoku_5_1_4 + Sudoku_5_1_5 + Sudoku_5_1_6 + Sudoku_5_2_4
 + Sudoku_5_2_5 + Sudoku_5_2_6 + Sudoku_5_3_4 + Sudoku_5_3_5 + Sudoku_5_3_6
 = 1
_C21: Sudoku_1_3_3 + Sudoku_2_3_3 + Sudoku_3_3_3 + Sudoku_4_3_3 + Sudoku_5_3_3
 + Sudoku_6_3_3 + Sudoku_7_3_3 + Sudoku_8_3_3 + Sudoku_9_3_3 = 1
_C210: Sudoku_5_1_7 + Sudoku_5_1_8 + Sudoku_5_1_9 + Sudoku_5_2_7
 + Sudoku_5_2_8 + Sudoku_5_2_9 + Sudoku_5_3_7 + Sudoku_5_3_8 + Sudoku_5_3_9
 = 1
_C211: Sudoku_5_4_1 + Sudoku_5_4_2 + Sudoku_5_4_3 + Sudoku_5_5_1
 + Sudoku_5_5_2 + Sudoku_5_5_3 + Sudoku_5_6_1 + Sudoku_5_6_2 + Sudoku_5_6_3
 = 1
_C212: Sudoku_5_4_4 + Sudoku_5_4_5 + Sudoku_5_4_6 + Sudoku_5_5_4
 + Sudoku_5_5_5 + Sudoku_5_5_6 + Sudoku_5_6_4 + Sudoku_5_6_5 + Sudoku_5_6_6
 = 1
_C213: Sudoku_5_4_7 + Sudoku_5_4_8 + Sudoku_5_4_9 + Sudoku_5_5_7
 + Sudoku_5_5_8 + Sudoku_5_5_9 + Sudoku_5_6_7 + Sudoku_5_6_8 + Sudoku_5_6_9
 = 1
_C214: Sudoku_5_7_1 + Sudoku_5_7_2 + Sudoku_5_7_3 + Sudoku_5_8_1
 + Sudoku_5_8_2 + Sudoku_5_8_3 + Sudoku_5_9_1 + Sudoku_5_9_2 + Sudoku_5_9_3
 = 1
_C215: Sudoku_5_7_4 + Sudoku_5_7_5 + Sudoku_5_7_6 + Sudoku_5_8_4
 + Sudoku_5_8_5 + Sudoku_5_8_6 + Sudoku_5_9_4 + Sudoku_5_9_5 + Sudoku_5_9_6
 = 1
_C216: Sudoku_5_7_7 + Sudoku_5_7_8 + Sudoku_5_7_9 + Sudoku_5_8_7
 + Sudoku_5_8_8 + Sudoku_5_8_9 + Sudoku_5_9_7 + Sudoku_5_9_8 + Sudoku_5_9_9
 = 1
_C217: Sudoku_6_1_1 + Sudoku_6_1_2 + Sudoku_6_1_3 + Sudoku_6_1_4
 + Sudoku_6_1_5 + Sudoku_6_1_6 + Sudoku_6_1_7 + Sudoku_6_1_8 + Sudoku_6_1_9
 = 1
_C218: Sudoku_6_2_1 + Sudoku_6_2_2 + Sudoku_6_2_3 + Sudoku_6_2_4
 + Sudoku_6_2_5 + Sudoku_6_2_6 + Sudoku_6_2_7 + Sudoku_6_2_8 + Sudoku_6_2_9
 = 1
_C219: Sudoku_6_3_1 + Sudoku_6_3_2 + Sudoku_6_3_3 + Sudoku_6_3_4
 + Sudoku_6_3_5 + Sudoku_6_3_6 + Sudoku_6_3_7 + Sudoku_6_3_8 + Sudoku_6_3_9
 = 1
_C22: Sudoku_1_3_4 + Sudoku_2_3_4 + Sudoku_3_3_4 + Sudoku_4_3_4 + Sudoku_5_3_4
 + Sudoku_6_3_4 + Sudoku_7_3_4 + Sudoku_8_3_4 + Sudoku_9_3_4 = 1
_C220: Sudoku_6_4_1 + Sudoku_6_4_2 + Sudoku_6_4_3 + Sudoku_6_4_4
 + Sudoku_6_4_5 + Sudoku_6_4_6 + Sudoku_6_4_7 + Sudoku_6_4_8 + Sudoku_6_4_9
 = 1
_C221: Sudoku_6_5_1 + Sudoku_6_5_2 + Sudoku_6_5_3 + Sudoku_6_5_4
 + Sudoku_6_5_5 + Sudoku_6_5_6 + Sudoku_6_5_7 + Sudoku_6_5_8 + Sudoku_6_5_9
 = 1
_C222: Sudoku_6_6_1 + Sudoku_6_6_2 + Sudoku_6_6_3 + Sudoku_6_6_4
 + Sudoku_6_6_5 + Sudoku_6_6_6 + Sudoku_6_6_7 + Sudoku_6_6_8 + Sudoku_6_6_9
 = 1
_C223: Sudoku_6_7_1 + Sudoku_6_7_2 + Sudoku_6_7_3 + Sudoku_6_7_4
 + Sudoku_6_7_5 + Sudoku_6_7_6 + Sudoku_6_7_7 + Sudoku_6_7_8 + Sudoku_6_7_9
 = 1
_C224: Sudoku_6_8_1 + Sudoku_6_8_2 + Sudoku_6_8_3 + Sudoku_6_8_4
 + Sudoku_6_8_5 + Sudoku_6_8_6 + Sudoku_6_8_7 + Sudoku_6_8_8 + Sudoku_6_8_9
 = 1
_C225: Sudoku_6_9_1 + Sudoku_6_9_2 + Sudoku_6_9_3 + Sudoku_6_9_4
 + Sudoku_6_9_5 + Sudoku_6_9_6 + Sudoku_6_9_7 + Sudoku_6_9_8 + Sudoku_6_9_9
 = 1
_C226: Sudoku_6_1_1 + Sudoku_6_2_1 + Sudoku_6_3_1 + Sudoku_6_4_1
 + Sudoku_6_5_1 + Sudoku_6_6_1 + Sudoku_6_7_1 + Sudoku_6_8_1 + Sudoku_6_9_1
 = 1
_C227: Sudoku_6_1_2 + Sudoku_6_2_2 + Sudoku_6_3_2 + Sudoku_6_4_2
 + Sudoku_6_5_2 + Sudoku_6_6_2 + Sudoku_6_7_2 + Sudoku_6_8_2 + Sudoku_6_9_2
 = 1
_C228: Sudoku_6_1_3 + Sudoku_6_2_3 + Sudoku_6_3_3 + Sudoku_6_4_3
 + Sudoku_6_5_3 + Sudoku_6_6_3 + Sudoku_6_7_3 + Sudoku_6_8_3 + Sudoku_6_9_3
 = 1
_C229: Sudoku_6_1_4 + Sudoku_6_2_4 + Sudoku_6_3_4 + Sudoku_6_4_4
 + Sudoku_6_5_4 + Sudoku_6_6_4 + Sudoku_6_7_4 + Sudoku_6_8_4 + Sudoku_6_9_4
 = 1
_C23: Sudoku_1_3_5 + Sudoku_2_3_5 + Sudoku_3_3_5 + Sudoku_4_3_5 + Sudoku_5_3_5
 + Sudoku_6_3_5 + Sudoku_7_3_5 + Sudoku_8_3_5 + Sudoku_9_3_5 = 1
_C230: Sudoku_6_1_5 + Sudoku_6_2_5 + Sudoku_6_3_5 + Sudoku_6_4_5
 + Sudoku_6_5_5 + Sudoku_6_6_5 + Sudoku_6_7_5 + Sudoku_6_8_5 + Sudoku_6_9_5
 = 1
_C231: Sudoku_6_1_6 + Sudoku_6_2_6 + Sudoku_6_3_6 + Sudoku_6_4_6
 + Sudoku_6_5_6 + Sudoku_6_6_6 + Sudoku_6_7_6 + Sudoku_6_8_6 + Sudoku_6_9_6
 = 1
_C232: Sudoku_6_1_7 + Sudoku_6_2_7 + Sudoku_6_3_7 + Sudoku_6_4_7
 + Sudoku_6_5_7 + Sudoku_6_6_7 + Sudoku_6_7_7 + Sudoku_6_8_7 + Sudoku_6_9_7
 = 1
_C233: Sudoku_6_1_8 + Sudoku_6_2_8 + Sudoku_6_3_8 + Sudoku_6_4_8
 + Sudoku_6_5_8 + Sudoku_6_6_8 + Sudoku_6_7_8 + Sudoku_6_8_8 + Sudoku_6_9_8
 = 1
_C234: Sudoku_6_1_9 + Sudoku_6_2_9 + Sudoku_6_3_9 + Sudoku_6_4_9
 + Sudoku_6_5_9 + Sudoku_6_6_9 + Sudoku_6_7_9 + Sudoku_6_8_9 + Sudoku_6_9_9
 = 1
_C235: Sudoku_6_1_1 + Sudoku_6_1_2 + Sudoku_6_1_3 + Sudoku_6_2_1
 + Sudoku_6_2_2 + Sudoku_6_2_3 + Sudoku_6_3_1 + Sudoku_6_3_2 + Sudoku_6_3_3
 = 1
_C236: Sudoku_6_1_4 + Sudoku_6_1_5 + Sudoku_6_1_6 + Sudoku_6_2_4
 + Sudoku_6_2_5 + Sudoku_6_2_6 + Sudoku_6_3_4 + Sudoku_6_3_5 + Sudoku_6_3_6
 = 1
_C237: Sudoku_6_1_7 + Sudoku_6_1_8 + Sudoku_6_1_9 + Sudoku_6_2_7
 + Sudoku_6_2_8 + Sudoku_6_2_9 + Sudoku_6_3_7 + Sudoku_6_3_8 + Sudoku_6_3_9
 = 1
_C238: Sudoku_6_4_1 + Sudoku_6_4_2 + Sudoku_6_4_3 + Sudoku_6_5_1
 + Sudoku_6_5_2 + Sudoku_6_5_3 + Sudoku_6_6_1 + Sudoku_6_6_2 + Sudoku_6_6_3
 = 1
_C239: Sudoku_6_4_4 + Sudoku_6_4_5 + Sudoku_6_4_6 + Sudoku_6_5_4
 + Sudoku_6_5_5 + Sudoku_6_5_6 + Sudoku_6_6_4 + Sudoku_6_6_5 + Sudoku_6_6_6
 = 1
_C24: Sudoku_1_3_6 + Sudoku_2_3_6 + Sudoku_3_3_6 + Sudoku_4_3_6 + Sudoku_5_3_6
 + Sudoku_6_3_6 + Sudoku_7_3_6 + Sudoku_8_3_6 + Sudoku_9_3_6 = 1
_C240: Sudoku_6_4_7 + Sudoku_6_4_8 + Sudoku_6_4_9 + Sudoku_6_5_7
 + Sudoku_6_5_8 + Sudoku_6_5_9 + Sudoku_6_6_7 + Sudoku_6_6_8 + Sudoku_6_6_9
 = 1
_C241: Sudoku_6_7_1 + Sudoku_6_7_2 + Sudoku_6_7_3 + Sudoku_6_8_1
 + Sudoku_6_8_2 + Sudoku_6_8_3 + Sudoku_6_9_1 + Sudoku_6_9_2 + Sudoku_6_9_3
 = 1
_C242: Sudoku_6_7_4 + Sudoku_6_7_5 + Sudoku_6_7_6 + Sudoku_6_8_4
 + Sudoku_6_8_5 + Sudoku_6_8_6 + Sudoku_6_9_4 + Sudoku_6_9_5 + Sudoku_6_9_6
 = 1
_C243: Sudoku_6_7_7 + Sudoku_6_7_8 + Sudoku_6_7_9 + Sudoku_6_8_7
 + Sudoku_6_8_8 + Sudoku_6_8_9 + Sudoku_6_9_7 + Sudoku_6_9_8 + Sudoku_6_9_9
 = 1
_C244: Sudoku_7_1_1 + Sudoku_7_1_2 + Sudoku_7_1_3 + Sudoku_7_1_4
 + Sudoku_7_1_5 + Sudoku_7_1_6 + Sudoku_7_1_7 + Sudoku_7_1_8 + Sudoku_7_1_9
 = 1
_C245: Sudoku_7_2_1 + Sudoku_7_2_2 + Sudoku_7_2_3 + Sudoku_7_2_4
 + Sudoku_7_2_5 + Sudoku_7_2_6 + Sudoku_7_2_7 + Sudoku_7_2_8 + Sudoku_7_2_9
 = 1
_C246: Sudoku_7_3_1 + Sudoku_7_3_2 + Sudoku_7_3_3 + Sudoku_7_3_4
 + Sudoku_7_3_5 + Sudoku_7_3_6 + Sudoku_7_3_7 + Sudoku_7_3_8 + Sudoku_7_3_9
 = 1
_C247: Sudoku_7_4_1 + Sudoku_7_4_2 + Sudoku_7_4_3 + Sudoku_7_4_4
 + Sudoku_7_4_5 + Sudoku_7_4_6 + Sudoku_7_4_7 + Sudoku_7_4_8 + Sudoku_7_4_9
 = 1
_C248: Sudoku_7_5_1 + Sudoku_7_5_2 + Sudoku_7_5_3 + Sudoku_7_5_4
 + Sudoku_7_5_5 + Sudoku_7_5_6 + Sudoku_7_5_7 + Sudoku_7_5_8 + Sudoku_7_5_9
 = 1
_C249: Sudoku_7_6_1 + Sudoku_7_6_2 + Sudoku_7_6_3 + Sudoku_7_6_4
 + Sudoku_7_6_5 + Sudoku_7_6_6 + Sudoku_7_6_7 + Sudoku_7_6_8 + Sudoku_7_6_9
 = 1
_C25: Sudoku_1_3_7 + Sudoku_2_3_7 + Sudoku_3_3_7 + Sudoku_4_3_7 + Sudoku_5_3_7
 + Sudoku_6_3_7 + Sudoku_7_3_7 + Sudoku_8_3_7 + Sudoku_9_3_7 = 1
_C250: Sudoku_7_7_1 + Sudoku_7_7_2 + Sudoku_7_7_3 + Sudoku_7_7_4
 + Sudoku_7_7_5 + Sudoku_7_7_6 + Sudoku_7_7_7 + Sudoku_7_7_8 + Sudoku_7_7_9
 = 1
_C251: Sudoku_7_8_1 + Sudoku_7_8_2 + Sudoku_7_8_3 + Sudoku_7_8_4
 + Sudoku_7_8_5 + Sudoku_7_8_6 + Sudoku_7_8_7 + Sudoku_7_8_8 + Sudoku_7_8_9
 = 1
_C252: Sudoku_7_9_1 + Sudoku_7_9_2 + Sudoku_7_9_3 + Sudoku_7_9_4
 + Sudoku_7_9_5 + Sudoku_7_9_6 + Sudoku_7_9_7 + Sudoku_7_9_8 + Sudoku_7_9_9
 = 1
_C253: Sudoku_7_1_1 + Sudoku_7_2_1 + Sudoku_7_3_1 + Sudoku_7_4_1
 + Sudoku_7_5_1 + Sudoku_7_6_1 + Sudoku_7_7_1 + Sudoku_7_8_1 + Sudoku_7_9_1
 = 1
_C254: Sudoku_7_1_2 + Sudoku_7_2_2 + Sudoku_7_3_2 + Sudoku_7_4_2
 + Sudoku_7_5_2 + Sudoku_7_6_2 + Sudoku_7_7_2 + Sudoku_7_8_2 + Sudoku_7_9_2
 = 1
_C255: Sudoku_7_1_3 + Sudoku_7_2_3 + Sudoku_7_3_3 + Sudoku_7_4_3
 + Sudoku_7_5_3 + Sudoku_7_6_3 + Sudoku_7_7_3 + Sudoku_7_8_3 + Sudoku_7_9_3
 = 1
_C256: Sudoku_7_1_4 + Sudoku_7_2_4 + Sudoku_7_3_4 + Sudoku_7_4_4
 + Sudoku_7_5_4 + Sudoku_7_6_4 + Sudoku_7_7_4 + Sudoku_7_8_4 + Sudoku_7_9_4
 = 1
_C257: Sudoku_7_1_5 + Sudoku_7_2_5 + Sudoku_7_3_5 + Sudoku_7_4_5
 + Sudoku_7_5_5 + Sudoku_7_6_5 + Sudoku_7_7_5 + Sudoku_7_8_5 + Sudoku_7_9_5
 = 1
_C258: Sudoku_7_1_6 + Sudoku_7_2_6 + Sudoku_7_3_6 + Sudoku_7_4_6
 + Sudoku_7_5_6 + Sudoku_7_6_6 + Sudoku_7_7_6 + Sudoku_7_8_6 + Sudoku_7_9_6
 = 1
_C259: Sudoku_7_1_7 + Sudoku_7_2_7 + Sudoku_7_3_7 + Sudoku_7_4_7
 + Sudoku_7_5_7 + Sudoku_7_6_7 + Sudoku_7_7_7 + Sudoku_7_8_7 + Sudoku_7_9_7
 = 1
_C26: Sudoku_1_3_8 + Sudoku_2_3_8 + Sudoku_3_3_8 + Sudoku_4_3_8 + Sudoku_5_3_8
 + Sudoku_6_3_8 + Sudoku_7_3_8 + Sudoku_8_3_8 + Sudoku_9_3_8 = 1
_C260: Sudoku_7_1_8 + Sudoku_7_2_8 + Sudoku_7_3_8 + Sudoku_7_4_8
 + Sudoku_7_5_8 + Sudoku_7_6_8 + Sudoku_7_7_8 + Sudoku_7_8_8 + Sudoku_7_9_8
 = 1
_C261: Sudoku_7_1_9 + Sudoku_7_2_9 + Sudoku_7_3_9 + Sudoku_7_4_9
 + Sudoku_7_5_9 + Sudoku_7_6_9 + Sudoku_7_7_9 + Sudoku_7_8_9 + Sudoku_7_9_9
 = 1
_C262: Sudoku_7_1_1 + Sudoku_7_1_2 + Sudoku_7_1_3 + Sudoku_7_2_1
 + Sudoku_7_2_2 + Sudoku_7_2_3 + Sudoku_7_3_1 + Sudoku_7_3_2 + Sudoku_7_3_3
 = 1
_C263: Sudoku_7_1_4 + Sudoku_7_1_5 + Sudoku_7_1_6 + Sudoku_7_2_4
 + Sudoku_7_2_5 + Sudoku_7_2_6 + Sudoku_7_3_4 + Sudoku_7_3_5 + Sudoku_7_3_6
 = 1
_C264: Sudoku_7_1_7 + Sudoku_7_1_8 + Sudoku_7_1_9 + Sudoku_7_2_7
 + Sudoku_7_2_8 + Sudoku_7_2_9 + Sudoku_7_3_7 + Sudoku_7_3_8 + Sudoku_7_3_9
 = 1
_C265: Sudoku_7_4_1 + Sudoku_7_4_2 + Sudoku_7_4_3 + Sudoku_7_5_1
 + Sudoku_7_5_2 + Sudoku_7_5_3 + Sudoku_7_6_1 + Sudoku_7_6_2 + Sudoku_7_6_3
 = 1
_C266: Sudoku_7_4_4 + Sudoku_7_4_5 + Sudoku_7_4_6 + Sudoku_7_5_4
 + Sudoku_7_5_5 + Sudoku_7_5_6 + Sudoku_7_6_4 + Sudoku_7_6_5 + Sudoku_7_6_6
 = 1
_C267: Sudoku_7_4_7 + Sudoku_7_4_8 + Sudoku_7_4_9 + Sudoku_7_5_7
 + Sudoku_7_5_8 + Sudoku_7_5_9 + Sudoku_7_6_7 + Sudoku_7_6_8 + Sudoku_7_6_9
 = 1
_C268: Sudoku_7_7_1 + Sudoku_7_7_2 + Sudoku_7_7_3 + Sudoku_7_8_1
 + Sudoku_7_8_2 + Sudoku_7_8_3 + Sudoku_7_9_1 + Sudoku_7_9_2 + Sudoku_7_9_3
 = 1
_C269: Sudoku_7_7_4 + Sudoku_7_7_5 + Sudoku_7_7_6 + Sudoku_7_8_4
 + Sudoku_7_8_5 + Sudoku_7_8_6 + Sudoku_7_9_4 + Sudoku_7_9_5 + Sudoku_7_9_6
 = 1
_C27: Sudoku_1_3_9 + Sudoku_2_3_9 + Sudoku_3_3_9 + Sudoku_4_3_9 + Sudoku_5_3_9
 + Sudoku_6_3_9 + Sudoku_7_3_9 + Sudoku_8_3_9 + Sudoku_9_3_9 = 1
_C270: Sudoku_7_7_7 + Sudoku_7_7_8 + Sudoku_7_7_9 + Sudoku_7_8_7
 + Sudoku_7_8_8 + Sudoku_7_8_9 + Sudoku_7_9_7 + Sudoku_7_9_8 + Sudoku_7_9_9
 = 1
_C271: Sudoku_8_1_1 + Sudoku_8_1_2 + Sudoku_8_1_3 + Sudoku_8_1_4
 + Sudoku_8_1_5 + Sudoku_8_1_6 + Sudoku_8_1_7 + Sudoku_8_1_8 + Sudoku_8_1_9
 = 1
_C272: Sudoku_8_2_1 + Sudoku_8_2_2 + Sudoku_8_2_3 + Sudoku_8_2_4
 + Sudoku_8_2_5 + Sudoku_8_2_6 + Sudoku_8_2_7 + Sudoku_8_2_8 + Sudoku_8_2_9
 = 1
_C273: Sudoku_8_3_1 + Sudoku_8_3_2 + Sudoku_8_3_3 + Sudoku_8_3_4
 + Sudoku_8_3_5 + Sudoku_8_3_6 + Sudoku_8_3_7 + Sudoku_8_3_8 + Sudoku_8_3_9
 = 1
_C274: Sudoku_8_4_1 + Sudoku_8_4_2 + Sudoku_8_4_3 + Sudoku_8_4_4
 + Sudoku_8_4_5 + Sudoku_8_4_6 + Sudoku_8_4_7 + Sudoku_8_4_8 + Sudoku_8_4_9
 = 1
_C275: Sudoku_8_5_1 + Sudoku_8_5_2 + Sudoku_8_5_3 + Sudoku_8_5_4
 + Sudoku_8_5_5 + Sudoku_8_5_6 + Sudoku_8_5_7 + Sudoku_8_5_8 + Sudoku_8_5_9
 = 1
_C276: Sudoku_8_6_1 + Sudoku_8_6_2 + Sudoku_8_6_3 + Sudoku_8_6_4
 + Sudoku_8_6_5 + Sudoku_8_6_6 + Sudoku_8_6_7 + Sudoku_8_6_8 + Sudoku_8_6_9
 = 1
_C277: Sudoku_8_7_1 + Sudoku_8_7_2 + Sudoku_8_7_3 + Sudoku_8_7_4
 + Sudoku_8_7_5 + Sudoku_8_7_6 + Sudoku_8_7_7 + Sudoku_8_7_8 + Sudoku_8_7_9
 = 1
_C278: Sudoku_8_8_1 + Sudoku_8_8_2 + Sudoku_8_8_3 + Sudoku_8_8_4
 + Sudoku_8_8_5 + Sudoku_8_8_6 + Sudoku_8_8_7 + Sudoku_8_8_8 + Sudoku_8_8_9
 = 1
_C279: Sudoku_8_9_1 + Sudoku_8_9_2 + Sudoku_8_9_3 + Sudoku_8_9_4
 + Sudoku_8_9_5 + Sudoku_8_9_6 + Sudoku_8_9_7 + Sudoku_8_9_8 + Sudoku_8_9_9
 = 1
_C28: Sudoku_1_4_1 + Sudoku_2_4_1 + Sudoku_3_4_1 + Sudoku_4_4_1 + Sudoku_5_4_1
 + Sudoku_6_4_1 + Sudoku_7_4_1 + Sudoku_8_4_1 + Sudoku_9_4_1 = 1
_C280: Sudoku_8_1_1 + Sudoku_8_2_1 + Sudoku_8_3_1 + Sudoku_8_4_1
 + Sudoku_8_5_1 + Sudoku_8_6_1 + Sudoku_8_7_1 + Sudoku_8_8_1 + Sudoku_8_9_1
 = 1
_C281: Sudoku_8_1_2 + Sudoku_8_2_2 + Sudoku_8_3_2 + Sudoku_8_4_2
 + Sudoku_8_5_2 + Sudoku_8_6_2 + Sudoku_8_7_2 + Sudoku_8_8_2 + Sudoku_8_9_2
 = 1
_C282: Sudoku_8_1_3 + Sudoku_8_2_3 + Sudoku_8_3_3 + Sudoku_8_4_3
 + Sudoku_8_5_3 + Sudoku_8_6_3 + Sudoku_8_7_3 + Sudoku_8_8_3 + Sudoku_8_9_3
 = 1
_C283: Sudoku_8_1_4 + Sudoku_8_2_4 + Sudoku_8_3_4 + Sudoku_8_4_4
 + Sudoku_8_5_4 + Sudoku_8_6_4 + Sudoku_8_7_4 + Sudoku_8_8_4 + Sudoku_8_9_4
 = 1
_C284: Sudoku_8_1_5 + Sudoku_8_2_5 + Sudoku_8_3_5 + Sudoku_8_4_5
 + Sudoku_8_5_5 + Sudoku_8_6_5 + Sudoku_8_7_5 + Sudoku_8_8_5 + Sudoku_8_9_5
 = 1
_C285: Sudoku_8_1_6 + Sudoku_8_2_6 + Sudoku_8_3_6 + Sudoku_8_4_6
 + Sudoku_8_5_6 + Sudoku_8_6_6 + Sudoku_8_7_6 + Sudoku_8_8_6 + Sudoku_8_9_6
 = 1
_C286: Sudoku_8_1_7 + Sudoku_8_2_7 + Sudoku_8_3_7 + Sudoku_8_4_7
 + Sudoku_8_5_7 + Sudoku_8_6_7 + Sudoku_8_7_7 + Sudoku_8_8_7 + Sudoku_8_9_7
 = 1
_C287: Sudoku_8_1_8 + Sudoku_8_2_8 + Sudoku_8_3_8 + Sudoku_8_4_8
 + Sudoku_8_5_8 + Sudoku_8_6_8 + Sudoku_8_7_8 + Sudoku_8_8_8 + Sudoku_8_9_8
 = 1
_C288: Sudoku_8_1_9 + Sudoku_8_2_9 + Sudoku_8_3_9 + Sudoku_8_4_9
 + Sudoku_8_5_9 + Sudoku_8_6_9 + Sudoku_8_7_9 + Sudoku_8_8_9 + Sudoku_8_9_9
 = 1
_C289: Sudoku_8_1_1 + Sudoku_8_1_2 + Sudoku_8_1_3 + Sudoku_8_2_1
 + Sudoku_8_2_2 + Sudoku_8_2_3 + Sudoku_8_3_1 + Sudoku_8_3_2 + Sudoku_8_3_3
 = 1
_C29: Sudoku_1_4_2 + Sudoku_2_4_2 + Sudoku_3_4_2 + Sudoku_4_4_2 + Sudoku_5_4_2
 + Sudoku_6_4_2 + Sudoku_7_4_2 + Sudoku_8_4_2 + Sudoku_9_4_2 = 1
_C290: Sudoku_8_1_4 + Sudoku_8_1_5 + Sudoku_8_1_6 + Sudoku_8_2_4
 + Sudoku_8_2_5 + Sudoku_8_2_6 + Sudoku_8_3_4 + Sudoku_8_3_5 + Sudoku_8_3_6
 = 1
_C291: Sudoku_8_1_7 + Sudoku_8_1_8 + Sudoku_8_1_9 + Sudoku_8_2_7
 + Sudoku_8_2_8 + Sudoku_8_2_9 + Sudoku_8_3_7 + Sudoku_8_3_8 + Sudoku_8_3_9
 = 1
_C292: Sudoku_8_4_1 + Sudoku_8_4_2 + Sudoku_8_4_3 + Sudoku_8_5_1
 + Sudoku_8_5_2 + Sudoku_8_5_3 + Sudoku_8_6_1 + Sudoku_8_6_2 + Sudoku_8_6_3
 = 1
_C293: Sudoku_8_4_4 + Sudoku_8_4_5 + Sudoku_8_4_6 + Sudoku_8_5_4
 + Sudoku_8_5_5 + Sudoku_8_5_6 + Sudoku_8_6_4 + Sudoku_8_6_5 + Sudoku_8_6_6
 = 1
_C294: Sudoku_8_4_7 + Sudoku_8_4_8 + Sudoku_8_4_9 + Sudoku_8_5_7
 + Sudoku_8_5_8 + Sudoku_8_5_9 + Sudoku_8_6_7 + Sudoku_8_6_8 + Sudoku_8_6_9
 = 1
_C295: Sudoku_8_7_1 + Sudoku_8_7_2 + Sudoku_8_7_3 + Sudoku_8_8_1
 + Sudoku_8_8_2 + Sudoku_8_8_3 + Sudoku_8_9_1 + Sudoku_8_9_2 + Sudoku_8_9_3
 = 1
_C296: Sudoku_8_7_4 + Sudoku_8_7_5 + Sudoku_8_7_6 + Sudoku_8_8_4
 + Sudoku_8_8_5 + Sudoku_8_8_6 + Sudoku_8_9_4 + Sudoku_8_9_5 + Sudoku_8_9_6
 = 1
_C297: Sudoku_8_7_7 + Sudoku_8_7_8 + Sudoku_8_7_9 + Sudoku_8_8_7
 + Sudoku_8_8_8 + Sudoku_8_8_9 + Sudoku_8_9_7 + Sudoku_8_9_8 + Sudoku_8_9_9
 = 1
_C298: Sudoku_9_1_1 + Sudoku_9_1_2 + Sudoku_9_1_3 + Sudoku_9_1_4
 + Sudoku_9_1_5 + Sudoku_9_1_6 + Sudoku_9_1_7 + Sudoku_9_1_8 + Sudoku_9_1_9
 = 1
_C299: Sudoku_9_2_1 + Sudoku_9_2_2 + Sudoku_9_2_3 + Sudoku_9_2_4
 + Sudoku_9_2_5 + Sudoku_9_2_6 + Sudoku_9_2_7 + Sudoku_9_2_8 + Sudoku_9_2_9
 = 1
_C3: Sudoku_1_1_3 + Sudoku_2_1_3 + Sudoku_3_1_3 + Sudoku_4_1_3 + Sudoku_5_1_3
 + Sudoku_6_1_3 + Sudoku_7_1_3 + Sudoku_8_1_3 + Sudoku_9_1_3 = 1
_C30: Sudoku_1_4_3 + Sudoku_2_4_3 + Sudoku_3_4_3 + Sudoku_4_4_3 + Sudoku_5_4_3
 + Sudoku_6_4_3 + Sudoku_7_4_3 + Sudoku_8_4_3 + Sudoku_9_4_3 = 1
_C300: Sudoku_9_3_1 + Sudoku_9_3_2 + Sudoku_9_3_3 + Sudoku_9_3_4
 + Sudoku_9_3_5 + Sudoku_9_3_6 + Sudoku_9_3_7 + Sudoku_9_3_8 + Sudoku_9_3_9
 = 1
_C301: Sudoku_9_4_1 + Sudoku_9_4_2 + Sudoku_9_4_3 + Sudoku_9_4_4
 + Sudoku_9_4_5 + Sudoku_9_4_6 + Sudoku_9_4_7 + Sudoku_9_4_8 + Sudoku_9_4_9
 = 1
_C302: Sudoku_9_5_1 + Sudoku_9_5_2 + Sudoku_9_5_3 + Sudoku_9_5_4
 + Sudoku_9_5_5 + Sudoku_9_5_6 + Sudoku_9_5_7 + Sudoku_9_5_8 + Sudoku_9_5_9
 = 1
_C303: Sudoku_9_6_1 + Sudoku_9_6_2 + Sudoku_9_6_3 + Sudoku_9_6_4
 + Sudoku_9_6_5 + Sudoku_9_6_6 + Sudoku_9_6_7 + Sudoku_9_6_8 + Sudoku_9_6_9
 = 1
_C304: Sudoku_9_7_1 + Sudoku_9_7_2 + Sudoku_9_7_3 + Sudoku_9_7_4
 + Sudoku_9_7_5 + Sudoku_9_7_6 + Sudoku_9_7_7 + Sudoku_9_7_8 + Sudoku_9_7_9
 = 1
_C305: Sudoku_9_8_1 + Sudoku_9_8_2 + Sudoku_9_8_3 + Sudoku_9_8_4
 + Sudoku_9_8_5 + Sudoku_9_8_6 + Sudoku_9_8_7 + Sudoku_9_8_8 + Sudoku_9_8_9
 = 1
_C306: Sudoku_9_9_1 + Sudoku_9_9_2 + Sudoku_9_9_3 + Sudoku_9_9_4
 + Sudoku_9_9_5 + Sudoku_9_9_6 + Sudoku_9_9_7 + Sudoku_9_9_8 + Sudoku_9_9_9
 = 1
_C307: Sudoku_9_1_1 + Sudoku_9_2_1 + Sudoku_9_3_1 + Sudoku_9_4_1
 + Sudoku_9_5_1 + Sudoku_9_6_1 + Sudoku_9_7_1 + Sudoku_9_8_1 + Sudoku_9_9_1
 = 1
_C308: Sudoku_9_1_2 + Sudoku_9_2_2 + Sudoku_9_3_2 + Sudoku_9_4_2
 + Sudoku_9_5_2 + Sudoku_9_6_2 + Sudoku_9_7_2 + Sudoku_9_8_2 + Sudoku_9_9_2
 = 1
_C309: Sudoku_9_1_3 + Sudoku_9_2_3 + Sudoku_9_3_3 + Sudoku_9_4_3
 + Sudoku_9_5_3 + Sudoku_9_6_3 + Sudoku_9_7_3 + Sudoku_9_8_3 + Sudoku_9_9_3
 = 1
_C31: Sudoku_1_4_4 + Sudoku_2_4_4 + Sudoku_3_4_4 + Sudoku_4_4_4 + Sudoku_5_4_4
 + Sudoku_6_4_4 + Sudoku_7_4_4 + Sudoku_8_4_4 + Sudoku_9_4_4 = 1
_C310: Sudoku_9_1_4 + Sudoku_9_2_4 + Sudoku_9_3_4 + Sudoku_9_4_4
 + Sudoku_9_5_4 + Sudoku_9_6_4 + Sudoku_9_7_4 + Sudoku_9_8_4 + Sudoku_9_9_4
 = 1
_C311: Sudoku_9_1_5 + Sudoku_9_2_5 + Sudoku_9_3_5 + Sudoku_9_4_5
 + Sudoku_9_5_5 + Sudoku_9_6_5 + Sudoku_9_7_5 + Sudoku_9_8_5 + Sudoku_9_9_5
 = 1
_C312: Sudoku_9_1_6 + Sudoku_9_2_6 + Sudoku_9_3_6 + Sudoku_9_4_6
 + Sudoku_9_5_6 + Sudoku_9_6_6 + Sudoku_9_7_6 + Sudoku_9_8_6 + Sudoku_9_9_6
 = 1
_C313: Sudoku_9_1_7 + Sudoku_9_2_7 + Sudoku_9_3_7 + Sudoku_9_4_7
 + Sudoku_9_5_7 + Sudoku_9_6_7 + Sudoku_9_7_7 + Sudoku_9_8_7 + Sudoku_9_9_7
 = 1
_C314: Sudoku_9_1_8 + Sudoku_9_2_8 + Sudoku_9_3_8 + Sudoku_9_4_8
 + Sudoku_9_5_8 + Sudoku_9_6_8 + Sudoku_9_7_8 + Sudoku_9_8_8 + Sudoku_9_9_8
 = 1
_C315: Sudoku_9_1_9 + Sudoku_9_2_9 + Sudoku_9_3_9 + Sudoku_9_4_9
 + Sudoku_9_5_9 + Sudoku_9_6_9 + Sudoku_9_7_9 + Sudoku_9_8_9 + Sudoku_9_9_9
 = 1
_C316: Sudoku_9_1_1 + Sudoku_9_1_2 + Sudoku_9_1_3 + Sudoku_9_2_1
 + Sudoku_9_2_2 + Sudoku_9_2_3 + Sudoku_9_3_1 + Sudoku_9_3_2 + Sudoku_9_3_3
 = 1
_C317: Sudoku_9_1_4 + Sudoku_9_1_5 + Sudoku_9_1_6 + Sudoku_9_2_4
 + Sudoku_9_2_5 + Sudoku_9_2_6 + Sudoku_9_3_4 + Sudoku_9_3_5 + Sudoku_9_3_6
 = 1
_C318: Sudoku_9_1_7 + Sudoku_9_1_8 + Sudoku_9_1_9 + Sudoku_9_2_7
 + Sudoku_9_2_8 + Sudoku_9_2_9 + Sudoku_9_3_7 + Sudoku_9_3_8 + Sudoku_9_3_9
 = 1
_C319: Sudoku_9_4_1 + Sudoku_9_4_2 + Sudoku_9_4_3 + Sudoku_9_5_1
 + Sudoku_9_5_2 + Sudoku_9_5_3 + Sudoku_9_6_1 + Sudoku_9_6_2 + Sudoku_9_6_3
 = 1
_C32: Sudoku_1_4_5 + Sudoku_2_4_5 + Sudoku_3_4_5 + Sudoku_4_4_5 + Sudoku_5_4_5
 + Sudoku_6_4_5 + Sudoku_7_4_5 + Sudoku_8_4_5 + Sudoku_9_4_5 = 1
_C320: Sudoku_9_4_4 + Sudoku_9_4_5 + Sudoku_9_4_6 + Sudoku_9_5_4
 + Sudoku_9_5_5 + Sudoku_9_5_6 + Sudoku_9_6_4 + Sudoku_9_6_5 + Sudoku_9_6_6
 = 1
_C321: Sudoku_9_4_7 + Sudoku_9_4_8 + Sudoku_9_4_9 + Sudoku_9_5_7
 + Sudoku_9_5_8 + Sudoku_9_5_9 + Sudoku_9_6_7 + Sudoku_9_6_8 + Sudoku_9_6_9
 = 1
_C322: Sudoku_9_7_1 + Sudoku_9_7_2 + Sudoku_9_7_3 + Sudoku_9_8_1
 + Sudoku_9_8_2 + Sudoku_9_8_3 + Sudoku_9_9_1 + Sudoku_9_9_2 + Sudoku_9_9_3
 = 1
_C323: Sudoku_9_7_4 + Sudoku_9_7_5 + Sudoku_9_7_6 + Sudoku_9_8_4
 + Sudoku_9_8_5 + Sudoku_9_8_6 + Sudoku_9_9_4 + Sudoku_9_9_5 + Sudoku_9_9_6
 = 1
_C324: Sudoku_9_7_7 + Sudoku_9_7_8 + Sudoku_9_7_9 + Sudoku_9_8_7
 + Sudoku_9_8_8 + Sudoku_9_8_9 + Sudoku_9_9_7 + Sudoku_9_9_8 + Sudoku_9_9_9
 = 1
_C325: Sudoku_9_1_3 = 1
_C326: Sudoku_7_1_7 = 1
_C327: Sudoku_8_1_8 = 1
_C328: Sudoku_8_2_1 = 1
_C329: Sudoku_3_2_2 = 1
_C33: Sudoku_1_4_6 + Sudoku_2_4_6 + Sudoku_3_4_6 + Sudoku_4_4_6 + Sudoku_5_4_6
 + Sudoku_6_4_6 + Sudoku_7_4_6 + Sudoku_8_4_6 + Sudoku_9_4_6 = 1
_C330: Sudoku_1_2_5 = 1
_C331: Sudoku_9_2_6 = 1
_C332: Sudoku_6_3_1 = 1
_C333: Sudoku_1_3_2 = 1
_C334: Sudoku_4_3_7 = 1
_C335: Sudoku_3_3_9 = 1
_C336: Sudoku_1_4_3 = 1
_C337: Sudoku_9_4_4 = 1
_C338: Sudoku_2_4_8 = 1
_C339: Sudoku_7_4_9 = 1
_C34: Sudoku_1_4_7 + Sudoku_2_4_7 + Sudoku_3_4_7 + Sudoku_4_4_7 + Sudoku_5_4_7
 + Sudoku_6_4_7 + Sudoku_7_4_7 + Sudoku_8_4_7 + Sudoku_9_4_7 = 1
_C340: Sudoku_4_5_5 = 1
_C341: Sudoku_5_6_1 = 1
_C342: Sudoku_9_6_2 = 1
_C343: Sudoku_8_6_6 = 1
_C344: Sudoku_3_6_7 = 1
_C345: Sudoku_9_7_1 = 1
_C346: Sudoku_5_7_3 = 1
_C347: Sudoku_7_7_8 = 1
_C348: Sudoku_2_7_9 = 1
_C349: Sudoku_5_8_4 = 1
_C35: Sudoku_1_4_8 + Sudoku_2_4_8 + Sudoku_3_4_8 + Sudoku_4_4_8 + Sudoku_5_4_8
 + Sudoku_6_4_8 + Sudoku_7_4_8 + Sudoku_8_4_8 + Sudoku_9_4_8 = 1
_C350: Sudoku_9_8_5 = 1
_C351: Sudoku_4_8_8 = 1
_C352: Sudoku_8_8_9 = 1
_C353: Sudoku_8_9_2 = 1
_C354: Sudoku_2_9_3 = 1
_C355: Sudoku_9_9_7 = 1
_C36: Sudoku_1_4_9 + Sudoku_2_4_9 + Sudoku_3_4_9 + Sudoku_4_4_9 + Sudoku_5_4_9
 + Sudoku_6_4_9 + Sudoku_7_4_9 + Sudoku_8_4_9 + Sudoku_9_4_9 = 1
_C37: Sudoku_1_5_1 + Sudoku_2_5_1 + Sudoku_3_5_1 + Sudoku_4_5_1 + Sudoku_5_5_1
 + Sudoku_6_5_1 + Sudoku_7_5_1 + Sudoku_8_5_1 + Sudoku_9_5_1 = 1
_C38: Sudoku_1_5_2 + Sudoku_2_5_2 + Sudoku_3_5_2 + Sudoku_4_5_2 + Sudoku_5_5_2
 + Sudoku_6_5_2 + Sudoku_7_5_2 + Sudoku_8_5_2 + Sudoku_9_5_2 = 1
_C39: Sudoku_1_5_3 + Sudoku_2_5_3 + Sudoku_3_5_3 + Sudoku_4_5_3 + Sudoku_5_5_3
 + Sudoku_6_5_3 + Sudoku_7_5_3 + Sudoku_8_5_3 + Sudoku_9_5_3 = 1
_C4: Sudoku_1_1_4 + Sudoku_2_1_4 + Sudoku_3_1_4 + Sudoku_4_1_4 + Sudoku_5_1_4
 + Sudoku_6_1_4 + Sudoku_7_1_4 + Sudoku_8_1_4 + Sudoku_9_1_4 = 1
_C40: Sudoku_1_5_4 + Sudoku_2_5_4 + Sudoku_3_5_4 + Sudoku_4_5_4 + Sudoku_5_5_4
 + Sudoku_6_5_4 + Sudoku_7_5_4 + Sudoku_8_5_4 + Sudoku_9_5_4 = 1
_C41: Sudoku_1_5_5 + Sudoku_2_5_5 + Sudoku_3_5_5 + Sudoku_4_5_5 + Sudoku_5_5_5
 + Sudoku_6_5_5 + Sudoku_7_5_5 + Sudoku_8_5_5 + Sudoku_9_5_5 = 1
_C42: Sudoku_1_5_6 + Sudoku_2_5_6 + Sudoku_3_5_6 + Sudoku_4_5_6 + Sudoku_5_5_6
 + Sudoku_6_5_6 + Sudoku_7_5_6 + Sudoku_8_5_6 + Sudoku_9_5_6 = 1
_C43: Sudoku_1_5_7 + Sudoku_2_5_7 + Sudoku_3_5_7 + Sudoku_4_5_7 + Sudoku_5_5_7
 + Sudoku_6_5_7 + Sudoku_7_5_7 + Sudoku_8_5_7 + Sudoku_9_5_7 = 1
_C44: Sudoku_1_5_8 + Sudoku_2_5_8 + Sudoku_3_5_8 + Sudoku_4_5_8 + Sudoku_5_5_8
 + Sudoku_6_5_8 + Sudoku_7_5_8 + Sudoku_8_5_8 + Sudoku_9_5_8 = 1
_C45: Sudoku_1_5_9 + Sudoku_2_5_9 + Sudoku_3_5_9 + Sudoku_4_5_9 + Sudoku_5_5_9
 + Sudoku_6_5_9 + Sudoku_7_5_9 + Sudoku_8_5_9 + Sudoku_9_5_9 = 1
_C46: Sudoku_1_6_1 + Sudoku_2_6_1 + Sudoku_3_6_1 + Sudoku_4_6_1 + Sudoku_5_6_1
 + Sudoku_6_6_1 + Sudoku_7_6_1 + Sudoku_8_6_1 + Sudoku_9_6_1 = 1
_C47: Sudoku_1_6_2 + Sudoku_2_6_2 + Sudoku_3_6_2 + Sudoku_4_6_2 + Sudoku_5_6_2
 + Sudoku_6_6_2 + Sudoku_7_6_2 + Sudoku_8_6_2 + Sudoku_9_6_2 = 1
_C48: Sudoku_1_6_3 + Sudoku_2_6_3 + Sudoku_3_6_3 + Sudoku_4_6_3 + Sudoku_5_6_3
 + Sudoku_6_6_3 + Sudoku_7_6_3 + Sudoku_8_6_3 + Sudoku_9_6_3 = 1
_C49: Sudoku_1_6_4 + Sudoku_2_6_4 + Sudoku_3_6_4 + Sudoku_4_6_4 + Sudoku_5_6_4
 + Sudoku_6_6_4 + Sudoku_7_6_4 + Sudoku_8_6_4 + Sudoku_9_6_4 = 1
_C5: Sudoku_1_1_5 + Sudoku_2_1_5 + Sudoku_3_1_5 + Sudoku_4_1_5 + Sudoku_5_1_5
 + Sudoku_6_1_5 + Sudoku_7_1_5 + Sudoku_8_1_5 + Sudoku_9_1_5 = 1
_C50: Sudoku_1_6_5 + Sudoku_2_6_5 + Sudoku_3_6_5 + Sudoku_4_6_5 + Sudoku_5_6_5
 + Sudoku_6_6_5 + Sudoku_7_6_5 + Sudoku_8_6_5 + Sudoku_9_6_5 = 1
_C51: Sudoku_1_6_6 + Sudoku_2_6_6 + Sudoku_3_6_6 + Sudoku_4_6_6 + Sudoku_5_6_6
 + Sudoku_6_6_6 + Sudoku_7_6_6 + Sudoku_8_6_6 + Sudoku_9_6_6 = 1
_C52: Sudoku_1_6_7 + Sudoku_2_6_7 + Sudoku_3_6_7 + Sudoku_4_6_7 + Sudoku_5_6_7
 + Sudoku_6_6_7 + Sudoku_7_6_7 + Sudoku_8_6_7 + Sudoku_9_6_7 = 1
_C53: Sudoku_1_6_8 + Sudoku_2_6_8 + Sudoku_3_6_8 + Sudoku_4_6_8 + Sudoku_5_6_8
 + Sudoku_6_6_8 + Sudoku_7_6_8 + Sudoku_8_6_8 + Sudoku_9_6_8 = 1
_C54: Sudoku_1_6_9 + Sudoku_2_6_9 + Sudoku_3_6_9 + Sudoku_4_6_9 + Sudoku_5_6_9
 + Sudoku_6_6_9 + Sudoku_7_6_9 + Sudoku_8_6_9 + Sudoku_9_6_9 = 1
_C55: Sudoku_1_7_1 + Sudoku_2_7_1 + Sudoku_3_7_1 + Sudoku_4_7_1 + Sudoku_5_7_1
 + Sudoku_6_7_1 + Sudoku_7_7_1 + Sudoku_8_7_1 + Sudoku_9_7_1 = 1
_C56: Sudoku_1_7_2 + Sudoku_2_7_2 + Sudoku_3_7_2 + Sudoku_4_7_2 + Sudoku_5_7_2
 + Sudoku_6_7_2 + Sudoku_7_7_2 + Sudoku_8_7_2 + Sudoku_9_7_2 = 1
_C57: Sudoku_1_7_3 + Sudoku_2_7_3 + Sudoku_3_7_3 + Sudoku_4_7_3 + Sudoku_5_7_3
 + Sudoku_6_7_3 + Sudoku_7_7_3 + Sudoku_8_7_3 + Sudoku_9_7_3 = 1
_C58: Sudoku_1_7_4 + Sudoku_2_7_4 + Sudoku_3_7_4 + Sudoku_4_7_4 + Sudoku_5_7_4
 + Sudoku_6_7_4 + Sudoku_7_7_4 + Sudoku_8_7_4 + Sudoku_9_7_4 = 1
_C59: Sudoku_1_7_5 + Sudoku_2_7_5 + Sudoku_3_7_5 + Sudoku_4_7_5 + Sudoku_5_7_5
 + Sudoku_6_7_5 + Sudoku_7_7_5 + Sudoku_8_7_5 + Sudoku_9_7_5 = 1
_C6: Sudoku_1_1_6 + Sudoku_2_1_6 + Sudoku_3_1_6 + Sudoku_4_1_6 + Sudoku_5_1_6
 + Sudoku_6_1_6 + Sudoku_7_1_6 + Sudoku_8_1_6 + Sudoku_9_1_6 = 1
_C60: Sudoku_1_7_6 + Sudoku_2_7_6 + Sudoku_3_7_6 + Sudoku_4_7_6 + Sudoku_5_7_6
 + Sudoku_6_7_6 + Sudoku_7_7_6 + Sudoku_8_7_6 + Sudoku_9_7_6 = 1
_C61: Sudoku_1_7_7 + Sudoku_2_7_7 + Sudoku_3_7_7 + Sudoku_4_7_7 + Sudoku_5_7_7
 + Sudoku_6_7_7 + Sudoku_7_7_7 + Sudoku_8_7_7 + Sudoku_9_7_7 = 1
_C62: Sudoku_1_7_8 + Sudoku_2_7_8 + Sudoku_3_7_8 + Sudoku_4_7_8 + Sudoku_5_7_8
 + Sudoku_6_7_8 + Sudoku_7_7_8 + Sudoku_8_7_8 + Sudoku_9_7_8 = 1
_C63: Sudoku_1_7_9 + Sudoku_2_7_9 + Sudoku_3_7_9 + Sudoku_4_7_9 + Sudoku_5_7_9
 + Sudoku_6_7_9 + Sudoku_7_7_9 + Sudoku_8_7_9 + Sudoku_9_7_9 = 1
_C64: Sudoku_1_8_1 + Sudoku_2_8_1 + Sudoku_3_8_1 + Sudoku_4_8_1 + Sudoku_5_8_1
 + Sudoku_6_8_1 + Sudoku_7_8_1 + Sudoku_8_8_1 + Sudoku_9_8_1 = 1
_C65: Sudoku_1_8_2 + Sudoku_2_8_2 + Sudoku_3_8_2 + Sudoku_4_8_2 + Sudoku_5_8_2
 + Sudoku_6_8_2 + Sudoku_7_8_2 + Sudoku_8_8_2 + Sudoku_9_8_2 = 1
_C66: Sudoku_1_8_3 + Sudoku_2_8_3 + Sudoku_3_8_3 + Sudoku_4_8_3 + Sudoku_5_8_3
 + Sudoku_6_8_3 + Sudoku_7_8_3 + Sudoku_8_8_3 + Sudoku_9_8_3 = 1
_C67: Sudoku_1_8_4 + Sudoku_2_8_4 + Sudoku_3_8_4 + Sudoku_4_8_4 + Sudoku_5_8_4
 + Sudoku_6_8_4 + Sudoku_7_8_4 + Sudoku_8_8_4 + Sudoku_9_8_4 = 1
_C68: Sudoku_1_8_5 + Sudoku_2_8_5 + Sudoku_3_8_5 + Sudoku_4_8_5 + Sudoku_5_8_5
 + Sudoku_6_8_5 + Sudoku_7_8_5 + Sudoku_8_8_5 + Sudoku_9_8_5 = 1
_C69: Sudoku_1_8_6 + Sudoku_2_8_6 + Sudoku_3_8_6 + Sudoku_4_8_6 + Sudoku_5_8_6
 + Sudoku_6_8_6 + Sudoku_7_8_6 + Sudoku_8_8_6 + Sudoku_9_8_6 = 1
_C7: Sudoku_1_1_7 + Sudoku_2_1_7 + Sudoku_3_1_7 + Sudoku_4_1_7 + Sudoku_5_1_7
 + Sudoku_6_1_7 + Sudoku_7_1_7 + Sudoku_8_1_7 + Sudoku_9_1_7 = 1
_C70: Sudoku_1_8_7 + Sudoku_2_8_7 + Sudoku_3_8_7 + Sudoku_4_8_7 + Sudoku_5_8_7
 + Sudoku_6_8_7 + Sudoku_7_8_7 + Sudoku_8_8_7 + Sudoku_9_8_7 = 1
_C71: Sudoku_1_8_8 + Sudoku_2_8_8 + Sudoku_3_8_8 + Sudoku_4_8_8 + Sudoku_5_8_8
 + Sudoku_6_8_8 + Sudoku_7_8_8 + Sudoku_8_8_8 + Sudoku_9_8_8 = 1
_C72: Sudoku_1_8_9 + Sudoku_2_8_9 + Sudoku_3_8_9 + Sudoku_4_8_9 + Sudoku_5_8_9
 + Sudoku_6_8_9 + Sudoku_7_8_9 + Sudoku_8_8_9 + Sudoku_9_8_9 = 1
_C73: Sudoku_1_9_1 + Sudoku_2_9_1 + Sudoku_3_9_1 + Sudoku_4_9_1 + Sudoku_5_9_1
 + Sudoku_6_9_1 + Sudoku_7_9_1 + Sudoku_8_9_1 + Sudoku_9_9_1 = 1
_C74: Sudoku_1_9_2 + Sudoku_2_9_2 + Sudoku_3_9_2 + Sudoku_4_9_2 + Sudoku_5_9_2
 + Sudoku_6_9_2 + Sudoku_7_9_2 + Sudoku_8_9_2 + Sudoku_9_9_2 = 1
_C75: Sudoku_1_9_3 + Sudoku_2_9_3 + Sudoku_3_9_3 + Sudoku_4_9_3 + Sudoku_5_9_3
 + Sudoku_6_9_3 + Sudoku_7_9_3 + Sudoku_8_9_3 + Sudoku_9_9_3 = 1
_C76: Sudoku_1_9_4 + Sudoku_2_9_4 + Sudoku_3_9_4 + Sudoku_4_9_4 + Sudoku_5_9_4
 + Sudoku_6_9_4 + Sudoku_7_9_4 + Sudoku_8_9_4 + Sudoku_9_9_4 = 1
_C77: Sudoku_1_9_5 + Sudoku_2_9_5 + Sudoku_3_9_5 + Sudoku_4_9_5 + Sudoku_5_9_5
 + Sudoku_6_9_5 + Sudoku_7_9_5 + Sudoku_8_9_5 + Sudoku_9_9_5 = 1
_C78: Sudoku_1_9_6 + Sudoku_2_9_6 + Sudoku_3_9_6 + Sudoku_4_9_6 + Sudoku_5_9_6
 + Sudoku_6_9_6 + Sudoku_7_9_6 + Sudoku_8_9_6 + Sudoku_9_9_6 = 1
_C79: Sudoku_1_9_7 + Sudoku_2_9_7 + Sudoku_3_9_7 + Sudoku_4_9_7 + Sudoku_5_9_7
 + Sudoku_6_9_7 + Sudoku_7_9_7 + Sudoku_8_9_7 + Sudoku_9_9_7 = 1
_C8: Sudoku_1_1_8 + Sudoku_2_1_8 + Sudoku_3_1_8 + Sudoku_4_1_8 + Sudoku_5_1_8
 + Sudoku_6_1_8 + Sudoku_7_1_8 + Sudoku_8_1_8 + Sudoku_9_1_8 = 1
_C80: Sudoku_1_9_8 + Sudoku_2_9_8 + Sudoku_3_9_8 + Sudoku_4_9_8 + Sudoku_5_9_8
 + Sudoku_6_9_8 + Sudoku_7_9_8 + Sudoku_8_9_8 + Sudoku_9_9_8 = 1
_C81: Sudoku_1_9_9 + Sudoku_2_9_9 + Sudoku_3_9_9 + Sudoku_4_9_9 + Sudoku_5_9_9
 + Sudoku_6_9_9 + Sudoku_7_9_9 + Sudoku_8_9_9 + Sudoku_9_9_9 = 1
_C82: Sudoku_1_1_1 + Sudoku_1_1_2 + Sudoku_1_1_3 + Sudoku_1_1_4 + Sudoku_1_1_5
 + Sudoku_1_1_6 + Sudoku_1_1_7 + Sudoku_1_1_8 + Sudoku_1_1_9 = 1
_C83: Sudoku_1_2_1 + Sudoku_1_2_2 + Sudoku_1_2_3 + Sudoku_1_2_4 + Sudoku_1_2_5
 + Sudoku_1_2_6 + Sudoku_1_2_7 + Sudoku_1_2_8 + Sudoku_1_2_9 = 1
_C84: Sudoku_1_3_1 + Sudoku_1_3_2 + Sudoku_1_3_3 + Sudoku_1_3_4 + Sudoku_1_3_5
 + Sudoku_1_3_6 + Sudoku_1_3_7 + Sudoku_1_3_8 + Sudoku_1_3_9 = 1
_C85: Sudoku_1_4_1 + Sudoku_1_4_2 + Sudoku_1_4_3 + Sudoku_1_4_4 + Sudoku_1_4_5
 + Sudoku_1_4_6 + Sudoku_1_4_7 + Sudoku_1_4_8 + Sudoku_1_4_9 = 1
_C86: Sudoku_1_5_1 + Sudoku_1_5_2 + Sudoku_1_5_3 + Sudoku_1_5_4 + Sudoku_1_5_5
 + Sudoku_1_5_6 + Sudoku_1_5_7 + Sudoku_1_5_8 + Sudoku_1_5_9 = 1
_C87: Sudoku_1_6_1 + Sudoku_1_6_2 + Sudoku_1_6_3 + Sudoku_1_6_4 + Sudoku_1_6_5
 + Sudoku_1_6_6 + Sudoku_1_6_7 + Sudoku_1_6_8 + Sudoku_1_6_9 = 1
_C88: Sudoku_1_7_1 + Sudoku_1_7_2 + Sudoku_1_7_3 + Sudoku_1_7_4 + Sudoku_1_7_5
 + Sudoku_1_7_6 + Sudoku_1_7_7 + Sudoku_1_7_8 + Sudoku_1_7_9 = 1
_C89: Sudoku_1_8_1 + Sudoku_1_8_2 + Sudoku_1_8_3 + Sudoku_1_8_4 + Sudoku_1_8_5
 + Sudoku_1_8_6 + Sudoku_1_8_7 + Sudoku_1_8_8 + Sudoku_1_8_9 = 1
_C9: Sudoku_1_1_9 + Sudoku_2_1_9 + Sudoku_3_1_9 + Sudoku_4_1_9 + Sudoku_5_1_9
 + Sudoku_6_1_9 + Sudoku_7_1_9 + Sudoku_8_1_9 + Sudoku_9_1_9 = 1
_C90: Sudoku_1_9_1 + Sudoku_1_9_2 + Sudoku_1_9_3 + Sudoku_1_9_4 + Sudoku_1_9_5
 + Sudoku_1_9_6 + Sudoku_1_9_7 + Sudoku_1_9_8 + Sudoku_1_9_9 = 1
_C91: Sudoku_1_1_1 + Sudoku_1_2_1 + Sudoku_1_3_1 + Sudoku_1_4_1 + Sudoku_1_5_1
 + Sudoku_1_6_1 + Sudoku_1_7_1 + Sudoku_1_8_1 + Sudoku_1_9_1 = 1
_C92: Sudoku_1_1_2 + Sudoku_1_2_2 + Sudoku_1_3_2 + Sudoku_1_4_2 + Sudoku_1_5_2
 + Sudoku_1_6_2 + Sudoku_1_7_2 + Sudoku_1_8_2 + Sudoku_1_9_2 = 1
_C93: Sudoku_1_1_3 + Sudoku_1_2_3 + Sudoku_1_3_3 + Sudoku_1_4_3 + Sudoku_1_5_3
 + Sudoku_1_6_3 + Sudoku_1_7_3 + Sudoku_1_8_3 + Sudoku_1_9_3 = 1
_C94: Sudoku_1_1_4 + Sudoku_1_2_4 + Sudoku_1_3_4 + Sudoku_1_4_4 + Sudoku_1_5_4
 + Sudoku_1_6_4 + Sudoku_1_7_4 + Sudoku_1_8_4 + Sudoku_1_9_4 = 1
_C95: Sudoku_1_1_5 + Sudoku_1_2_5 + Sudoku_1_3_5 + Sudoku_1_4_5 + Sudoku_1_5_5
 + Sudoku_1_6_5 + Sudoku_1_7_5 + Sudoku_1_8_5 + Sudoku_1_9_5 = 1
_C96: Sudoku_1_1_6 + Sudoku_1_2_6 + Sudoku_1_3_6 + Sudoku_1_4_6 + Sudoku_1_5_6
 + Sudoku_1_6_6 + Sudoku_1_7_6 + Sudoku_1_8_6 + Sudoku_1_9_6 = 1
_C97: Sudoku_1_1_7 + Sudoku_1_2_7 + Sudoku_1_3_7 + Sudoku_1_4_7 + Sudoku_1_5_7
 + Sudoku_1_6_7 + Sudoku_1_7_7 + Sudoku_1_8_7 + Sudoku_1_9_7 = 1
_C98: Sudoku_1_1_8 + Sudoku_1_2_8 + Sudoku_1_3_8 + Sudoku_1_4_8 + Sudoku_1_5_8
 + Sudoku_1_6_8 + Sudoku_1_7_8 + Sudoku_1_8_8 + Sudoku_1_9_8 = 1
_C99: Sudoku_1_1_9 + Sudoku_1_2_9 + Sudoku_1_3_9 + Sudoku_1_4_9 + Sudoku_1_5_9
 + Sudoku_1_6_9 + Sudoku_1_7_9 + Sudoku_1_8_9 + Sudoku_1_9_9 = 1
Bounds
__dummy = 0
Binaries
Sudoku_1_1_1
Sudoku_1_1_2
Sudoku_1_1_3
Sudoku_1_1_4
Sudoku_1_1_5
Sudoku_1_1_6
Sudoku_1_1_7
Sudoku_1_1_8
Sudoku_1_1_9
Sudoku_1_2_1
Sudoku_1_2_2
Sudoku_1_2_3
Sudoku_1_2_4
Sudoku_1_2_5
Sudoku_1_2_6
Sudoku_1_2_7
Sudoku_1_2_8
Sudoku_1_2_9
Sudoku_1_3_1
Sudoku_1_3_2
Sudoku_1_3_3
Sudoku_1_3_4
Sudoku_1_3_5
Sudoku_1_3_6
Sudoku_1_3_7
Sudoku_1_3_8
Sudoku_1_3_9
Sudoku_1_4_1
Sudoku_1_4_2
Sudoku_1_4_3
Sudoku_1_4_4
Sudoku_1_4_5
Sudoku_1_4_6
Sudoku_1_4_7
Sudoku_1_4_8
Sudoku_1_4_9
Sudoku_1_5_1
Sudoku_1_5_2
Sudoku_1_5_3
Sudoku_1_5_4
Sudoku_1_5_5
Sudoku_1_5_6
Sudoku_1_5_7
Sudoku_1_5_8
Sudoku_1_5_9
Sudoku_1_6_1
Sudoku_1_6_2
Sudoku_1_6_3
Sudoku_1_6_4
Sudoku_1_6_5
Sudoku_1_6_6
Sudoku_1_6_7
Sudoku_1_6_8
Sudoku_1_6_9
Sudoku_1_7_1
Sudoku_1_7_2
Sudoku_1_7_3
Sudoku_1_7_4
Sudoku_1_7_5
Sudoku_1_7_6
Sudoku_1_7_7
Sudoku_1_7_8
Sudoku_1_7_9
Sudoku_1_8_1
Sudoku_1_8_2
Sudoku_1_8_3
Sudoku_1_8_4
Sudoku_1_8_5
Sudoku_1_8_6
Sudoku_1_8_7
Sudoku_1_8_8
Sudoku_1_8_9
Sudoku_1_9_1
Sudoku_1_9_2
Sudoku_1_9_3
Sudoku_1_9_4
Sudoku_1_9_5
Sudoku_1_9_6
Sudoku_1_9_7
Sudoku_1_9_8
Sudoku_1_9_9
Sudoku_2_1_1
Sudoku_2_1_2
Sudoku_2_1_3
Sudoku_2_1_4
Sudoku_2_1_5
Sudoku_2_1_6
Sudoku_2_1_7
Sudoku_2_1_8
Sudoku_2_1_9
Sudoku_2_2_1
Sudoku_2_2_2
Sudoku_2_2_3
Sudoku_2_2_4
Sudoku_2_2_5
Sudoku_2_2_6
Sudoku_2_2_7
Sudoku_2_2_8
Sudoku_2_2_9
Sudoku_2_3_1
Sudoku_2_3_2
Sudoku_2_3_3
Sudoku_2_3_4
Sudoku_2_3_5
Sudoku_2_3_6
Sudoku_2_3_7
Sudoku_2_3_8
Sudoku_2_3_9
Sudoku_2_4_1
Sudoku_2_4_2
Sudoku_2_4_3
Sudoku_2_4_4
Sudoku_2_4_5
Sudoku_2_4_6
Sudoku_2_4_7
Sudoku_2_4_8
Sudoku_2_4_9
Sudoku_2_5_1
Sudoku_2_5_2
Sudoku_2_5_3
Sudoku_2_5_4
Sudoku_2_5_5
Sudoku_2_5_6
Sudoku_2_5_7
Sudoku_2_5_8
Sudoku_2_5_9
Sudoku_2_6_1
Sudoku_2_6_2
Sudoku_2_6_3
Sudoku_2_6_4
Sudoku_2_6_5
Sudoku_2_6_6
Sudoku_2_6_7
Sudoku_2_6_8
Sudoku_2_6_9
Sudoku_2_7_1
Sudoku_2_7_2
Sudoku_2_7_3
Sudoku_2_7_4
Sudoku_2_7_5
Sudoku_2_7_6
Sudoku_2_7_7
Sudoku_2_7_8
Sudoku_2_7_9
Sudoku_2_8_1
Sudoku_2_8_2
Sudoku_2_8_3
Sudoku_2_8_4
Sudoku_2_8_5
Sudoku_2_8_6
Sudoku_2_8_7
Sudoku_2_8_8
Sudoku_2_8_9
Sudoku_2_9_1
Sudoku_2_9_2
Sudoku_2_9_3
Sudoku_2_9_4
Sudoku_2_9_5
Sudoku_2_9_6
Sudoku_2_9_7
Sudoku_2_9_8
Sudoku_2_9_9
Sudoku_3_1_1
Sudoku_3_1_2
Sudoku_3_1_3
Sudoku_3_1_4
Sudoku_3_1_5
Sudoku_3_1_6
Sudoku_3_1_7
Sudoku_3_1_8
Sudoku_3_1_9
Sudoku_3_2_1
Sudoku_3_2_2
Sudoku_3_2_3
Sudoku_3_2_4
Sudoku_3_2_5
Sudoku_3_2_6
Sudoku_3_2_7
Sudoku_3_2_8
Sudoku_3_2_9
Sudoku_3_3_1
Sudoku_3_3_2
Sudoku_3_3_3
Sudoku_3_3_4
Sudoku_3_3_5
Sudoku_3_3_6
Sudoku_3_3_7
Sudoku_3_3_8
Sudoku_3_3_9
Sudoku_3_4_1
Sudoku_3_4_2
Sudoku_3_4_3
Sudoku_3_4_4
Sudoku_3_4_5
Sudoku_3_4_6
Sudoku_3_4_7
Sudoku_3_4_8
Sudoku_3_4_9
Sudoku_3_5_1
Sudoku_3_5_2
Sudoku_3_5_3
Sudoku_3_5_4
Sudoku_3_5_5
Sudoku_3_5_6
Sudoku_3_5_7
Sudoku_3_5_8
Sudoku_3_5_9
Sudoku_3_6_1
Sudoku_3_6_2
Sudoku_3_6_3
Sudoku_3_6_4
Sudoku_3_6_5
Sudoku_3_6_6
Sudoku_3_6_7
Sudoku_3_6_8
Sudoku_3_6_9
Sudoku_3_7_1
Sudoku_3_7_2
Sudoku_3_7_3
Sudoku_3_7_4
Sudoku_3_7_5
Sudoku_3_7_6
Sudoku_3_7_7
Sudoku_3_7_8
Sudoku_3_7_9
Sudoku_3_8_1
Sudoku_3_8_2
Sudoku_3_8_3
Sudoku_3_8_4
Sudoku_3_8_5
Sudoku_3_8_6
Sudoku_3_8_7
Sudoku_3_8_8
Sudoku_3_8_9
Sudoku_3_9_1
Sudoku_3_9_2
Sudoku_3_9_3
Sudoku_3_9_4
Sudoku_3_9_5
Sudoku_3_9_6
Sudoku_3_9_7
Sudoku_3_9_8
Sudoku_3_9_9
Sudoku_4_1_1
Sudoku_4_1_2
Sudoku_4_1_3
Sudoku_4_1_4
Sudoku_4_1_5
Sudoku_4_1_6
Sudoku_4_1_7
Sudoku_4_1_8
Sudoku_4_1_9
Sudoku_4_2_1
Sudoku_4_2_2
Sudoku_4_2_3
Sudoku_4_2_4
Sudoku_4_2_5
Sudoku_4_2_6
Sudoku_4_2_7
Sudoku_4_2_8
Sudoku_4_2_9
Sudoku_4_3_1
Sudoku_4_3_2
Sudoku_4_3_3
Sudoku_4_3_4
Sudoku_4_3_5
Sudoku_4_3_6
Sudoku_4_3_7
Sudoku_4_3_8
Sudoku_4_3_9
Sudoku_4_4_1
Sudoku_4_4_2
Sudoku_4_4_3
Sudoku_4_4_4
Sudoku_4_4_5
Sudoku_4_4_6
Sudoku_4_4_7
Sudoku_4_4_8
Sudoku_4_4_9
Sudoku_4_5_1
Sudoku_4_5_2
Sudoku_4_5_3
Sudoku_4_5_4
Sudoku_4_5_5
Sudoku_4_5_6
Sudoku_4_5_7
Sudoku_4_5_8
Sudoku_4_5_9
Sudoku_4_6_1
Sudoku_4_6_2
Sudoku_4_6_3
Sudoku_4_6_4
Sudoku_4_6_5
Sudoku_4_6_6
Sudoku_4_6_7
Sudoku_4_6_8
Sudoku_4_6_9
Sudoku_4_7_1
Sudoku_4_7_2
Sudoku_4_7_3
Sudoku_4_7_4
Sudoku_4_7_5
Sudoku_4_7_6
Sudoku_4_7_7
Sudoku_4_7_8
Sudoku_4_7_9
Sudoku_4_8_1
Sudoku_4_8_2
Sudoku_4_8_3
Sudoku_4_8_4
Sudoku_4_8_5
Sudoku_4_8_6
Sudoku_4_8_7
Sudoku_4_8_8
Sudoku_4_8_9
Sudoku_4_9_1
Sudoku_4_9_2
Sudoku_4_9_3
Sudoku_4_9_4
Sudoku_4_9_5
Sudoku_4_9_6
Sudoku_4_9_7
Sudoku_4_9_8
Sudoku_4_9_9
Sudoku_5_1_1
Sudoku_5_1_2
Sudoku_5_1_3
Sudoku_5_1_4
Sudoku_5_1_5
Sudoku_5_1_6
Sudoku_5_1_7
Sudoku_5_1_8
Sudoku_5_1_9
Sudoku_5_2_1
Sudoku_5_2_2
Sudoku_5_2_3
Sudoku_5_2_4
Sudoku_5_2_5
Sudoku_5_2_6
Sudoku_5_2_7
Sudoku_5_2_8
Sudoku_5_2_9
Sudoku_5_3_1
Sudoku_5_3_2
Sudoku_5_3_3
Sudoku_5_3_4
Sudoku_5_3_5
Sudoku_5_3_6
Sudoku_5_3_7
Sudoku_5_3_8
Sudoku_5_3_9
Sudoku_5_4_1
Sudoku_5_4_2
Sudoku_5_4_3
Sudoku_5_4_4
Sudoku_5_4_5
Sudoku_5_4_6
Sudoku_5_4_7
Sudoku_5_4_8
Sudoku_5_4_9
Sudoku_5_5_1
Sudoku_5_5_2
Sudoku_5_5_3
Sudoku_5_5_4
Sudoku_5_5_5
Sudoku_5_5_6
Sudoku_5_5_7
Sudoku_5_5_8
Sudoku_5_5_9
Sudoku_5_6_1
Sudoku_5_6_2
Sudoku_5_6_3
Sudoku_5_6_4
Sudoku_5_6_5
Sudoku_5_6_6
Sudoku_5_6_7
Sudoku_5_6_8
Sudoku_5_6_9
Sudoku_5_7_1
Sudoku_5_7_2
Sudoku_5_7_3
Sudoku_5_7_4
Sudoku_5_7_5
Sudoku_5_7_6
Sudoku_5_7_7
Sudoku_5_7_8
Sudoku_5_7_9
Sudoku_5_8_1
Sudoku_5_8_2
Sudoku_5_8_3
Sudoku_5_8_4
Sudoku_5_8_5
Sudoku_5_8_6
Sudoku_5_8_7
Sudoku_5_8_8
Sudoku_5_8_9
Sudoku_5_9_1
Sudoku_5_9_2
Sudoku_5_9_3
Sudoku_5_9_4
Sudoku_5_9_5
Sudoku_5_9_6
Sudoku_5_9_7
Sudoku_5_9_8
Sudoku_5_9_9
Sudoku_6_1_1
Sudoku_6_1_2
Sudoku_6_1_3
Sudoku_6_1_4
Sudoku_6_1_5
Sudoku_6_1_6
Sudoku_6_1_7
Sudoku_6_1_8
Sudoku_6_1_9
Sudoku_6_2_1
Sudoku_6_2_2
Sudoku_6_2_3
Sudoku_6_2_4
Sudoku_6_2_5
Sudoku_6_2_6
Sudoku_6_2_7
Sudoku_6_2_8
Sudoku_6_2_9
Sudoku_6_3_1
Sudoku_6_3_2
Sudoku_6_3_3
Sudoku_6_3_4
Sudoku_6_3_5
Sudoku_6_3_6
Sudoku_6_3_7
Sudoku_6_3_8
Sudoku_6_3_9
Sudoku_6_4_1
Sudoku_6_4_2
Sudoku_6_4_3
Sudoku_6_4_4
Sudoku_6_4_5
Sudoku_6_4_6
Sudoku_6_4_7
Sudoku_6_4_8
Sudoku_6_4_9
Sudoku_6_5_1
Sudoku_6_5_2
Sudoku_6_5_3
Sudoku_6_5_4
Sudoku_6_5_5
Sudoku_6_5_6
Sudoku_6_5_7
Sudoku_6_5_8
Sudoku_6_5_9
Sudoku_6_6_1
Sudoku_6_6_2
Sudoku_6_6_3
Sudoku_6_6_4
Sudoku_6_6_5
Sudoku_6_6_6
Sudoku_6_6_7
Sudoku_6_6_8
Sudoku_6_6_9
Sudoku_6_7_1
Sudoku_6_7_2
Sudoku_6_7_3
Sudoku_6_7_4
Sudoku_6_7_5
Sudoku_6_7_6
Sudoku_6_7_7
Sudoku_6_7_8
Sudoku_6_7_9
Sudoku_6_8_1
Sudoku_6_8_2
Sudoku_6_8_3
Sudoku_6_8_4
Sudoku_6_8_5
Sudoku_6_8_6
Sudoku_6_8_7
Sudoku_6_8_8
Sudoku_6_8_9
Sudoku_6_9_1
Sudoku_6_9_2
Sudoku_6_9_3
Sudoku_6_9_4
Sudoku_6_9_5
Sudoku_6_9_6
Sudoku_6_9_7
Sudoku_6_9_8
Sudoku_6_9_9
Sudoku_7_1_1
Sudoku_7_1_2
Sudoku_7_1_3
Sudoku_7_1_4
Sudoku_7_1_5
Sudoku_7_1_6
Sudoku_7_1_7
Sudoku_7_1_8
Sudoku_7_1_9
Sudoku_7_2_1
Sudoku_7_2_2
Sudoku_7_2_3
Sudoku_7_2_4
Sudoku_7_2_5
Sudoku_7_2_6
Sudoku_7_2_7
Sudoku_7_2_8
Sudoku_7_2_9
Sudoku_7_3_1
Sudoku_7_3_2
Sudoku_7_3_3
Sudoku_7_3_4
Sudoku_7_3_5
Sudoku_7_3_6
Sudoku_7_3_7
Sudoku_7_3_8
Sudoku_7_3_9
Sudoku_7_4_1
Sudoku_7_4_2
Sudoku_7_4_3
Sudoku_7_4_4
Sudoku_7_4_5
Sudoku_7_4_6
Sudoku_7_4_7
Sudoku_7_4_8
Sudoku_7_4_9
Sudoku_7_5_1
Sudoku_7_5_2
Sudoku_7_5_3
Sudoku_7_5_4
Sudoku_7_5_5
Sudoku_7_5_6
Sudoku_7_5_7
Sudoku_7_5_8
Sudoku_7_5_9
Sudoku_7_6_1
Sudoku_7_6_2
Sudoku_7_6_3
Sudoku_7_6_4
Sudoku_7_6_5
Sudoku_7_6_6
Sudoku_7_6_7
Sudoku_7_6_8
Sudoku_7_6_9
Sudoku_7_7_1
Sudoku_7_7_2
Sudoku_7_7_3
Sudoku_7_7_4
Sudoku_7_7_5
Sudoku_7_7_6
Sudoku_7_7_7
Sudoku_7_7_8
Sudoku_7_7_9
Sudoku_7_8_1
Sudoku_7_8_2
Sudoku_7_8_3
Sudoku_7_8_4
Sudoku_7_8_5
Sudoku_7_8_6
Sudoku_7_8_7
Sudoku_7_8_8
Sudoku_7_8_9
Sudoku_7_9_1
Sudoku_7_9_2
Sudoku_7_9_3
Sudoku_7_9_4
Sudoku_7_9_5
Sudoku_7_9_6
Sudoku_7_9_7
Sudoku_7_9_8
Sudoku_7_9_9
Sudoku_8_1_1
Sudoku_8_1_2
Sudoku_8_1_3
Sudoku_8_1_4
Sudoku_8_1_5
Sudoku_8_1_6
Sudoku_8_1_7
Sudoku_8_1_8
Sudoku_8_1_9
Sudoku_8_2_1
Sudoku_8_2_2
Sudoku_8_2_3
Sudoku_8_2_4
Sudoku_8_2_5
Sudoku_8_2_6
Sudoku_8_2_7
Sudoku_8_2_8
Sudoku_8_2_9
Sudoku_8_3_1
Sudoku_8_3_2
Sudoku_8_3_3
Sudoku_8_3_4
Sudoku_8_3_5
Sudoku_8_3_6
Sudoku_8_3_7
Sudoku_8_3_8
Sudoku_8_3_9
Sudoku_8_4_1
Sudoku_8_4_2
Sudoku_8_4_3
Sudoku_8_4_4
Sudoku_8_4_5
Sudoku_8_4_6
Sudoku_8_4_7
Sudoku_8_4_8
Sudoku_8_4_9
Sudoku_8_5_1
Sudoku_8_5_2
Sudoku_8_5_3
Sudoku_8_5_4
Sudoku_8_5_5
Sudoku_8_5_6
Sudoku_8_5_7
Sudoku_8_5_8
Sudoku_8_5_9
Sudoku_8_6_1
Sudoku_8_6_2
Sudoku_8_6_3
Sudoku_8_6_4
Sudoku_8_6_5
Sudoku_8_6_6
Sudoku_8_6_7
Sudoku_8_6_8
Sudoku_8_6_9
Sudoku_8_7_1
Sudoku_8_7_2
Sudoku_8_7_3
Sudoku_8_7_4
Sudoku_8_7_5
Sudoku_8_7_6
Sudoku_8_7_7
Sudoku_8_7_8
Sudoku_8_7_9
Sudoku_8_8_1
Sudoku_8_8_2
Sudoku_8_8_3
Sudoku_8_8_4
Sudoku_8_8_5
Sudoku_8_8_6
Sudoku_8_8_7
Sudoku_8_8_8
Sudoku_8_8_9
Sudoku_8_9_1
Sudoku_8_9_2
Sudoku_8_9_3
Sudoku_8_9_4
Sudoku_8_9_5
Sudoku_8_9_6
Sudoku_8_9_7
Sudoku_8_9_8
Sudoku_8_9_9
Sudoku_9_1_1
Sudoku_9_1_2
Sudoku_9_1_3
Sudoku_9_1_4
Sudoku_9_1_5
Sudoku_9_1_6
Sudoku_9_1_7
Sudoku_9_1_8
Sudoku_9_1_9
Sudoku_9_2_1
Sudoku_9_2_2
Sudoku_9_2_3
Sudoku_9_2_4
Sudoku_9_2_5
Sudoku_9_2_6
Sudoku_9_2_7
Sudoku_9_2_8
Sudoku_9_2_9
Sudoku_9_3_1
Sudoku_9_3_2
Sudoku_9_3_3
Sudoku_9_3_4
Sudoku_9_3_5
Sudoku_9_3_6
Sudoku_9_3_7
Sudoku_9_3_8
Sudoku_9_3_9
Sudoku_9_4_1
Sudoku_9_4_2
Sudoku_9_4_3
Sudoku_9_4_4
Sudoku_9_4_5
Sudoku_9_4_6
Sudoku_9_4_7
Sudoku_9_4_8
Sudoku_9_4_9
Sudoku_9_5_1
Sudoku_9_5_2
Sudoku_9_5_3
Sudoku_9_5_4
Sudoku_9_5_5
Sudoku_9_5_6
Sudoku_9_5_7
Sudoku_9_5_8
Sudoku_9_5_9
Sudoku_9_6_1
Sudoku_9_6_2
Sudoku_9_6_3
Sudoku_9_6_4
Sudoku_9_6_5
Sudoku_9_6_6
Sudoku_9_6_7
Sudoku_9_6_8
Sudoku_9_6_9
Sudoku_9_7_1
Sudoku_9_7_2
Sudoku_9_7_3
Sudoku_9_7_4
Sudoku_9_7_5
Sudoku_9_7_6
Sudoku_9_7_7
Sudoku_9_7_8
Sudoku_9_7_9
Sudoku_9_8_1
Sudoku_9_8_2
Sudoku_9_8_3
Sudoku_9_8_4
Sudoku_9_8_5
Sudoku_9_8_6
Sudoku_9_8_7
Sudoku_9_8_8
Sudoku_9_8_9
Sudoku_9_9_1
Sudoku_9_9_2
Sudoku_9_9_3
Sudoku_9_9_4
Sudoku_9_9_5
Sudoku_9_9_6
Sudoku_9_9_7
Sudoku_9_9_8
Sudoku_9_9_9
End
