import sympy as sp
import numpy as np
from prettytable import PrettyTable
from termcolor import colored

# ========== Define symbols ==========
# Entries of a matrix M from PGL(3)
a = sp.symbols('a:9')
# Coordinates of points in the image plane
x1, x2 = sp.symbols('x1 x2')   
y1, y2 = sp.symbols('y1 y2')

# ========== Define the matrix M ==========
"""
M = sp.Matrix([[a[0], a[1], a[2]],
               [a[3], a[4], a[5]],
               [a[6], a[7], a[8]]])
"""

def find_transform(x1_val, y1_val, x2_val, y2_val):
    """
    When x1 = x2 = 0:
        +--------+---------------------------------+
        | Symbol |              Value              |
        +--------+---------------------------------+
        |  a_0   | y1*y2/(a5*(a8*y1 - a8*y2 - y1)) |
        |  a_1   |                0                |
        |  a_2   |                0                |
        |  a_3   |                1                |
        |  a_4   |              -a5/y1             |
        |  a_5   |                1                |
        |  a_6   |                1                |
        |  a_7   |           -(a8 - 1)/y2          |
        |  a_8   |                1                |
        +--------+---------------------------------+

    Else if x1 * y2 - x2 * y1 = 0:
        +--------+----------------------------------------------------------------------------------------------------------+
        | Symbol |                                                  Value                                                   |
        +--------+----------------------------------------------------------------------------------------------------------+
        |  a_0   |    x1*x2*(y1 - y2)/((a4*x1*y2 - a4*x2*y1 + a5*x1 - a5*x2)*(a7*x1*y2 - a7*x2*y1 + a8*x1 - a8*x2 - x1))    |
        |  a_1   |   -x1*x2*(x1 - x2)/((a4*x1*y2 - a4*x2*y1 + a5*x1 - a5*x2)*(a7*x1*y2 - a7*x2*y1 + a8*x1 - a8*x2 - x1))    |
        |  a_2   | x1*x2*(x1*y2 - x2*y1)/((a4*x1*y2 - a4*x2*y1 + a5*x1 - a5*x2)*(a7*x1*y2 - a7*x2*y1 + a8*x1 - a8*x2 - x1)) |
        |  a_3   |                                             -(a4*y1 + a5)/x1                                             |
        |  a_4   |                                                    1                                                     |
        |  a_5   |                                                    1                                                     |
        |  a_6   |                                           -(a7*y2 + a8 - 1)/x2                                           |
        |  a_7   |                                                    1                                                     |
        |  a_8   |                                                    1                                                     |
        +--------+----------------------------------------------------------------------------------------------------------+

    Else:
        +--------+---------------------------------------+
        | Symbol |                 Value                 |
        +--------+---------------------------------------+
        |  a_0   |                -y1 + y2               |
        |  a_1   |                x1 - x2                |
        |  a_2   |             -x1*y2 + x2*y1            |
        |  a_3   |  (a5*y1 - a5*y2 - y1)/(x1*y2 - x2*y1) |
        |  a_4   | -(a5*x1 - a5*x2 - x1)/(x1*y2 - x2*y1) |
        |  a_5   |                   a5                  |
        |  a_6   |   (a8 - 1)*(y1 - y2)/(x1*y2 - x2*y1)  |
        |  a_7   |  -(a8 - 1)*(x1 - x2)/(x1*y2 - x2*y1)  |
        |  a_8   |                   a8                  |
        +--------+---------------------------------------+
    
    """
    if x1_val == 0 and x2_val == 0:
        a_values = {
            a[0]: y1_val * y2_val / (a[5] * (a[8] * y1_val - a[8] * y2_val - y1_val)),
            a[1]: 0,
            a[2]: 0,
            a[3]: 1,
            a[4]: -a[5] / y1_val,
            a[5]: 1,
            a[6]: 1,
            a[7]: -(a[8] - 1) / y2_val,
            a[8]: 1
        }
    elif x1_val * y2_val - x2_val * y1_val == 0:
        a_values = {
            a[0]: x1_val * x2_val * (y1_val - y2_val) / ((a[4] * x1_val * y2_val - a[4] * x2_val * y1_val + a[5] * x1_val - a[5] * x2_val) * (a[7] * x1_val * y2_val - a[7] * x2_val * y1_val + a[8] * x1_val - a[8] * x2_val - x1_val)),
            a[1]: -x1_val * x2_val * (x1_val - x2_val) / ((a[4] * x1_val * y2_val - a[4] * x2_val * y1_val + a[5] * x1_val - a[5] * x2_val) * (a[7] * x1_val * y2_val - a[7] * x2_val * y1_val + a[8] * x1_val - a[8] * x2_val - x1_val)),
            a[2]: x1_val * x2_val * (x1_val * y2_val - x2_val * y1_val) / ((a[4] * x1_val * y2_val - a[4] * x2_val * y1_val + a[5] * x1_val - a[5] * x2_val) * (a[7] * x1_val * y2_val - a[7] * x2_val * y1_val + a[8] * x1_val - a[8] * x2_val - x1_val)),
            a[3]: -(a[4] * y1_val + a[5]) / x1_val,
            a[4]: 1,
            a[5]: 1,
            a[6]: -(a[7] * y2_val + a[8] - 1) / x2_val,
            a[7]: 1,
            a[8]: 1
        }
    else:
        a_values = {
            a[0]: -y1_val + y2_val,
            a[1]: x1_val - x2_val,
            a[2]: -x1_val * y2_val + x2_val * y1_val,
            a[3]: (a[5] * y1_val - a[5] * y2_val - y1_val) / (x1_val * y2_val - x2_val * y1_val),
            a[4]: -(a[5] * x1_val - a[5] * x2_val - x1_val) / (x1_val * y2_val - x2_val * y1_val),
            a[5]: a[5],
            a[6]: (a[8] - 1) * (y1_val - y2_val) / (x1_val * y2_val - x2_val * y1_val),
            a[7]: -(a[8] - 1) * (x1_val - x2_val) / (x1_val * y2_val - x2_val * y1_val),
            a[8]: a[8]
        }

    M = sp.Matrix([[a_values[a[0]], a_values[a[1]], a_values[a[2]]],
                   [a_values[a[3]], a_values[a[4]], a_values[a[5]]],
                   [a_values[a[6]], a_values[a[7]], a_values[a[8]]]])
    
    p1 = M * sp.Matrix([x1_val, y1_val, 1])
    p2 = M * sp.Matrix([x2_val, y2_val, 1])

    return M, p1, p2

def test(min = -1000, max = 1000, iters = 100, details = False):
    for i in range(iters):
        print(colored(f"Test {i + 1}:", 'cyan'), end="")
        correct = True
        
        # Generate random points
        x1_val, y1_val = np.random.randint(min, max, 2)
        x2_val, y2_val = np.random.randint(min, max, 2)

        M, p1, p2 = find_transform(x1_val, y1_val, x2_val, y2_val)

        # Check that p1 is 0, 0
        if p1[0] != 0 or p1[1] != 0:
            correct = False

        # Check that p2[0] is 0 and p2[2] is 1
        if p2[0] != 0 or p2[2] != 1:
            correct = False

        # Check that determinant of M is 1
        if M.det() != 1:
            correct = False

        if correct:
            print(colored(" Correct", 'green'))
            if details:
                print(f"\t Points: ({x1_val}, {y1_val}), ({x2_val}, {y2_val})")
                print(f"\t p1 = {p1}, p2 = {p2}")
        else:
            print(colored(" Incorrect", 'red'))
            print(f"Points: ({x1_val}, {y1_val}), ({x2_val}, {y2_val})")
            print(colored("\nMatrix M:", 'cyan'))
            for i in range(3):
                for j in range(3):
                    print(f"{M[i * 3 + j]:10.4f}", end=" ")
                print()
            print(f"p1 = {p1}")
            print(f"p2 = {p2}")
            break
            
if __name__ == "__main__":
    test(iters=500)
