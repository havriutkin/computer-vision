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
M = sp.Matrix([[a[0], a[1], a[2]],
               [a[3], a[4], a[5]],
               [a[6], a[7], a[8]]])

# ========== Define equations ==========
# Requirements: M(x1, y1, 1) = (0, 0) and M(x2, y2, 1) = (0, n)
eq1 = sp.Eq(M[0, 0] * x1 + M[0, 1] * y1 + M[0, 2], 0)
eq2 = sp.Eq(M[1, 0] * x1 + M[1, 1] * y1 + M[1, 2], 0)
eq3 = sp.Eq(M[0, 0] * x2 + M[0, 1] * y2 + M[0, 2], 0)
eq4 = sp.Eq(M[2, 0] * x2 + M[2, 1] * y2 + M[2, 2], 1)

# Constraints: det M != 0 and x1*y2 - x2*y1 = 0
eq5 = sp.Eq(sp.det(M), 1)
eq6 = sp.Eq(x1 * y2 - x2 * y1, 0)

# ========== Solve the system of equations ==========
# Solve with respect to the entries of the matrix M
print("Solving the system of equations...")
solution = sp.solve([eq1, eq2, eq3, eq4, eq5, eq6], a) # Solution is a list
print(f"Number of solutions: {len(solution)}")
solution = solution[0] # Extract the only solution from the list

# ========== Print the solution ==========
print(colored("Solution:", 'cyan'))
table = PrettyTable()
table.field_names = ["Symbol", "Value"]
for i in range(9):
    table.add_row([f"a_{i}", solution[i]])
print(table)

"""
Solution:
+--------+----------------------------------------------------------------------------------------------------------+
| Symbol |                                                  Value                                                   |
+--------+----------------------------------------------------------------------------------------------------------+
|  a_0   |    x1*x2*(y1 - y2)/((a4*x1*y2 - a4*x2*y1 + a5*x1 - a5*x2)*(a7*x1*y2 - a7*x2*y1 + a8*x1 - a8*x2 - x1))    |
|  a_1   |   -x1*x2*(x1 - x2)/((a4*x1*y2 - a4*x2*y1 + a5*x1 - a5*x2)*(a7*x1*y2 - a7*x2*y1 + a8*x1 - a8*x2 - x1))    |
|  a_2   | x1*x2*(x1*y2 - x2*y1)/((a4*x1*y2 - a4*x2*y1 + a5*x1 - a5*x2)*(a7*x1*y2 - a7*x2*y1 + a8*x1 - a8*x2 - x1)) |
|  a_3   |                                             -(a4*y1 + a5)/x1                                             |
|  a_4   |                                                    a4                                                    |
|  a_5   |                                                    a5                                                    |
|  a_6   |                                           -(a7*y2 + a8 - 1)/x2                                           |
|  a_7   |                                                    a7                                                    |
|  a_8   |                                                    a8                                                    |
+--------+----------------------------------------------------------------------------------------------------------+
"""


# ========== Test the solution ==========
# Create random two non zero points, such that x1*y2 - x2*y1 = 0
x1_val, y1_val = np.random.randint(-10, 10, 2)
factor = np.random.randint(1, 10)
x2_val, y2_val = factor * x1_val, factor * y1_val

print(colored(f"\nRandom points: ({x1_val}, {y1_val}), ({x2_val}, {y2_val})", 'cyan'))

# Substitute solution into the matrix M
M_val = M.subs({
    a[0]: solution[0], a[1]: solution[1], a[2]: solution[2],
    a[3]: solution[3], a[4]: solution[4], a[5]: solution[5],
    a[6]: solution[6], a[7]: solution[7], a[8]: solution[8],
})

# Substitute points into the matrix M, set a5 = 1, a8 = 1
M_val = M_val.subs({x1: x1_val, y1: y1_val, x2: x2_val, y2: y2_val, 
                    a[4]: 1, a[5]: 1, a[6]: 1, a[7]: 1, a[8]: 1})

print(colored("\nMatrix M:", 'cyan'))
for i in range(3):
    for j in range(3):
        print(f"{M_val[i * 3 + j]:10.4f}", end=" ")
    print()

# Check determinant of the matrix M
det = M_val.det()
print(colored(f"\nDet M = {det}", 'cyan'))

# Act on the points with the matrix M
p1 = M_val * sp.Matrix([x1_val, y1_val, 1])
p2 = M_val * sp.Matrix([x2_val, y2_val, 1])

print(colored("\nTransformed points:", 'cyan'))
print(f"p1 = {p1}", 'cyan')
print(f"p2 = {p2}", 'cyan')