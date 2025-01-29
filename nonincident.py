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
eq4 = sp.Eq(M[1, 0] * x2 + M[1, 1] * y2 + M[1, 2], 1)

# Constraints: det M != 0 and third coordinate is not zero
eq5 = sp.Eq(sp.det(M), 1)
eq6 = sp.Eq(M[2, 0] * x1 + M[2, 1] * y1 + M[2, 2], 1)
eq7 = sp.Eq(M[2, 0] * x2 + M[2, 1] * y2 + M[2, 2], 1)

# Orthogonality constraints
product = M * M.T
eq8 = sp.Eq(product[0, 0], 1)
eq9 = sp.Eq(product[1, 1], 1)
eq10 = sp.Eq(product[2, 2], 1)
eq11 = sp.Eq(product[0, 1], 0)


# ========== Solve the system of equations ==========
# Solve with respect to the entries of the matrix M
solution = sp.solve([eq1, eq2, eq3, eq4, eq5, eq8, eq9, eq10], a) # Solution is a list
solution = solution[0] # Extract only solution from the list

# ========== Print the solution ==========
print(colored("Solution:", 'cyan'))
table = PrettyTable()
table.field_names = ["Symbol", "Value"]
for i in range(9):
    table.add_row([f"a_{i}", solution[i]])
print(table)

"""
Solution:
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

# ========== Test the solution ==========
# Create random two non zero points
x1_val, y1_val = np.random.randint(-10, 10, 2)
x2_val, y2_val = np.random.randint(-10, 10, 2)
print(colored(f"\nRandom points: ({x1_val}, {y1_val}), ({x2_val}, {y2_val})", 'cyan'))

# Substitute solution into the matrix M
M_val = M.subs({
    a[0]: solution[0], a[1]: solution[1], a[2]: solution[2],
    a[3]: solution[3], a[4]: solution[4], a[5]: solution[5],
    a[6]: solution[6], a[7]: solution[7], a[8]: solution[8],
})

# Substitute points into the matrix M, set a5 = 1, a8 = 1
M_val = M_val.subs({x1: x1_val, y1: y1_val, x2: x2_val, y2: y2_val, a[7]: 1, a[8]: 1})

# Check if M_val is orthogonal
eq11 = sp.Eq(M_val * M_val.T, sp.eye(3))
print(colored("\nOrthogonality check:", 'cyan'))
print(eq11)
product = M_val * M_val.T
print(f"Product of M_val and its transpose:")
for i in range(3):
    for j in range(3):
        print(f"{sp.simplify(product[i, j])}", end=" ")
    print()


print(colored("\nMatrix M:", 'cyan'))
for i in range(3):
    for j in range(3):
        print(f"{M_val[i * 3 + j]:10.4f}", end=" ")
    print()

# Check determinant of the matrix M
det = M_val.det()
print(colored(f"\nDet M = {det}", 'cyan'))

# Print the inverse of M_val
M_inv = M_val.inv()
print(colored("\nInverse of M:", 'cyan'))
for i in range(3):
    for j in range(3):
        print(f"{M_inv[i * 3 + j]:10.4f}", end=" ")
    print()

# Act on the points with the matrix M
p1 = M_val * sp.Matrix([x1_val, y1_val, 1])
p2 = M_val * sp.Matrix([x2_val, y2_val, 1])

print(colored("\nTransformed points:", 'cyan'))
print(f"p1 = {p1}", 'cyan')
print(f"p2 = {p2}", 'cyan')

# Take a scaled x1_val
x1_val = 2 * x1_val
y1_val = 2 * y1_val
p1 = M_val * sp.Matrix([x1_val, y1_val, 1])
print(colored("\nTransformed point after scaling x1:", 'cyan'))
print(f"p1 scaled = {p1}", 'cyan')
