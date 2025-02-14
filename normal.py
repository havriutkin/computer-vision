import sympy as sp
import numpy as np
from prettytable import PrettyTable
from termcolor import colored

def print_table(title, values):
    print(colored(title, 'cyan'))
    table = PrettyTable()
    table.field_names = ["Symbol", "Value"]
    for key, value in values.items():
        table.add_row([key, value])
    print(table)


# Coordinates of points in the image plane
x1, x2 = sp.symbols('x1 x2')   
y1, y2 = sp.symbols('y1 y2')

# Define unit vectors
e3 = sp.Matrix([x1, y1, 1])

e2 = sp.Matrix([x2 - x1, y2 - y1, 0])  # Direction vector
#e2 = e2 / e2.norm()  # Normalize e2

# Rotate e1 by the same angle as e2
cos_alpha = (y2 - y1) * sp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
sin_alpha = sp.Abs(x2 - x1) * sp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
rotation_matrix = sp.Matrix([
    [cos_alpha, -sin_alpha, 0],
    [sin_alpha, cos_alpha, 0],
    [0, 0, 1],
])
e1 = rotation_matrix * sp.Matrix([1, 0, 0])

print(e1)
print(e2)
print(e3)

# Compute e3 as cross product of e1 and e2
#e3 = e3 / e3.norm()  # Normalize e3

# Define the matrix M
M = sp.Matrix([
    [e1[0], e2[0], e3[0]],
    [e1[1], e2[1], e3[1]],
    [e1[2], e2[2], e3[2]],
])

# ========== Testing ==========
# Generate p1 = (x1, y1, 1) and p2 = (x2, y2, 1)
p1 = sp.Matrix([2, 1, 1])
p2 = sp.Matrix([6, 4, 1])

# Substitute the values of the points
M_val = M.subs({
    x1: p1[0],
    y1: p1[1],
    x2: p2[0],
    y2: p2[1],
})

# Apply the transformation M to the points p1 and p2
p1_new = M_val * p1
p2_new = M_val * p2

# Print the results
print_table("New coordinates of the point p1 after rotation by M:", {
    "x_1": sp.simplify(p1_new[0]),
    "y_1": sp.simplify(p1_new[1]),
    "z_1": sp.simplify(p1_new[2]),
})

print_table("New coordinates of the point p2 after rotation by M:", {
    "x_2": sp.simplify(p2_new[0]),
    "y_2": sp.simplify(p2_new[1]),
    "z_2": sp.simplify(p2_new[2]),
})