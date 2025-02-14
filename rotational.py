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

def find_angle_between_vectors(u, v):
    norm1 = sp.sqrt(u[0] ** 2 + u[1] ** 2)
    norm2 = sp.sqrt(v[0] ** 2 + v[1] ** 2)
    dot_product = u[0] * v[0] + u[1] * v[1]
    c_alpha = sp.simplify(dot_product / (norm1 * norm2))
    s_alpha = sp.simplify(sp.sqrt(1 - c_alpha ** 2))
    return c_alpha, s_alpha

# ========== Define symbols ==========
cos_alpha, sin_alpha = sp.symbols(u'cos(α) sin(α)')
cos_beta, sin_beta = sp.symbols(u'cos(β) sin(β)')
cos_gamma, sin_gamma = sp.symbols(u'cos(γ) sin(γ')

# Define general rotational matrices
R_y = sp.Matrix([[cos_alpha, 0, -sin_alpha], [0, 1, 0], [sin_alpha, 0, cos_alpha]])
R_x = sp.Matrix([[1, 0, 0], [0, cos_beta, -sin_beta], [0, sin_beta, cos_beta]])
R_z = sp.Matrix([[cos_gamma, -sin_gamma, 0], [sin_gamma, cos_gamma, 0], [0, 0, 1]])

# Coordinates of points in the image plane
x1, x2 = sp.symbols('x1 x2')   
y1, y2 = sp.symbols('y1 y2')

p1 = sp.Matrix([2, 1, 1])
p2 = sp.Matrix([6, 4, 1])

# Basis vectors
e1 = sp.Matrix([1, 0, 0])
e2 = sp.Matrix([0, 1, 0])
e3 = sp.Matrix([0, 0, 1])


# First rotate around the y-axis by angle between [0, 0, 1] and [x1, 0, 1]
print(colored("Rotating around the y-axis...", 'cyan'))
c_alpha, s_alpha = find_angle_between_vectors([e1[0], e1[2]], [p1[0], p1[2]])
R_y = R_y.subs({
    cos_alpha: c_alpha,
    sin_alpha: s_alpha,
})

# Apply R_y to all vectors
e1 = sp.simplify(R_y * e1)
e2 = sp.simplify(R_y * e2)
e3 = sp.simplify(R_y * e3)
p1 = sp.simplify(R_y * p1)
p2 = sp.simplify(R_y * p2)

print_table("After rotation by R_y", {
    "e1": [e1[0], e1[1], e1[2]],
    "e2": [e2[0], e2[1], e2[2]],
    "e3": [e3[0], e3[1], e3[2]],
    "p1": [p1[0], p1[1], p1[2]],
    "p2": [p2[0], p2[1], p2[2]],
})

# Rotate around the x-axis by angle between e3 and p1
print(colored("Rotating around the x-axis...", 'cyan'))
c_beta, s_beta = find_angle_between_vectors([e3[1], e3[2]], [p1[1], p1[2]])
R_x = R_x.subs({
    cos_beta: c_beta,
    sin_beta: s_beta,
})

# Apply R_x to all vectors
e1 = sp.simplify(R_x * e1)
e2 = sp.simplify(R_x * e2)
e3 = sp.simplify(R_x * e3)
p1 = sp.simplify(R_x * p1)
p2 = sp.simplify(R_x * p2)

print_table("After rotation by R_x", {
    "e1": [e1[0], e1[1], e1[2]],
    "e2": [e2[0], e2[1], e2[2]],
    "e3": [e3[0], e3[1], e3[2]],
    "p1": [p1[0], p1[1], p1[2]],
    "p2": [p2[0], p2[1], p2[2]],
})

# Rotate around the z-axis by angle between e2 and p2
print(colored("Rotating around the z-axis...", 'cyan'))
c_gamma, s_gamma = find_angle_between_vectors([e2[0], e2[1]], [p2[0], p2[1]])
R_z = R_z.subs({
    cos_gamma: c_gamma,
    sin_gamma: s_gamma,
})

# Apply R_z to all vectors
e1 = sp.simplify(R_z * e1)
e2 = sp.simplify(R_z * e2)
e3 = sp.simplify(R_z * e3)
p1 = sp.simplify(R_z * p1)
p2 = sp.simplify(R_z * p2)

print_table("After rotation by R_z", {
    "e1": [e1[0], e1[1], e1[2]],
    "e2": [e2[0], e2[1], e2[2]],
    "e3": [e3[0], e3[1], e3[2]],
    "p1": [p1[0], p1[1], p1[2]],
    "p2": [p2[0], p2[1], p2[2]],
})


# Combine the rotations
M = R_z * R_x * R_y

"""
# ========== Testing ==========
# Generate p1 = (x1, y1, 1) and p2 = (x2, y2, 1)


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
})"""