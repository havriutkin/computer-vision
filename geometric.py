import sympy as sp
import numpy as np
from prettytable import PrettyTable
from termcolor import colored

# ========== Define symbols ==========
x1, y1 = sp.symbols('x1 y1') # Vector u = (x1, y1, 1)
x2, y2 = sp.symbols('x2 y2') # Vector v = (x2, y2, 1)

# Angle of rotation around the y-axis
cos_alpha = sp.symbols(u'cos(α)') 
sin_alpha = sp.symbols(u'sin(α)')

# Angle of rotation around the x-axis
cos_beta = sp.symbols(u'cos(β)')
sin_beta = sp.symbols(u'sin(β)')

# Angle of rotation aroung the z-axis
cos_gamma = sp.symbols(u'cos(γ)')
sin_gamma = sp.symbols(u'sin(γ')

# Define general rotational matrices
R_y = sp.Matrix([[cos_alpha, 0, sin_alpha], [0, 1, 0], [-sin_alpha, 0, cos_alpha]])
R_x = sp.Matrix([[1, 0, 0], [0, cos_beta, -sin_beta], [0, sin_beta, cos_beta]])
R_z = sp.Matrix([[cos_gamma, -sin_gamma, 0], [sin_gamma, cos_gamma, 0], [0, 0, 1]])


# First rotate around the y-axis
# Find the angle between the vector [0, 0, 1] and [x1, 0, 1]
norm1 = 1
norm2 = sp.sqrt(x1 ** 2 + 1)
dot_product = 0 * x1 + 0 * 0 + 1 * 1
c_alpha = dot_product / (norm1 * norm2)
s_alpha = sp.sqrt(1 - c_alpha ** 2)

R_y_val = R_y.subs({
    cos_alpha: sp.sqrt(x1 ** 2 + 1) / (x1 ** 2 + 1),
    sin_alpha: sp.sqrt(1 - (sp.sqrt(x1 ** 2 + 1) / (x1 ** 2 + 1)) ** 2),
})

# Apply R_y to the vector u = (x1, y1, 1)
u = sp.Matrix([x1, y1, 1])
u_new = R_y_val * u
print(colored("New coordinates of the vector u after rotation by R_y:", 'cyan'))
table = PrettyTable()
table.field_names = ["Symbol", "Value"]
table.add_row(["x_1", sp.simplify(u_new[0])])
table.add_row(["y_1", sp.simplify(u_new[1])])
table.add_row(["z_1", sp.simplify(u_new[2])])
table.add_row(["x_1 / z_1", sp.simplify(u_new[0] / u_new[2])])
print(table)

# Then rotate around the x-axis
# Find the angle between the vector [x1, 0, 1] and [x1, y1, 1]
norm1 = sp.sqrt(x1 ** 2 + 1)
norm2 = sp.sqrt(x1 ** 2 + y1 ** 2 + 1)
dot_product = x1 * x1 + 0 * y1 + 1 * 1
c_beta = dot_product / (norm1 * norm2)
s_beta = sp.sqrt(1 - c_beta ** 2)

R_x_val = R_x.subs({
    cos_beta: c_beta,
    sin_beta: s_beta,
})

# Find the new coordinates of x_2 and y_2 after rotating by R_y then R_x
v = sp.Matrix([x2, y2, 1])
v_new = R_x_val * R_y_val * v

print(colored("New coordinates of the vector v after rotation by R_y then R_x:", 'cyan'))
table = PrettyTable()
table.field_names = ["Symbol", "Value"]
table.add_row(["x_2", v_new[0]])
table.add_row(["y_2", v_new[1]])
print(table)

x2_new = v_new[0]
y2_new = v_new[1]

# Rotate around the z-axis 
# cos(γ) = y2_new / (sqrt(y2_new^2 + x2_new ^2)) and sin(γ) = sqrt(1 - cos(γ)^2)
R_z_val = R_z.subs({
    cos_gamma: y2_new / sp.sqrt(y2_new ** 2 + x2_new ** 2),
    sin_gamma: sp.sqrt(1 - (y2_new / sp.sqrt(y2_new ** 2 + x2_new ** 2)) ** 2),
})

# Calculate M = R_z * R_x * R_y
M = R_z_val * R_x_val * R_y_val

# Try concrete u and v
u = sp.Matrix([3, 2, 1])
v = sp.Matrix([5, 5, 1])

# Substitute u and v into M
M_val = M.subs({x1: u[0], y1: u[1], x2: v[0], y2: v[1]})

print(colored("\nMatrix M:", 'cyan'))
for i in range(3):
    for j in range(3):
        print(f"{M_val[i * 3 + j]:10.4f}", end=" ")
    print()

# Check determinant of the matrix M
det = M_val.det()
print(colored(f"\nDet M = {det}", 'cyan'))

# Apply M to the vector u and v
u_new = M_val * u
v_new = M_val * v

print(colored("\nNew coordinates of the vector u after rotation by R_z, R_x, R_y:", 'cyan'))
table = PrettyTable()
table.field_names = ["Symbol", "Value"]
table.add_row(["x_1", sp.simplify(u_new[0])])
table.add_row(["y_1", sp.simplify(u_new[1])])
table.add_row(["z_1", sp.simplify(u_new[2])])
print(table)

print(colored("\nNew coordinates of the vector v after rotation by R_z, R_x, R_y:", 'cyan'))
table = PrettyTable()
table.field_names = ["Symbol", "Value"]
table.add_row(["x_2", sp.simplify(v_new[0])])
table.add_row(["y_2", sp.simplify(v_new[1])])
table.add_row(["z_2", sp.simplify(v_new[2])])
print(table)



