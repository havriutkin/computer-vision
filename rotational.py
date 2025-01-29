import sympy as sp
import numpy as np
from prettytable import PrettyTable
from termcolor import colored

# ========== Define symbols ==========
cos_alpha, sin_alpha = sp.symbols(u'cos(α) sin(α)')
cos_beta, sin_beta = sp.symbols(u'cos(β) sin(β)')
cos_gamma, sin_gamma = sp.symbols(u'cos(γ) sin(γ')

# Define general rotational matrices
R_y = sp.Matrix([[cos_alpha, 0, sin_alpha], [0, 1, 0], [-sin_alpha, 0, cos_alpha]])
R_x = sp.Matrix([[1, 0, 0], [0, cos_beta, -sin_beta], [0, sin_beta, cos_beta]])
R_z = sp.Matrix([[cos_gamma, -sin_gamma, 0], [sin_gamma, cos_gamma, 0], [0, 0, 1]])

# Coordinates of points in the image plane
x1, x2 = sp.symbols('x1 x2')   
y1, y2 = sp.symbols('y1 y2')
t1, t2 = sp.symbols('t1 t2')


# ========== Rotations around y-axis and x-axis to bring [x1, y1] to the origin ==========

# In R_y substitute cos_alpha with t1x1 / x1(x1 + 1)
#                   sin_alpha with -t1x1 / (x1 + 1)
R_y_sub_trig = R_y.subs({
    cos_alpha: t1 * x1 / (x1 * (x1 + 1)),
    sin_alpha: -t1 * x1 / (x1 + 1)
})

# In R_x substitute cos_beta with t1t2y1 / y1(y1^2 _ t1^2)
#                   sin_beta with t2y1 / (y1^2 + t1^2)
R_x_sub_trig = R_x.subs({
    cos_beta: t1 * t2 * y1 / (y1 * (y1 ** 2 + t1 ** 2)),
    sin_beta: t2 * y1 / (y1 ** 2 + t1 ** 2)
})

# In R_y substitute t1 with |x1 + 1| / sqrt(x1^2 + 1)
R_y_sub_t1 = R_y_sub_trig.subs({
    t1: sp.Abs(x1 + 1) / sp.sqrt(x1 ** 2 + 1)
})

# In R_x substitute t2 with sqrt(y1^2 + t1^2)
R_x_sub_t2 = R_x_sub_trig.subs({
    t2: sp.sqrt(y1 ** 2 + t1 ** 2)
})

# In R_x substitute t1 with |x1 + 1| / sqrt(x1^2 + 1)
R_x_sub_t1 = R_x_sub_t2.subs({
    t1: sp.Abs(x1 + 1) / sp.sqrt(x1 ** 2 + 1)
})

# ========== Rotations around z-axis to bring [x2, y2] to [0, 1, n] ==========

# Combine R_y and R_x
T = R_x_sub_t1 * R_y_sub_t1

# Act on [x2, y2, 1] with T
u = sp.Matrix([x2, y2, 1])
u_new = T * u
x2_new, y2_new, z2_new = u_new

# In R_z substitute cos_gamma with y2_new / (x2_new^2 + y2_new^2)
#                   sin_gamma with x2_new / (x2_new^2 + y2_new^2)
R_z_sub_trig = R_z.subs({
    cos_gamma: y2_new / (x2_new ** 2 + y2_new ** 2),
    sin_gamma: x2_new / (x2_new ** 2 + y2_new ** 2)
})

# Combine all rotations
M = R_z_sub_trig * T


# ========== Test ==========

# Create values for points
x1_val, y1_val = 3, 5
x2_val, y2_val = 7, 7
u_val = np.array([x1_val, y1_val, 1])
v_val = np.array([x2_val, y2_val, 1])

# Substitute values into M
M_val = M.subs({
    x1: x1_val, y1: y1_val,
    x2: x2_val, y2: y2_val
})

# Check if M_val is orthogonal
product = M_val * M_val.T
product = sp.simplify(product)
print(colored("\nOrthogonality check:", 'cyan'))
print(product)

# Check the determinant of M
det = M_val.det()
print(colored(f"\nDet M = {det}", 'cyan'))

# Act by M on u and v
u_new = M_val * sp.Matrix(u_val)
v_new = M_val * sp.Matrix(v_val)

# Print the results
print(colored("New coordinates of the vector u after rotation by M:", 'cyan'))
table = PrettyTable()
table.field_names = ["Symbol", "Value"]
table.add_row(["x_1", sp.simplify(u_new[0])])
table.add_row(["y_1", sp.simplify(u_new[1])])
table.add_row(["z_1", sp.simplify(u_new[2])])
print(table)

print(colored("New coordinates of the vector v after rotation by M:", 'cyan'))
table = PrettyTable()
table.field_names = ["Symbol", "Value"]
table.add_row(["x_2", sp.simplify(v_new[0])])
table.add_row(["y_2", sp.simplify(v_new[1])])
table.add_row(["z_2", sp.simplify(v_new[2])])
table.add_row(["y_2 / z_2", sp.simplify(v_new[1] / v_new[2])])
print(table)

# Act by T on u
T_val = T.subs({
    x1: x1_val, y1: y1_val
})
u_new = T_val * sp.Matrix(u_val)
print(colored("New coordinates of the vector u after rotation by T:", 'cyan'))
table = PrettyTable()
table.field_names = ["Symbol", "Value"]
table.add_row(["x_1", sp.simplify(u_new[0])])
table.add_row(["y_1", sp.simplify(u_new[1])])
table.add_row(["z_1", sp.simplify(u_new[2])])
table.add_row(["y_1 / z_1", sp.simplify(u_new[1] / u_new[2]).evalf()])
print(table)