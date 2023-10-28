import numpy as np

# กำหนดค่า u และ g(u) เป็น NumPy arrays
u_values = np.array([-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
g_u_values = np.array([13, 14, 13, 9, 4, -4, 0, -2, -8, -14, -19, -21, -8, -2, 0.5, 4, 5, 9, 11, 10, 7])

# หาค่า v ที่เป็น argmin ของ g(u)
gu1 = np.min(g_u_values)
v1 = u_values[np.argmin((g_u_values))]
print('v1 =', v1)
print('gu1 =', gu1)
print()

gu2 = np.min(np.abs(g_u_values))
v2 = u_values[np.argmin(np.abs(g_u_values))]
print('v2 =', v2)
print('gu2 =', gu2)
print()

gu3 = np.max(g_u_values)
v3 = u_values[np.argmax(g_u_values)]
print('v3 =', v3)
print('gu3 =', gu3)
print()

gu4 = np.max((g_u_values[::-1]))
v4 = u_values[np.argmax((g_u_values[::-1]))]
print('v4 =', v4)
print('gu4 =', gu4)
print()

gu5 = np.max(-g_u_values)
v5 = u_values[np.argmax(-g_u_values)]
print('v4 =', v5)
print('gu4 =', gu5)
print()

gu6 = np.max( 5 - g_u_values)
v6 = u_values[np.argmax( 5 - g_u_values)]
print('v6 =', v6)
print('gu6 =', gu6)
print()

gu7 = np.max(g_u_values[1:])
v7 = u_values[np.argmax(g_u_values[1:])]
print('v7 =', v7)
print('gu7 =', gu7)
print()

