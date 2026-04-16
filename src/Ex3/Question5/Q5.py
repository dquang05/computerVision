import numpy as np
import pandas as pd

# Dữ liệu bảng 2: [u, v, X, Y, Z]
data = np.array([
    [1199, 553,  80,  30, 492],
    [1057, 462,  50,  50, 492],
    [ 639, 587, -40,  25, 492],
    [ 552, 796, -60, -20, 492],
    [1064, 880,  50, -40, 492],
    [1208, 552,  80,  30, 475],
    [1061, 459,  50,  50, 475],
    [ 629, 588, -40,  25, 475],
    [ 540, 804, -60, -20, 475],
    [1068, 888,  50, -40, 475],
    [1212, 554,  80,  30, 467],
    [1063, 458,  50,  50, 467],
    [ 624, 591, -40,  25, 467],
    [ 533, 811, -60, -20, 467],
    [1070, 897,  50, -40, 467]
], dtype=float)

# Tách dữ liệu
u = data[:, 0]
v = data[:, 1]
X = data[:, 2]
Y = data[:, 3]
Z = data[:, 4]

# Ma trận A = [X Y Z 1]
A = np.column_stack((X, Y, Z, np.ones_like(X)))

# Ma trận B = [Z*u, Z*v, Z]
B = np.column_stack((Z*u, Z*v, Z))

# Giải linear least squares: A x H_T ≈ B
# H_T có kích thước 4x3, nên H = H_T.T là 3x4 (chuyển vị H)
H_T, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
H = H_T.T

print("Ma trận H tìm được:")
print(H)

# Tính lại [Zu, Zv, Z] từ H
B_calc = A @ H_T  

Zu_calc = B_calc[:, 0]
Zv_calc = B_calc[:, 1]
Z_calc  = B_calc[:, 2]

# Suy ra u, v tính lại
u_calc = Zu_calc / Z_calc
v_calc = Zv_calc / Z_calc

# Sai số
du = u_calc - u
dv = v_calc - v
err = np.sqrt(du**2 + dv**2)

# Bảng so sánh
df = pd.DataFrame({
    "u_goc": u.astype(int),
    "v_goc": v.astype(int),
    "u_tinh": np.round(u_calc, 3),
    "v_tinh": np.round(v_calc, 3),
    "delta_u": np.round(du, 3),
    "delta_v": np.round(dv, 3),
    "err": np.round(err, 3)
})

print("\nBảng so sánh:")
print(df.to_string(index=False))

print("\nSai số trung bình |delta_u| =", np.mean(np.abs(du)))
print("Sai số trung bình |delta_v| =", np.mean(np.abs(dv)))
print("Sai số Euclid trung bình   =", np.mean(err))
