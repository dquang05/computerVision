import numpy as np

# =========================
# Ma trận A (đã chuẩn hóa 0-1)
# =========================
A = np.array([
[0,0,0,0,0,0,0],
[0,0,1,1,1,0,0],
[0,1,1,1,1,1,0],
[0,0,1,0,1,0,0],
[0,0,1,0,1,0,0],
[0,0,1,1,1,0,0],
[0,0,0,0,0,0,0]
])

# =========================
# Kernel B1 và B2
# =========================
B1 = np.array([
[1,0,1],
[0,0,0],
[0,0,0]
])

B2 = np.array([
[0,0,0],
[0,0,0],
[0,1,0]
])

# =========================
# Hàm erosion (duplicate border)
# =========================
def erosion(A, B):
    m, n = A.shape
    km, kn = B.shape
    pad_h = km // 2
    pad_w = kn // 2

    out = np.zeros_like(A)

    for i in range(m):
        for j in range(n):
            ok = True
            for u in range(km):
                for v in range(kn):
                    if B[u, v] == 1:
                        ii = i + u - pad_h
                        jj = j + v - pad_w

                        # duplicate border
                        ii = min(max(ii, 0), m-1)
                        jj = min(max(jj, 0), n-1)

                        if A[ii, jj] != 1:
                            ok = False
                            break
                if not ok:
                    break
            out[i, j] = 1 if ok else 0

    return out

# =========================
# Tính Hit-or-Miss
# =========================
E1 = erosion(A, B1)

Ac = 1 - A  # ảnh bù
E2 = erosion(Ac, B2)

H = E1 & E2

# =========================
# In kết quả
# =========================
print("\nE1 = A erosion B1 =")
print(E1*255)

print("\nE2 = Ac erosion B2 =")
print(E2*255)

print("\nH =")
print(H*255)
