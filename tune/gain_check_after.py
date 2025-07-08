import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

merged_csv = "/home/min/7cmdehdrb/project_th/gain_tuner_merged102.csv"
merged_df = pd.read_csv(merged_csv)

merged_df = merged_df[(merged_df["ShoulderLift"] < 0.05)]

# merged_df = merged_df[(merged_df["damping"] >= 20.0)]
# merged_df = merged_df[(merged_df["stiffness"] >= 600.0)]

from mpl_toolkits.mplot3d import Axes3D

"""
ShoulderPan
ShoulderLift
Elbow
Wrist1
Wrist2
Wrist3
"""

# damping = merged_df["damping"].values
# stiffness = merged_df["stiffness"].values
# shoulder_pan = merged_df["ShoulderPan"].values
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection="3d")
# # ax.scatter(damping, stiffness, shoulder_pan, c="r", marker="o")
# ax.plot_trisurf(damping, stiffness, shoulder_pan, cmap="viridis", edgecolor="none")
# ax.set_xlabel("Damping")
# ax.set_ylabel("Stiffness")
# ax.set_zlabel("Error")
# ax.set_title("3D Plot of Error vs Damping and Stiffness")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 데이터 준비
x = merged_df["damping"].values
y = merged_df["stiffness"].values

print(x)
print(y)

z = merged_df["Elbow"].values

# 보간을 위한 그리드 생성
xi = np.linspace(np.min(x), np.max(x), 100)
yi = np.linspace(np.min(y), np.max(y), 100)
xi, yi = np.meshgrid(xi, yi)

# z 값 보간 (cubic 보간)
zi = griddata((x, y), z, (xi, yi), method="cubic")

# 최소값 위치 찾기
min_idx = np.unravel_index(np.nanargmin(zi), zi.shape)
min_x, min_y, min_z = xi[min_idx], yi[min_idx], zi[min_idx]

# 시각화
fig = plt.figure(figsize=(16, 6))

# 1. 3D 서피스 플롯
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
surf = ax1.plot_surface(xi, yi, zi, cmap="viridis", edgecolor="none", alpha=0.9)
ax1.scatter(min_x, min_y, min_z, color="red", s=50, label="Minimum")
ax1.set_xlabel("Damping")
ax1.set_ylabel("Stiffness")
ax1.set_zlabel("ShoulderPan")
ax1.set_title("3D Surface Plot")
ax1.legend()

# 2. 등고선 플롯
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(xi, yi, zi, levels=50, cmap="viridis")
ax2.scatter(min_x, min_y, color="red", s=50, label=f"Min z = {min_z:.3f}")
fig.colorbar(contour, ax=ax2, label="ShoulderPan")
ax2.set_xlabel("Damping")
ax2.set_ylabel("Stiffness")
ax2.set_title("Contour Plot")
ax2.legend()

plt.tight_layout()
plt.show()
