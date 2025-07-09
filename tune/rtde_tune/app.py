import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


shoulder_pan_file = (
    "/home/min/7cmdehdrb/project_th/tune/rtde_tune/data/rtde_log_shoulder_pan.csv"
)
shoulder_lift_file = (
    "/home/min/7cmdehdrb/project_th/tune/rtde_tune/data/rtde_log_shoulder_lift.csv"
)


shoulder_lift_data = pd.read_csv(shoulder_lift_file)[
    ["Damping", "Stiffness", "R_shoulder_lift_p", "S_shoulder_lift_p"]
]

damping = shoulder_lift_data["Damping"]
stiffness = shoulder_lift_data["Stiffness"]

real_position = shoulder_lift_data["R_shoulder_lift_p"]
sim_position = shoulder_lift_data["S_shoulder_lift_p"]

res = []

for d in np.unique(damping):
    for s in np.unique(stiffness):
        mask = (shoulder_lift_data["Damping"] == d) & (
            shoulder_lift_data["Stiffness"] == s
        )

        real_pos = shoulder_lift_data.loc[mask, "R_shoulder_lift_p"].values
        sim_pos = shoulder_lift_data.loc[mask, "S_shoulder_lift_p"].values

        pos_error = np.abs(real_pos - sim_pos).mean()

        res.append(
            {
                "Damping": d,
                "Stiffness": s,
                "Error": pos_error,
            }
        )

df = pd.DataFrame(res)


from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from scipy.interpolate import LinearNDInterpolator

# df = df[df["Error"] < 0.1]

x = df["Damping"].values
y = df["Stiffness"].values
z = df["Error"].values

# Create a linear interpolator function for the scattered data
linear_interp = LinearNDInterpolator(list(zip(x, y)), z)

# 보간을 위한 그리드 생성
xi = np.linspace(np.min(x), np.max(x), 150)
yi = np.linspace(np.min(y), np.max(y), 150)
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

# ax1.scatter(x, y, z, color="black", s=10, alpha=0.5, label="Grid Points")

# 2. 등고선 플롯
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(xi, yi, zi, levels=50, cmap="viridis")
ax2.scatter(min_x, min_y, color="red", s=50, label=f"Min z = {min_z:.3f}")
fig.colorbar(contour, ax=ax2, label="ShoulderPan")
ax2.set_xlabel("Damping")
ax2.set_ylabel("Stiffness")
ax2.set_title("Contour Plot")
ax2.legend()

# ax2.scatter(x, y, color="black", s=10, alpha=0.5, label="Grid Points")

plt.tight_layout()
plt.show()
