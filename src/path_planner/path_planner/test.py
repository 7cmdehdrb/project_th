import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────
# 1) 기존에 이미 그려진 plot/scatter가 있다고 가정
#    예시: (실제 코드에 이미 들어가 있다고 보세요)
fig, ax = plt.subplots(figsize=(6, 4))
# 예: 뭔가 선을 그리고 싶다면
x_line = np.linspace(-0.5, 0.5, 200)
y_line = np.sin(2 * np.pi * x_line) * 0.5 + 0.45
ax.plot(x_line, y_line, color="k", lw=1)

# 예: scatter도 하나 찍어본다
x_s = np.random.uniform(-0.5, 0.5, 50)
y_s = np.random.uniform(-0.1, 1.0, 50)
ax.scatter(x_s, y_s, c="C1", s=20, label="points")

# 축 한계를 명확히 설정
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.1, 1.0)
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 2) 히트맵(컬러 맵)을 덧씌우기 위한 그리드 생성
#
#    - x_min, x_max, y_min, y_max 에 맞춰서 2D 그리드 생성
#    - 해상도(샘플링 개수)는 필요에 따라 조절하세요 (여기선 100×100)
#
x_min, x_max = -0.5, 0.5
y_min, y_max = -0.1, 1.0

nx, ny = 100, 100
x_edges = np.linspace(x_min, x_max, nx)
y_edges = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x_edges, y_edges)  # X.shape = (ny, nx), Y.shape = (ny, nx)

# 3) Z 값을 정의 (여기에 올리실 실제 히트맵 데이터 객체를 넣으시면 됩니다)
#    예시로 중앙 근처에 가우시안 형태 열 강도(simulated) 만들어 보겠습니다.
sigma = 0.2
x0, y0 = 0.2, 0.5
Z = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))

print(f"Z.shape: {Z.shape}")  # Z.shape = (ny, nx)

# 4) pcolormesh 를 이용해 히트맵 오버레이
#    - alpha를 0~1 사이 값으로 주면, 아래에 이미 그려진 plot/scatter가 반투명으로 보입니다.
#    - shading='auto' 옵션을 주면, X, Y, Z 형상에 따라 경계 처리 오류를 방지합니다.
heatmap = ax.pcolormesh(X, Y, Z, cmap="viridis", alpha=0.4, shading="auto")

# 5) 컬러바 (옵션)
cbar = fig.colorbar(heatmap, ax=ax)
cbar.set_label("heat intensity")

# 6) 레이블, 제목 등 (필요시)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("기존 Plot/Scatter 위에 히트맵 Overlay 예시")

plt.legend()
plt.show()
