import numpy as np


def generate_parabolic_path(x_start, y_start, num_points=100):
    """
    x, y: 시작점 좌표
    num_points: 보간 점 개수
    x_offset: x의 최대 변위 (포물선 폭)
    """
    x_offset = 0.02

    # y 값 보간 (선형)
    y_end = y_start - 0.2
    y_values = np.linspace(y_start, y_end, num_points)

    # 포물선 중심 y (x 최대값을 가지는 지점)
    y_mid = (y_start + y_end) / 2.0

    # 포물선 계수 a 계산: x = -a * (y - y_mid)^2 + x + x_offset
    a = x_offset / ((y_start - y_mid) ** 2)

    # x 값 생성 (포물선)
    x_values = -a * (y_values - y_mid) ** 2 + x_start + x_offset

    # path 생성
    path = np.vstack((x_values, y_values)).T

    # 방향 벡터 계산
    directions = np.zeros_like(path)
    directions[1:-1] = path[2:] - path[:-2]
    directions[0] = path[1] - path[0]
    directions[-1] = path[-1] - path[-2]

    # 단위 벡터화
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms

    return path, directions


x = 0.2
y = -0.2

path, directions = generate_parabolic_path(x, y)

print("Path:\n", path)
print("Directions:\n", directions)
