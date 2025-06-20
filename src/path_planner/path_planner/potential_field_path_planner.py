# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *

# TF
from tf2_ros import *

# Python
from typing import List, Tuple
from collections import deque
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt


"""

Potential Field based path planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

"""


class PotentialPoint(object):
    def __init__(self, x, y):
        """
        Initialize a point in the potential field.

        Parameters:
        - x: x-coordinate of the point
        - y: y-coordinate of the point
        """
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


class PotentialObstacle(object):
    def __init__(self, x, y, r):
        """
        Initialize an obstacle in the potential field.

        Parameters:
        - x: x-coordinate of the obstacle
        - y: y-coordinate of the obstacle
        - r: radius of the obstacle
        """
        self._x = x
        self._y = y
        self._r = r

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def r(self):
        return self._r

    @property
    def xs(self):
        return [pp.x for pp in self._obstacle()]

    @property
    def ys(self):
        return [pp.y for pp in self._obstacle()]

    def _obstacle(self) -> List[PotentialPoint]:
        pps = []

        for theta in np.linspace(0, 2 * np.pi, 5):
            pps.append(
                PotentialPoint(
                    self._x + self._r * np.cos(theta),
                    self._y + self._r * np.sin(theta),
                )
            )

        pps.append(
            PotentialPoint(self._x, self._y)
        )  # Add a point to close the obstacle

        return pps


class PotentialFieldPlanner:
    def __init__(
        self,
        rr: float,
        resolution: float,
        kp: float = 5.0,
        eta: float = 100.0,
        area_offset: float = 15.0,
        oscillations: int = 3,
    ):
        """
        Parameters:
        (REQUIRED)
        - rr: radius of the robot [m]
        - resolution: grid resolution [m]
        - kp: attractive potential gain
        - eta: repulsive potential gain
        (OPTIONAL)
        - area_offset: potential area offset distance [m]. Default is 15.0.
        - oscillations: the number of previous positions used to check oscillations. Default is 3.
        """

        self._rr = rr
        self._resolution = resolution
        self._kp = kp
        self._eta = eta
        self._area_offset = area_offset
        self._oscillations = oscillations

        self._pmap: np.ndarray = None  # Potential field map

    # >>> Potential Field Map Generation >>>
    def _get_potential_map(
        self,
        start: PotentialPoint,
        goal: PotentialPoint,
        obstacles: List[PotentialObstacle],
    ) -> Tuple[np.ndarray, float, float]:
        """
        Generate a potential field map based on the start and goal positions, and the obstacles.

        Parameters:
        - start: Starting point of the potential field.
        - goal: Goal point of the potential field.
        - obstacles: List of obstacles in the potential field.

        Returns:
        - pmap: 2D numpy array representing the potential field.
        - xw: Width of the potential field in grid cells.
        - yw: Height of the potential field in grid cells.
        - minx: Minimum x-coordinate of the potential field area.
        - miny: Minimum y-coordinate of the potential field area.
        """
        ox = [x for obs in obstacles for x in obs.xs]
        oy = [y for obs in obstacles for y in obs.ys]

        minx = min(min(ox), start.x, goal.x) - self._area_offset
        miny = min(min(oy), start.y, goal.y) - self._area_offset
        maxx = max(max(ox), start.x, goal.x) + self._area_offset
        maxy = max(max(oy), start.y, goal.y) + self._area_offset

        xw = int(round((maxx - minx) / self._resolution))
        yw = int(round((maxy - miny) / self._resolution))

        # calc each potential
        pmap = np.zeros((xw, yw), dtype=float)

        return pmap, xw, yw, minx, miny

    def _calc_potential_field(
        self,
        start: PotentialPoint,
        goal: PotentialPoint,
        obstacles: List[PotentialObstacle],
    ) -> Tuple[np.ndarray, float, float]:
        """
        목표 위치로 끌어당기는 힘을 거리 기반으로 계산

        Parameters:
        - start: Starting point of the potential field.
        - goal: Goal point of the potential field.
        - obstacles: List of obstacles in the potential field.

        Returns:
        - pmap: 2D numpy array representing the potential field.
        - minx: Minimum x-coordinate of the potential field area.
        - miny: Minimum y-coordinate of the potential field area.
        """

        pmap, xw, yw, minx, miny = self._get_potential_map(start, goal, obstacles)

        for ix in range(xw):
            x = ix * self._resolution + minx

            for iy in range(yw):
                y = iy * self._resolution + miny

                pmap[ix][iy] = self._calc_potential(x, y, goal, obstacles)

        # 잠재장 맵 pmap, 맵의 왼쪽 하단 좌표 minx, miny
        return pmap, minx, miny

    # <<< Potential Field Map Generation <<<

    # >>> Potential Field Calculation >>>
    def _calc_attractive_potential(self, x, y, goal: PotentialPoint) -> float:
        return 0.5 * self._kp * np.hypot(x - goal.x, y - goal.y)

    def _calc_repulsive_potential(
        self, x, y, obstacles: List[PotentialObstacle]
    ) -> float:
        ox = [x for obs in obstacles for x in obs.xs]
        oy = [y for obs in obstacles for y in obs.ys]

        # search nearest obstacle
        minid = -1
        dmin = float("inf")
        for i, _ in enumerate(ox):
            d = np.hypot(x - ox[i], y - oy[i])
            if dmin >= d:
                dmin = d
                minid = i

        # calc repulsive potential
        dq = np.hypot(x - ox[minid], y - oy[minid])

        if dq <= self._rr:
            if dq <= 0.1:
                dq = 0.1

            return 0.5 * self._eta * (1.0 / dq - 1.0 / self._rr) ** 2
        else:
            return 0.0

    def _calc_potential(
        self, x, y, goal: PotentialPoint, obstacles: List[PotentialObstacle]
    ) -> float:
        """
        Calculate the potential at a given point (x, y) based on the goal and obstacles.

        Parameters:
        - x: x-coordinate of the point.
        - y: y-coordinate of the point.
        - goal: Goal point of the potential field.
        - obstacles: List of obstacles in the potential field.

        Returns:
        - Potential value at the point (x, y).
        """
        ug = self._calc_attractive_potential(x, y, goal)
        uo = self._calc_repulsive_potential(x, y, obstacles)

        return ug + uo

    # <<< Potential Field Calculation <<<

    def _get_motion_model(self) -> List[List[int]]:
        """
        Get the motion model for the potential field planner.
        """

        motion = [[1, 0], [0, 1], [-1, 0], [0, -1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
        return motion

    def _oscillations_detection(self, previous_ids: deque, ix, iy) -> bool:
        previous_ids.append((ix, iy))

        if len(previous_ids) > self._oscillations:
            previous_ids.popleft()

        # check if contains any duplicates by copying into a set
        previous_ids_set = set()

        for index in previous_ids:
            if index in previous_ids_set:
                return True
            else:
                previous_ids_set.add(index)

        return False

    # >>> Path Planning >>>
    def _bezier_spline(
        self, path: np.ndarray, s: float = 0.001, n: float = 500
    ) -> np.ndarray:
        """
        Generate a Bezier spline for the given path.

        Parameters:
        - path: A 2D numpy array of shape (N, 2) representing the path points.
        - s: Smoothing factor for the spline. Default is 0.001.
        - n: Number of interpolation points. Default is 500.

        Returns:
        - splined_path: A 2D numpy array of shape (n, 2) representing the smoothed path.
        """

        tck, _ = splprep(path.T, s=0.001)  # s: smoothing factor
        u_fine = np.linspace(0, 1, n)  # 보간 점 수 조절

        x_smooth, y_smooth = splev(u_fine, tck)

        splined_path = np.vstack((x_smooth, y_smooth)).T

        return splined_path

    def planning(
        self,
        start: PotentialPoint,
        goal: PotentialPoint,
        obstacles: List[PotentialObstacle],
    ):

        # calc potential field
        pmap, minx, miny = self._calc_potential_field(start, goal, obstacles)

        # search path
        d = np.hypot(start.x - goal.x, start.y - goal.y)
        ix = round((start.x - minx) / self._resolution)
        iy = round((start.y - miny) / self._resolution)

        rx, ry = [start.x], [start.y]
        motion = self._get_motion_model()
        previous_ids = deque()

        while d >= self._resolution:
            minp = float("inf")
            minix, miniy = -1, -1

            for i, _ in enumerate(motion):
                inx = int(ix + motion[i][0])
                iny = int(iy + motion[i][1])

                if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
                    p = float("inf")  # outside area
                    print("outside potential!")

                else:
                    p = pmap[inx][iny]

                if minp > p:
                    minp = p
                    minix = inx
                    miniy = iny

            ix = minix
            iy = miniy

            xp = ix * self._resolution + minx
            yp = iy * self._resolution + miny

            d = np.hypot(goal.x - xp, goal.y - yp)

            rx.append(xp)
            ry.append(yp)

            if self._oscillations_detection(previous_ids, ix, iy):
                print(f"Oscillation detected at ({ix},{iy})")
                break

        path = np.vstack((np.array(rx), np.array(ry))).T

        # Bezier spline smoothing
        path = self._bezier_spline(path)

        return path

    # <<< Path Planning <<<


def main2():
    pfp = PotentialFieldPlanner(
        rr=0.08,  # robot radius [m]
        resolution=0.02,  # grid resolution [m]
        kp=1.0,  # attractive potential gain
        eta=100.0,  # repulsive potential gain
        area_offset=1.0,  # potential area width [m]
        oscillations=3,  # number of previous positions to check for oscillations
    )

    start = PotentialPoint(0.13, 0.21)
    goal = PotentialPoint(x=-0.3, y=0.6)  # Example goal point

    obstacles = [
        # PotentialObstacle(x=0.0, y=0.75, r=0.03),
        # PotentialObstacle(x=0.2, y=0.75, r=0.03),
        # PotentialObstacle(x=-0.2, y=0.75, r=0.03),
        # PotentialObstacle(x=0.0, y=0.6, r=0.03),
        # PotentialObstacle(x=0.2, y=0.6, r=0.03),
        PotentialObstacle(x=-0.2, y=0.6, r=0.03),
        # PotentialObstacle(x=-0.4, y=0.6),
    ]

    import time

    start_time = time.time()
    print("Planning Start!")

    # path generation
    path = pfp.planning(start, goal, obstacles)

    end_time = time.time()
    dt = end_time - start_time

    print(f"Path found! Planning Time: {dt:.2f} seconds")

    ox = [x for obs in obstacles for x in obs.xs]
    oy = [y for obs in obstacles for y in obs.ys]

    # 시각화
    plt.figure(figsize=(8, 8))

    plt.scatter(start.x, start.y, c="green", marker="o", label="Start")
    plt.scatter(goal.x, goal.y, c="black", marker="x", label="Goal")
    plt.scatter(ox, oy, c="red", s=100, label="Obstacles")

    plt.plot(
        path[:, 0],
        path[:, 1],
        "-",
        c="blue",
        label="Path",
        markersize=3,
        alpha=0.8,
    )

    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.show()


if __name__ == "__main__":
    print(__file__ + " start!!")
    main2()
    print(__file__ + " Done!!")
