import math
from typing import List, Tuple

XY = Tuple[float, float]


def regular_ngon(center: XY, radius: float, n: int, start_angle_rad: float = 0.0) -> List[XY]:
    """
    Regular n-gon whose vertices lie on a circle of given radius (circumradius = radius).
    => 꼭짓점이 원 위에 닿는 방식(A).
    """
    cx, cy = center
    pts: List[XY] = []
    for k in range(n):
        th = start_angle_rad + 2.0 * math.pi * (k / n)
        pts.append((cx + radius * math.cos(th), cy + radius * math.sin(th)))
    return pts


def obstacles_as_polygons(obstacles_xy: List[XY], radius: float, n: int) -> List[List[XY]]:
    return [regular_ngon(c, radius, n=n) for c in obstacles_xy]