import math
from typing import List, Tuple

XY = Tuple[float, float]


def regular_ngon(center: XY, radius: float, n: int, start_angle_rad: float = 0.0) -> List[XY]:
    """
    Regular n-gon whose vertices lie on a circle of given radius (circumradius = radius).
    꼭짓점이 원 위에 닿는 방식(A).
    """
    cx, cy = center
    pts: List[XY] = []
    for k in range(n):
        th = start_angle_rad + 2.0 * math.pi * (k / n)
        pts.append((cx + radius * math.cos(th), cy + radius * math.sin(th)))
    return pts


def obstacles_as_polygons(obstacles_xy: List[XY], radius: float, n: int) -> List[List[XY]]:
    """센터 리스트 -> 각 센터당 n각형 폴리곤 리스트"""
    return [regular_ngon(c, radius, n=n) for c in obstacles_xy]


def _dist2(a: XY, b: XY) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def cluster_centers(centers: List[XY], merge_dist: float) -> List[List[XY]]:
    """
    merge_dist 이내면 같은 그룹으로 묶기(BFS).
    겹치거나 가까운 장애물들을 "클러스터"로 만들기.
    """
    if not centers:
        return []

    md2 = float(merge_dist) * float(merge_dist)
    used = [False] * len(centers)
    groups: List[List[XY]] = []

    for i in range(len(centers)):
        if used[i]:
            continue
        used[i] = True
        q = [i]
        group = [centers[i]]

        while q:
            k = q.pop()
            for j in range(len(centers)):
                if used[j]:
                    continue
                if _dist2(centers[k], centers[j]) <= md2:
                    used[j] = True
                    q.append(j)
                    group.append(centers[j])

        groups.append(group)

    return groups


def convex_hull(points: List[XY]) -> List[XY]:
    """
    Monotonic chain convex hull (CCW).
    점들을 둘러싸는 가장 바깥쪽 볼록 다각형.
    """
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o: XY, a: XY, b: XY) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[XY] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[XY] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def merged_obstacles_as_hulls(
    obstacles_xy: List[XY],
    radius: float,
    n_vertices: int,
    merge_dist: float,
) -> List[List[XY]]:
    """
    ✅ 핵심: 겹치는 장애물들을 "센터 1개"로 합치지 말고,
    그 클러스터가 커버하는 영역을 "폴리곤 합집합 근사"로 만든다.

    방법:
    - 각 원을 n각형으로 샘플링(regular_ngon)
    - 같은 클러스터의 샘플 점들을 모아서 convex hull
    => 두 원(두 장애물) 범위를 둘 다 커버하는 하나의 폴리곤이 됨(보수적).
    """
    groups = cluster_centers(obstacles_xy, merge_dist)
    merged_polys: List[List[XY]] = []

    for group in groups:
        sample_pts: List[XY] = []
        for c in group:
            sample_pts.extend(regular_ngon(c, radius, n=n_vertices))

        hull = convex_hull(sample_pts)

        # hull이 선/점이면 fallback
        if len(hull) < 3:
            hull = sample_pts[:]

        merged_polys.append(hull)

    return merged_polys