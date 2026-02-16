# vg_path_planning/vg/geometry.py
from typing import List, Tuple

XY = Tuple[float, float]


def _orient(a: XY, b: XY, c: XY) -> float:
    # cross((b-a),(c-a))
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: XY, b: XY, c: XY, d: XY, eps: float = 1e-9) -> bool:
    """
    '진짜 교차'만 잡는 아주 단순한 버전.
    - 끝점이 닿는 경우(접촉), 공선(collinear) 겹침은 교차로 보지 않음.
    - 즉, 선분이 서로 '가로질러' 지나갈 때만 True.
    """
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    # strict intersection only
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def segment_intersects_poly(a: XY, b: XY, poly: List[XY]) -> bool:
    """
    매우 단순한 충돌 판정:
    - 선분(a-b)이 폴리곤의 어떤 변과 '진짜 교차'하면 blocked(True)
    - a/b가 폴리곤 내부인지 여부는 여기서 보지 않음(단순화)
    """
    n = len(poly)
    for i in range(n):
        c = poly[i]
        d = poly[(i + 1) % n]
        if segments_intersect(a, b, c, d):
            return True
    return False