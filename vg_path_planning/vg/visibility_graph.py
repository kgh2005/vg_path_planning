from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from vg_path_planning.vg.geometry import segment_intersects_poly
from vg_path_planning.vg.shortest_path import astar_path

XY = Tuple[float, float]


@dataclass
class VGData:
    nodes: List[XY]
    start_idx: int
    goal_idx: int
    obstacle_polys: List[List[XY]]
    node_poly_id: List[int]   # node가 어떤 poly에 속하는지(없으면 -1)


def build_vg_nodes_from_polys(
    robot_xy: Optional[XY],
    goal_xy: Optional[XY],
    polys: List[List[XY]],
) -> Optional[VGData]:
    """
    폴리곤을 직접 받아서 VG 노드 리스트 구성
    (start, goal, 그 다음 모든 polygon vertices)
    """
    if robot_xy is None or goal_xy is None:
        return None

    nodes: List[XY] = [robot_xy, goal_xy]
    node_poly_id: List[int] = [-1, -1]

    for pid, poly in enumerate(polys):
        nodes.extend(poly)
        node_poly_id.extend([pid] * len(poly))

    return VGData(
        nodes=nodes,
        start_idx=0,
        goal_idx=1,
        obstacle_polys=polys,
        node_poly_id=node_poly_id,
    )


def _is_adjacent_on_polygon(i_local: int, j_local: int, m: int) -> bool:
    """polygon vertex index i_local, j_local are adjacent on m-gon?"""
    if m <= 2:
        return False
    d = abs(i_local - j_local)
    return d == 1 or d == (m - 1)


def build_vg_edges(vg: VGData) -> List[Tuple[int, int]]:
    """
    Step 2: visibility edges.

    Rule:
    - 선분(i,j)이 어떤 obstacle polygon과 교차/관통하면 edge 불가
    - 단, 같은 polygon의 인접 꼭짓점끼리(경계 edge)는 허용
    """
    edges: List[Tuple[int, int]] = []
    nodes = vg.nodes
    polys = vg.obstacle_polys
    poly_id = vg.node_poly_id

    # 각 polygon의 전역 시작 인덱스(로컬 index 계산용)
    poly_start_idx: Dict[int, int] = {}
    idx = 2  # start, goal 다음부터 polygon vertices
    for pid, poly in enumerate(polys):
        poly_start_idx[pid] = idx
        idx += len(poly)

    n = len(nodes)

    for i in range(n):
        for j in range(i + 1, n):
            a = nodes[i]
            b = nodes[j]

            pi = poly_id[i]
            pj = poly_id[j]

            # 1) 같은 폴리곤 내부: 인접 변만 허용
            if pi != -1 and pi == pj:
                m = len(polys[pi])
                si = poly_start_idx[pi]
                i_local = i - si
                j_local = j - si
                if _is_adjacent_on_polygon(i_local, j_local, m):
                    edges.append((i, j))
                continue

            # 2) 다른 경우: 어떤 폴리곤과라도 교차하면 blocked
            blocked = False
            for poly in polys:
                if segment_intersects_poly(a, b, poly):
                    blocked = True
                    break

            if not blocked:
                edges.append((i, j))

    return edges


def build_vg_path(vg: VGData, edges: List[Tuple[int, int]]):
    """
    A*로 start_idx -> goal_idx 인덱스 경로를 구한 뒤,
    실제 좌표로 변환해서 반환
    """
    idx_path = astar_path(vg.nodes, edges, vg.start_idx, vg.goal_idx)
    if not idx_path or len(idx_path) < 2:
        return None
    return [vg.nodes[i] for i in idx_path]