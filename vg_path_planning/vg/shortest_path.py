import heapq
from typing import Dict, List, Optional, Tuple

XY = Tuple[float, float]


def _dist(a: XY, b: XY) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def astar_path(nodes: List[XY],
               edges: List[Tuple[int, int]],
               start_idx: int,
               goal_idx: int) -> Optional[List[int]]:
    """
    A* on an undirected weighted graph.
    Returns a list of node indices [start ... goal], or None if no path.
    """
    if start_idx == goal_idx:
        return [start_idx]

    # adjacency
    adj: Dict[int, List[int]] = {i: [] for i in range(len(nodes))}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    def h(n: int) -> float:
        return _dist(nodes[n], nodes[goal_idx])

    g_cost: Dict[int, float] = {start_idx: 0.0}
    parent: Dict[int, int] = {}

    pq = []
    heapq.heappush(pq, (h(start_idx), start_idx))

    closed = set()

    while pq:
        f, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        if cur == goal_idx:
            # reconstruct
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return path

        closed.add(cur)

        for nb in adj[cur]:
            tentative = g_cost[cur] + _dist(nodes[cur], nodes[nb])
            if nb not in g_cost or tentative < g_cost[nb]:
                g_cost[nb] = tentative
                parent[nb] = cur
                heapq.heappush(pq, (tentative + h(nb), nb))

    return None