def _dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _segment_point_dist2(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return _dist2(p, a)

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))

    cx = ax + t * abx
    cy = ay + t * aby
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy


class Planner:
    """
    지금 단계: 직선(로봇->목표)만 만들고,
    장애물 원(반경 r)과 선분이 충돌하면 경로 None.
    """
    def plan(self, state):
        if state.robot_xy is None or state.goal_xy is None:
            return None

        s = state.robot_xy
        g = state.goal_xy

        r = float(state.obstacle_radius)
        r2 = r * r

        for c in state.obstacles_xy:
            if _segment_point_dist2(c, s, g) <= r2:
                return None

        return [s, g]