from dataclasses import dataclass, field
from typing import List, Optional, Tuple

XY = Tuple[float, float]


@dataclass
class PlanningState:
    robot_xy: Optional[XY] = None
    goal_xy: Optional[XY] = None

    obstacles_xy: List[XY] = field(default_factory=list)
    obstacle_radius: float = 0.30

    frame_id: str = "map"

    def ready(self) -> bool:
        return (self.robot_xy is not None) and (self.goal_xy is not None)

    def set_robot(self, x: float, y: float):
        self.robot_xy = (float(x), float(y))

    def set_goal(self, x: float, y: float):
        self.goal_xy = (float(x), float(y))

    def add_obstacle(self, x: float, y: float):
        self.obstacles_xy.append((float(x), float(y)))

    def clear_obstacles(self):
        self.obstacles_xy.clear()

    def clear_all(self):
        self.robot_xy = None
        self.goal_xy = None
        self.obstacles_xy.clear()