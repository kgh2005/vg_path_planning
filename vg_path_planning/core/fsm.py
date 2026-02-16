from enum import Enum, auto


class Phase(Enum):
    IDLE = auto()
    HAVE_ROBOT = auto()
    HAVE_GOAL = auto()
    READY = auto()
    PLANNING = auto()


class PlannerFSM:
    def __init__(self, state, planner, on_path=None, logger=None):
        self.state = state
        self.planner = planner
        self.on_path = on_path
        self.logger = logger
        self.phase = Phase.IDLE

    def _log(self, s: str):
        if self.logger:
            self.logger.info(s)

    def _recompute_phase(self):
        r = self.state.robot_xy is not None
        g = self.state.goal_xy is not None
        if r and g:
            self.phase = Phase.READY
        elif r:
            self.phase = Phase.HAVE_ROBOT
        elif g:
            self.phase = Phase.HAVE_GOAL
        else:
            self.phase = Phase.IDLE

    def _plan_if_ready(self):
        if not self.state.ready():
            return

        self.phase = Phase.PLANNING
        self._log("[fsm] planning...")
        path = self.planner.plan(self.state)
        self._log(f"[fsm] plan done. path_len={0 if path is None else len(path)}")
        if self.on_path:
            self.on_path(path)
        self.phase = Phase.READY

    def set_robot(self, x, y):
        self.state.set_robot(x, y)
        self._recompute_phase()
        self._log(f"[fsm] set_robot -> {self.phase.name}")
        self._plan_if_ready()

    def set_goal(self, x, y):
        self.state.set_goal(x, y)
        self._recompute_phase()
        self._log(f"[fsm] set_goal -> {self.phase.name}")
        self._plan_if_ready()

    def add_obstacle(self, x, y):
        self.state.add_obstacle(x, y)
        self._recompute_phase()
        self._log(f"[fsm] add_obstacle(n={len(self.state.obstacles_xy)}) -> {self.phase.name}")
        self._plan_if_ready()

    def clear_obstacles(self):
        self.state.clear_obstacles()
        self._recompute_phase()
        self._log("[fsm] clear_obstacles")
        self._plan_if_ready()

    def clear_all(self):
        self.state.clear_all()
        self._recompute_phase()
        self._log("[fsm] clear_all")