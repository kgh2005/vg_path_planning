# vg_path_planning

A ROS 2 (Humble) playground to prototype **Visibility Graph (VG)** path planning with **RViz** interaction (click robot/goal/obstacles and visualize markers/path).

## Development Environment

| Component | Version |
|---|---|
| **OS** | Ubuntu 22.04 |
| **ROS** | Humble Hawksbill |

## Tree
```bash
vg_path_planning/
├── config/
│   ├── rviz.rviz
│   └── params.yaml
├── launch/
│   └── vg_path_planning.launch.py
└── vg_path_planning/
├── core/
│   ├── fsm.py
│   ├── planner.py
│   └── state.py
├── nodes/
│   └── vg_rviz.py
└── vg/
    ├── geometry.py
    ├── obstacles.py
    ├── shortest_path.py
    └── visibility_graph.py
```