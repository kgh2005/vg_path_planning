import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped as PoseStampedMsg

from vg_path_planning.core.state import PlanningState
from vg_path_planning.core.planner import Planner
from vg_path_planning.core.fsm import PlannerFSM
from vg_path_planning.vg.obstacles import obstacles_as_polygons


class VgRvizNode(Node):
    def __init__(self):
        super().__init__("vg_rviz")

        # ---- parameters ----
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("field_width", 20.0)    # meters
        self.declare_parameter("field_height", 14.0)   # meters
        self.declare_parameter("obstacle_radius", 0.30)

        self.declare_parameter("initialpose_topic", "/initialpose")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("obstacle_topic", "/clicked_point")

        self.declare_parameter("markers_topic", "/vg/markers")
        self.declare_parameter("path_topic", "/vg/path")

        self.declare_parameter("square", 8)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.field_width = float(self.get_parameter("field_width").value)
        self.field_height = float(self.get_parameter("field_height").value)
        self.square = int(self.get_parameter("square").value)

        radius = float(self.get_parameter("obstacle_radius").value)
        self.state = PlanningState(obstacle_radius=radius, frame_id=self.frame_id)

        self.planner = Planner()
        self._latest_path_xy = None

        self.fsm = PlannerFSM(
            state=self.state,
            planner=self.planner,
            logger=self.get_logger(),
            on_path=self._on_path,
        )

        # ---- subscribers (RViz tools) ----
        self.create_subscription(
            PoseWithCovarianceStamped,
            str(self.get_parameter("initialpose_topic").value),
            self._cb_initialpose,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("goal_topic").value),
            self._cb_goal,
            10,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter("obstacle_topic").value),
            self._cb_obstacle,
            10,
        )

        # ---- publishers ----
        self.pub_markers = self.create_publisher(
            MarkerArray, str(self.get_parameter("markers_topic").value), 10
        )
        self.pub_path = self.create_publisher(
            Path, str(self.get_parameter("path_topic").value), 10
        )

        # publish markers periodically
        self.create_timer(0.1, self._publish_markers)

        self.get_logger().info("[vg_rviz] ready (RViz tools: /initialpose, /goal_pose, /clicked_point)")

    # -------- input callbacks --------
    def _cb_initialpose(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.fsm.set_robot(x, y)

    def _cb_goal(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.fsm.set_goal(x, y)

    def _cb_obstacle(self, msg: PointStamped):
        x = msg.point.x
        y = msg.point.y
        self.fsm.add_obstacle(x, y)

    # -------- planner result --------
    def _on_path(self, path_xy):
        self._latest_path_xy = path_xy
        self._publish_path()

    # -------- publishers --------
    def _publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = self.frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()

        if not self._latest_path_xy or len(self._latest_path_xy) < 2:
            self.pub_path.publish(path_msg)
            return

        for (x, y) in self._latest_path_xy:
            ps = PoseStampedMsg()
            ps.header = path_msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self.pub_path.publish(path_msg)

    def _publish_markers(self):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()

        # ---- 0) Clickable ground plane (Publish Point가 어디든 찍히게) ----
        ground = Marker()
        ground.header.frame_id = self.frame_id
        ground.header.stamp = now
        ground.ns = "ground"
        ground.id = 0
        ground.type = Marker.CUBE
        ground.action = Marker.ADD
        ground.pose.position.x = 0.0
        ground.pose.position.y = 0.0
        ground.pose.position.z = -0.01
        ground.pose.orientation.w = 1.0
        ground.scale.x = float(self.field_width)
        ground.scale.y = float(self.field_height)
        ground.scale.z = 0.02

        # 바닥 색(원하면 여기만 바꾸면 됨)
        ground.color.r = 0.4
        ground.color.g = 0.8
        ground.color.b = 0.4
        ground.color.a = 0.25

        ma.markers.append(ground)

        # helper: sphere
        def sphere(ns, mid, x, y, scale, rgba):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = ns
            m.id = mid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.02
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = float(scale)
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            return m

        # 1) robot
        if self.state.robot_xy is not None:
            x, y = self.state.robot_xy
            ma.markers.append(sphere("robot", 0, x, y, 0.18, (0.2, 0.9, 0.2, 1.0)))

        # 2) goal
        if self.state.goal_xy is not None:
            x, y = self.state.goal_xy
            ma.markers.append(sphere("goal", 0, x, y, 0.18, (0.9, 0.2, 0.2, 1.0)))

        # 3) obstacles: circle + octagon outline (A 방식)
        r = float(self.state.obstacle_radius)
        polys = obstacles_as_polygons(self.state.obstacles_xy, r, self.square)

        for i, ((cx, cy), poly) in enumerate(zip(self.state.obstacles_xy, polys)):
            # circle as cylinder (semi-transparent)
            circ = Marker()
            circ.header.frame_id = self.frame_id
            circ.header.stamp = now
            circ.ns = "obstacles_circle"
            circ.id = i
            circ.type = Marker.CYLINDER
            circ.action = Marker.ADD
            circ.pose.position.x = float(cx)
            circ.pose.position.y = float(cy)
            circ.pose.position.z = 0.0
            circ.pose.orientation.w = 1.0
            circ.scale.x = circ.scale.y = float(2.0 * r)
            circ.scale.z = 0.05
            circ.color.r, circ.color.g, circ.color.b, circ.color.a = (0.2, 0.4, 1.0, 0.20)
            ma.markers.append(circ)

            # polygon outline
            poly_m = Marker()
            poly_m.header.frame_id = self.frame_id
            poly_m.header.stamp = now
            poly_m.ns = "obstacles_poly"
            poly_m.id = i
            poly_m.type = Marker.LINE_STRIP
            poly_m.action = Marker.ADD
            poly_m.scale.x = 0.03
            poly_m.color.r, poly_m.color.g, poly_m.color.b, poly_m.color.a = (0.1, 0.2, 1.0, 0.9)

            for (x, y) in poly:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.02
                poly_m.points.append(p)

            # close loop
            if poly:
                x0, y0 = poly[0]
                p0 = Point()
                p0.x = float(x0)
                p0.y = float(y0)
                p0.z = 0.02
                poly_m.points.append(p0)

            ma.markers.append(poly_m)

        self.pub_markers.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = VgRvizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()