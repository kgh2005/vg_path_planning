import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped as PoseStampedMsg

from vg_path_planning.core.state import PlanningState
from vg_path_planning.core.planner import Planner
from vg_path_planning.core.fsm import PlannerFSM

from vg_path_planning.vg.obstacles import merged_obstacles_as_hulls
from vg_path_planning.vg.visibility_graph import (
    build_vg_nodes_from_polys,
    build_vg_edges,
    build_vg_path,
)


class VgRvizNode(Node):
    def __init__(self):
        super().__init__("vg_rviz")

        # ---- parameters ----
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("field_width", 20.0)     # meters
        self.declare_parameter("field_height", 14.0)    # meters
        self.declare_parameter("obstacle_radius", 0.30) # meters

        self.declare_parameter("obstacle_merge_dist", 0.60)

        self.declare_parameter("initialpose_topic", "/initialpose")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("obstacle_topic", "/clicked_point")

        self.declare_parameter("markers_topic", "/vg/markers")
        self.declare_parameter("path_topic", "/vg/path")

        # polygon sampling count
        self.declare_parameter("square", 8)

        # display toggles (YAML)
        self.declare_parameter("show_ground", True)
        self.declare_parameter("show_obstacles_circle", True)
        self.declare_parameter("show_obstacles_poly", True)
        self.declare_parameter("show_vg_nodes", False)
        self.declare_parameter("show_vg_edges", False)

        # ---- cached params ----
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.field_width = float(self.get_parameter("field_width").value)
        self.field_height = float(self.get_parameter("field_height").value)

        self.square = int(self.get_parameter("square").value)
        self.merge_dist = float(self.get_parameter("obstacle_merge_dist").value)

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

        self.get_logger().info(
            f"[vg_rviz] ready | R={radius:.2f} merge_dist={self.merge_dist:.2f} square={self.square}"
        )

    # -------- input callbacks --------
    def _cb_initialpose(self, msg: PoseWithCovarianceStamped):
        self.fsm.set_robot(msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _cb_goal(self, msg: PoseStamped):
        self.fsm.set_goal(msg.pose.position.x, msg.pose.position.y)

    def _cb_obstacle(self, msg: PointStamped):
        self.fsm.add_obstacle(msg.point.x, msg.point.y)

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

        # 토글은 매 tick 읽기(재시작 없이 파라미터 바꾸는 경우도 대응)
        show_ground = bool(self.get_parameter("show_ground").value)
        show_circle = bool(self.get_parameter("show_obstacles_circle").value)
        show_poly = bool(self.get_parameter("show_obstacles_poly").value)
        show_nodes = bool(self.get_parameter("show_vg_nodes").value)
        show_edges = bool(self.get_parameter("show_vg_edges").value)

        # ---- 0) RViz 잔상 제거 ----
        clear = Marker()
        clear.header.frame_id = self.frame_id
        clear.header.stamp = now
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        # ---- helper: sphere ----
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

        # ---- 1) ground ----
        if show_ground:
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
            ground.color.r, ground.color.g, ground.color.b, ground.color.a = (0.4, 0.8, 0.4, 0.25)
            ma.markers.append(ground)

        # ---- 2) robot / goal ----
        if self.state.robot_xy is not None:
            x, y = self.state.robot_xy
            ma.markers.append(sphere("robot", 0, x, y, 0.18, (0.2, 0.9, 0.2, 1.0)))

        if self.state.goal_xy is not None:
            x, y = self.state.goal_xy
            ma.markers.append(sphere("goal", 0, x, y, 0.18, (0.9, 0.2, 0.2, 1.0)))

        # ---- 3) obstacles ----
        r = float(self.state.obstacle_radius)

        # (A) raw circles (옵션)
        if show_circle:
            for i, (cx, cy) in enumerate(self.state.obstacles_xy):
                circ = Marker()
                circ.header.frame_id = self.frame_id
                circ.header.stamp = now
                circ.ns = "obstacles_circle_raw"
                circ.id = i
                circ.type = Marker.CYLINDER
                circ.action = Marker.ADD
                circ.pose.position.x = float(cx)
                circ.pose.position.y = float(cy)
                circ.pose.position.z = 0.0
                circ.pose.orientation.w = 1.0
                circ.scale.x = circ.scale.y = float(2.0 * r)
                circ.scale.z = 0.04
                circ.color.r, circ.color.g, circ.color.b, circ.color.a = (0.2, 0.4, 1.0, 0.12)
                ma.markers.append(circ)

        # (B) merged hull polygons (2개 범위 커버용)  obstacle_merge_dist 사용
        polys = merged_obstacles_as_hulls(
            self.state.obstacles_xy,
            radius=r,
            n_vertices=self.square,
            merge_dist=self.merge_dist,
        )

        if show_poly:
            for i, poly in enumerate(polys):
                poly_m = Marker()
                poly_m.header.frame_id = self.frame_id
                poly_m.header.stamp = now
                poly_m.ns = "obstacles_poly"
                poly_m.id = i
                poly_m.type = Marker.LINE_STRIP
                poly_m.action = Marker.ADD
                poly_m.scale.x = 0.03
                poly_m.color.r, poly_m.color.g, poly_m.color.b, poly_m.color.a = (0.1, 0.2, 1.0, 0.95)

                for (x, y) in poly:
                    p = Point()
                    p.x = float(x)
                    p.y = float(y)
                    p.z = 0.02
                    poly_m.points.append(p)

                if poly:
                    x0, y0 = poly[0]
                    p0 = Point()
                    p0.x = float(x0)
                    p0.y = float(y0)
                    p0.z = 0.02
                    poly_m.points.append(p0)

                ma.markers.append(poly_m)

        # ---- 4) VG nodes/edges/path ----
        vg = build_vg_nodes_from_polys(
            self.state.robot_xy,
            self.state.goal_xy,
            polys,
        )

        if vg is not None:
            edges = build_vg_edges(vg)

            if show_nodes:
                pts = Marker()
                pts.header.frame_id = self.frame_id
                pts.header.stamp = now
                pts.ns = "vg_nodes"
                pts.id = 0
                pts.type = Marker.SPHERE_LIST
                pts.action = Marker.ADD
                pts.scale.x = pts.scale.y = pts.scale.z = 0.06
                pts.color.r, pts.color.g, pts.color.b, pts.color.a = (1.0, 0.9, 0.1, 1.0)

                for (x, y) in vg.nodes:
                    p = Point()
                    p.x = float(x)
                    p.y = float(y)
                    p.z = 0.04
                    pts.points.append(p)

                ma.markers.append(pts)

            if show_edges:
                edge_m = Marker()
                edge_m.header.frame_id = self.frame_id
                edge_m.header.stamp = now
                edge_m.ns = "vg_edges"
                edge_m.id = 0
                edge_m.type = Marker.LINE_LIST
                edge_m.action = Marker.ADD
                edge_m.scale.x = 0.01
                edge_m.color.r, edge_m.color.g, edge_m.color.b, edge_m.color.a = (1.0, 1.0, 0.0, 0.35)

                for (i, j) in edges:
                    ax, ay = vg.nodes[i]
                    bx, by = vg.nodes[j]
                    pa = Point(); pa.x = float(ax); pa.y = float(ay); pa.z = 0.03
                    pb = Point(); pb.x = float(bx); pb.y = float(by); pb.z = 0.03
                    edge_m.points.append(pa)
                    edge_m.points.append(pb)

                ma.markers.append(edge_m)

            # path (A*)
            path_xy = build_vg_path(vg, edges)
            self._latest_path_xy = path_xy
            self._publish_path()

        self.pub_markers.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = VgRvizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()