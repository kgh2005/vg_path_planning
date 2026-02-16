from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("vg_path_planning")
    params = os.path.join(pkg_share, "config", "params.yaml")
    rviz_cfg = os.path.join(pkg_share, "config", "rviz.rviz")

    vg_path = Node(
        package="vg_path_planning",
        executable="vg_rviz",
        name="vg_rviz",
        output="screen",
        parameters=[params],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_cfg],
    )

    rviz_delayed = TimerAction(period=1.0, actions=[rviz])

    return LaunchDescription([
        vg_path,
        rviz_delayed
    ])