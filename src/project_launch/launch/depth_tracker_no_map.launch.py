from launch import LaunchDescription
from launch_ros.actions import Node
#launch file for not showing disparity map. you can change parameters in the Nodes.

def generate_launch_description():
    ld = LaunchDescription()

    camera_node = Node(
        package="merits_project",
        executable="camera_node",
        name="camera_node",
        remappings=[
        ],
        parameters=[
        ]

    )

    coords_synchronized = Node(
        package="merits_project",
        executable="coords_no_map",
        name="coords_no_map",
        remappings=[
        ],
        parameters=[
        ]
    )


    ld.add_action(camera_node)
    ld.add_action(coords_synchronized)
    return ld
