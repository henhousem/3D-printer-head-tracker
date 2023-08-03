from launch import LaunchDescription
from launch_ros.actions import Node
#launch file that shows disparity map. you can change parameters in the nodes.

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
        executable="coords_synchronized",
        name="coords_synchronized",
        remappings=[
        ],
        parameters=[
            {"Stereo_SGBM": True},
            {"custom_x_axis": 0},
            {"custom_y_axis": 0},
            {"custom_z_axis": 0}
        ]
    )


    ld.add_action(camera_node)
    ld.add_action(coords_synchronized)
    return ld
