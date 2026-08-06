import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 声明启动参数 (Declare Launch Arguments)
    # 这样你就可以在命令行通过 'is_color:=false' 这种方式改参数了
    declare_is_color = DeclareLaunchArgument(
        'is_color',
        default_value='true',
        description='Whether to enable coloring'
    )

    declare_pc_frame_id = DeclareLaunchArgument(
        'pc_frame_id',
        default_value='livox_frame',
        description='Point cloud frame ID'
    )

    declare_cam_frame_id = DeclareLaunchArgument(
        'cam_frame_id',
        default_value='camera_frame',
        description='Camera frame ID'
    )

    declare_frames_per_publish = DeclareLaunchArgument(
        'frames_per_publish',
        default_value='200',
        description='Number of frames to accumulate before publishing (long)'
    )

    declare_resolution = DeclareLaunchArgument(
        'resolution',
        default_value='4000x3000',
        description='Image resolution'
    )

    # 2. 获取参数引用 (Launch Configurations)
    is_color_config = LaunchConfiguration('is_color')
    pc_frame_id_config = LaunchConfiguration('pc_frame_id')
    cam_frame_id_config = LaunchConfiguration('cam_frame_id')
    frames_per_publish_config = LaunchConfiguration('frames_per_publish')
    resolution_config = LaunchConfiguration('resolution')

    # 3. 定义节点
    livox_node = Node(
        package='livox_color',           # 功能包名
        executable='livox_color_node',   # 可执行文件名 (Node Name)
        name='livox_color_node',         # 运行时的节点名
        output='screen',                 # 将日志输出到终端
        # arguments=['--ros-args', '--log-level', 'DEBUG'],  # 关键：这里设置
        parameters=[{
            'is_color': is_color_config,
            'pc_frame_id': pc_frame_id_config,
            'cam_frame_id': cam_frame_id_config,
            'frames_per_publish': frames_per_publish_config,
            'resolution': resolution_config
        }]
    )

    # 4. 构建启动描述符
    return LaunchDescription([
        declare_is_color,
        declare_pc_frame_id,
        declare_cam_frame_id,
        declare_frames_per_publish,
        declare_resolution,
        livox_node
    ])