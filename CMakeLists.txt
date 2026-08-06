cmake_minimum_required(VERSION 3.16)
project(livox_color)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-lstdc++fs)

# 必须先找到 ament_cmake
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(OpenCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)

find_library(LIVOX_SDK2_LIB livox_lidar_sdk_static REQUIRED)
find_library(MPP_LIB rockchip_mpp REQUIRED)
find_library(RGA_LIB rga REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

# ==================== 节点 1: livox_color_node ====================
add_executable(livox_color_node src/main.cpp)
ament_target_dependencies(livox_color_node rclcpp sensor_msgs Threads OpenCV OpenMP cv_bridge)
target_link_libraries(livox_color_node
        ${LIVOX_SDK2_LIB} ${MPP_LIB} ${RGA_LIB} stdc++fs ${OpenCL_LIBRARIES})

# ==================== 节点 2: livox_color_2e_node (新增) ====================
add_executable(livox_color_2e_node src/main_2e.cpp) # 假设 main_2e.cpp 同样放在 src 目录下
ament_target_dependencies(livox_color_2e_node rclcpp sensor_msgs Threads OpenCV OpenMP cv_bridge tf2_ros tf2_msgs)
target_link_libraries(livox_color_2e_node
        ${LIVOX_SDK2_LIB} ${MPP_LIB} ${RGA_LIB} stdc++fs ${OpenCL_LIBRARIES})

# ==================== 设置安装路径 ====================
# 将两个节点都添加到安装目标中
install(TARGETS
        livox_color_node
        livox_color_2e_node
        DESTINATION lib/${PROJECT_NAME})

# 安装 launch 文件夹及其所有内容
install(DIRECTORY
        launch
        DESTINATION share/${PROJECT_NAME}
        )

ament_package()
#colcon build --packages-select livox_color
#su - root -c "source /opt/ros/foxy/setup.bash && source /ros2_ws/install/setup.bash && ros2 run livox_color livox_color_2e_node --ros-args -p is_color:=false -p frames_per_publish:=10000 -p resolution:=3840x1080 -p tilt:=false"