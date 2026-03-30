cmake_minimum_required(VERSION 3.16)
project(piper_controller_mcy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Eigen3 REQUIRED)

# 선택적: dyros_robot_controller가 ament 패키지로 설치되어 있다면
find_package(dyros_robot_controller REQUIRED)

add_executable(piper_clik_node src/piper_clik.cpp)

target_include_directories(piper_clik_node PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(piper_clik_node
  rclcpp
  sensor_msgs
  geometry_msgs
  ament_index_cpp
)

# Install Python script
install(PROGRAMS scripts/piper_srmt.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME piper_srmt
)

# 선택적 링크 (패키지가 ament로 export 되어 있다면)
if(dyros_robot_controller_FOUND)
  ament_target_dependencies(piper_clik_node dyros_robot_controller)
endif()

install(TARGETS piper_clik_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# install(DIRECTORY launch
#   DESTINATION share/${PROJECT_NAME}
# )

ament_package()
