#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometric_shapes/shape_operations.h>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <moveit_msgs/msg/object_color.hpp>

using namespace std::chrono_literals;

double wm_y = -0.015 + 0.010 + 0.11; // ORIGIN 0.075 -> 0.09 -> 0.11 -> 0.01 -> -0.015 (front panel at y=0.24m), +1.0cm, +11cm
double wm_x = -0.01 ;  // 세탁기 x축 오프셋 (-1cm)


void addBox(moveit::planning_interface::PlanningSceneInterface& psi,
            const std::string& id,
            const std::vector<double>& size,
            const geometry_msgs::msg::Pose& pose,
            float r = 0.5, float g = 0.5, float b = 0.5, float a = 1.0)
{
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = "world";
  object.id = id;

  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = primitive.BOX;
  primitive.dimensions.resize(3);
  primitive.dimensions[0] = size[0];
  primitive.dimensions[1] = size[1];
  primitive.dimensions[2] = size[2];

  object.primitives.push_back(primitive);
  object.primitive_poses.push_back(pose);
  object.operation = object.ADD;

  // 색상 및 투명도 설정
  moveit_msgs::msg::ObjectColor color;
  color.id = id;
  color.color.r = r;
  color.color.g = g;
  color.color.b = b;
  color.color.a = a;

  std::vector<moveit_msgs::msg::CollisionObject> objects;
  objects.push_back(object);
  std::vector<moveit_msgs::msg::ObjectColor> colors;
  colors.push_back(color);

  psi.applyCollisionObjects(objects, colors);
}

geometry_msgs::msg::Pose makePose(double x, double y, double z)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation.w = 1.0;
  return pose;
}

geometry_msgs::msg::Pose makeRotatedPose(double x, double y, double z, double roll, double pitch, double yaw)
{
  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  return pose;
}

void addCircularDrum(moveit::planning_interface::PlanningSceneInterface& psi) //큰 원
{
  const int num_segments = 72;
  const double radius = 0.29;
  const double center_y = 0.28 + 0.3 + 0.05 +wm_y ;
  const double center_z = 0.88;
  const double box_thickness = 0.005*2;
  const double box_depth = 0.0127*2;
  const double box_height = 0.28*2;

  for (int i = 0; i < num_segments; ++i)
  {
    double angle = 2 * M_PI * i / num_segments;
    double x = radius * cos(angle);
    double z = radius * sin(angle);

    std::string id = "ow_drum_" + std::to_string(i);
    addBox(psi, id, {box_thickness, box_height, box_depth},
           makeRotatedPose(x + wm_x, center_y, center_z + z, 0, -angle, 0), 0.5, 0.5, 0.5, 0.3);
  }
}

void addWasherDrum(moveit::planning_interface::PlanningSceneInterface& psi) //작은 원
{
  const int num_segments = 36;
  const double radius = 0.205;
  const double center_y = 0.3+wm_y;  // wm_y 포함된 y
  const double center_z = 0.88;
  const double box_thickness = 0.017*2;  // x축 방향 길이
  const double box_height = 0.1;     // y축 방향 길이 (드럼 높이)
  const double box_depth = 0.017*2;      // z축 방향 길이

  for (int i = 0; i < num_segments; ++i)
  {
    double angle = 2 * M_PI * i / num_segments;
    double x = radius * cos(angle);
    double z = radius * sin(angle);
    std::string id = "wm_drum_seg_" + std::to_string(i);
    addBox(psi, id,
           {box_thickness, box_height, box_depth},
           makeRotatedPose(x + wm_x, center_y, center_z + z, 0, -angle, 0), 0.5, 0.5, 0.5, 0.3);
  }
}



int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("add_scene_objects_node");
  bool add_basket = node->declare_parameter<bool>("add_basket", true);
  moveit::planning_interface::PlanningSceneInterface psi;

  rclcpp::sleep_for(2s);

  addCircularDrum(psi);
  addWasherDrum(psi);

  // 책상: 로봇 base(x=0, y=0) 중심의 30cm x 30cm 정사각형 통짜 육면체 (다리 없음).
  // 상판 높이는 기존과 동일하게 z=0.445 유지 (로봇 base_link 마운트 높이와 일치)
  // → 바닥(z=0)부터 0.445까지 통짜 박스, 중심 z = 0.445/2 = 0.2225
  addBox(psi, "desk_block", {0.3, 0.3, 0.445}, makePose(0.0, 0.0, 0.2225));

  if (add_basket) {
    double basket_y = - (0.36+0.02);
    double basket_x = -0.06 + 0.010;
    double basket_z_offset = 0.25;

    // 바구니 - 바닥
    addBox(psi, "basket_bottom", {0.40, 0.28, 0.01}, makePose(basket_x, 0.0+basket_y, 0.115+basket_z_offset), 0.1, 0.1, 0.8, 0.3);

    double wall_thickness = 0.0175;
    double wall_y_offset = (0.28 - wall_thickness) / 2.0;
    double wall_x_offset = (0.40 - wall_thickness) / 2.0;

    // 바구니 - 앞/뒤 벽
    addBox(psi, "basket_wall_front", {0.40, wall_thickness, 0.25}, makePose(basket_x, wall_y_offset+basket_y, 0.2+basket_z_offset), 0.1, 0.1, 0.8, 0.3);
    addBox(psi, "basket_wall_back", {0.40, wall_thickness, 0.25}, makePose(basket_x, -wall_y_offset+basket_y, 0.2+basket_z_offset), 0.1, 0.1, 0.8, 0.3);

    // 바구니 - 좌/우 벽
    addBox(psi, "basket_wall_left", {wall_thickness, 0.28, 0.25}, makePose(basket_x+wall_x_offset, 0.0+basket_y, 0.2+basket_z_offset), 0.1, 0.1, 0.8, 0.3);
    addBox(psi, "basket_wall_right", {wall_thickness, 0.28, 0.25}, makePose(basket_x-wall_x_offset, 0.0+basket_y, 0.2+basket_z_offset), 0.1, 0.1, 0.8, 0.3);
  }

  // 세탁기 - 앞판 위쪽
  addBox(psi, "front_top", {0.6860, 0.01, 0.410}, makeRotatedPose(0+wm_x, 0.255+wm_y, 1.245, 0.00000, 0.00000, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "front_bottom", {0.6860, 0.010, 0.35}, makeRotatedPose(0+wm_x, 0.255+wm_y, 0.535, 0.00000, 0.00000, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "front_left", {0.1580, 0.010, 0.370}, makeRotatedPose(0.08600-0.35+wm_x, 0.255+wm_y, 0.885, 0.00000, 0.00000, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "front_right", {0.1580, 0.010, 0.370}, makeRotatedPose(0.61400-0.35+wm_x, 0.255+wm_y, 0.885, 0.00000, 0.00000, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "corner_filler_tr", {0.150, 0.010, 0.150}, makeRotatedPose(0.53500-0.35+wm_x, 0.255+wm_y, 1.06000, 0.00000, -0.78540, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "corner_filler_tl", {0.150, 0.010, 0.150}, makeRotatedPose(0.16500-0.35+wm_x, 0.255+wm_y, 1.06000, 0.00000, -0.78540, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "corner_filler_br", {0.150, 0.010, 0.150}, makeRotatedPose(0.53500-0.35+0.01+wm_x, 0.255+wm_y, 0.7200, 0.00000, -0.78540, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "corner_filler_bl", {0.150, 0.010, 0.150}, makeRotatedPose(0.16500-0.35-0.01+wm_x, 0.255+wm_y, 0.7200, 0.00000, -0.78540, 0.00000), 0.5, 0.5, 0.5, 0.3);

  addBox(psi, "stand_base", {0.686, 0.875, 0.36}, makePose(0+wm_x, 0.6875+wm_y, 0.18), 0.5, 0.5, 0.5, 0.3);  // 하단 받침대
  // 상단 몸통
  addBox(psi, "main_body_left", {0.02, 0.875, 1.11}, makePose(-0.333+wm_x, 0.6875+wm_y, 0.915), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "main_body_right", {0.02, 0.875, 1.11}, makePose(0.333+wm_x, 0.6875+wm_y, 0.915), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "main_body_top", {0.686, 0.875, 0.02}, makePose(0+wm_x, 0.6875+wm_y, 1.465), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "main_body_bottom", {0.686, 0.875, 0.02}, makePose(0+wm_x, 0.6875+wm_y, 0.37), 0.5, 0.5, 0.5, 0.3);


  RCLCPP_INFO(node->get_logger(), "Planning Scene이 추가되었습니다.");

  rclcpp::shutdown();
  return 0;
}
