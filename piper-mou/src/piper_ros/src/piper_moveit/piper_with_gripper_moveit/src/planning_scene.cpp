#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometric_shapes/shape_operations.h>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <moveit_msgs/msg/object_color.hpp>

using namespace std::chrono_literals;

// double wm_y = 0.11;
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

  // 책상 상판 (Z center=0.425, Thickness=0.04 => Top=0.445)
  addBox(psi, "desk_top", {0.56, 0.25, 0.04}, makePose(-0.2, 0.0, 0.425));
  // 책상 왼쪽 다리
  addBox(psi, "desk_leg_left", {0.03, 0.22, 0.33}, makePose(-0.46, 0.0, 0.265));
  // 책상 오른쪽 다리
  addBox(psi, "desk_leg_right", {0.03, 0.22, 0.33}, makePose(0.06, 0.0, 0.265));

  // original
  // // 책상 상판
  // addBox(psi, "desk_top", {0.56, 0.31, 0.04}, makePose(0.0, 0.0, 0.35));
  // // 책상 왼쪽 다리
  // addBox(psi, "desk_leg_left", {0.03, 0.28, 0.33}, makePose(-0.26, 0.0, 0.165));
  // // 책상 오른쪽 다리
  // addBox(psi, "desk_leg_right", {0.03, 0.28, 0.33}, makePose(0.26, 0.0, 0.165));

  if (add_basket) {
    double basket_y = - (0.36+0.02); // front wall outer face at 24cm from robot
    double basket_x = -0.06 + 0.010; // +1.0cm in x

    // 바구니 - 바닥 (가로 X=40cm, 세로 Y=28cm, 두께 1cm, Z=0.115 => Top=0.12)
    addBox(psi, "basket_bottom", {0.40, 0.28, 0.01}, makePose(basket_x, 0.0+basket_y, 0.115), 0.1, 0.1, 0.8, 0.3);

    // 벽 두께 1.75cm (0.0175m)
    double wall_thickness = 0.0175;
    double wall_y_offset = (0.28 - wall_thickness) / 2.0;
    double wall_x_offset = (0.40 - wall_thickness) / 2.0;

    // 바구니 - 앞/뒤 벽
    addBox(psi, "basket_wall_front", {0.40, wall_thickness, 0.25}, makePose(basket_x, wall_y_offset+basket_y, 0.2), 0.1, 0.1, 0.8, 0.3);
    addBox(psi, "basket_wall_back", {0.40, wall_thickness, 0.25}, makePose(basket_x, -wall_y_offset+basket_y, 0.2), 0.1, 0.1, 0.8, 0.3);

    // 바구니 - 좌/우 벽
    addBox(psi, "basket_wall_left", {wall_thickness, 0.28, 0.25}, makePose(basket_x+wall_x_offset, 0.0+basket_y, 0.2), 0.1, 0.1, 0.8, 0.3);
    addBox(psi, "basket_wall_right", {wall_thickness, 0.28, 0.25}, makePose(basket_x-wall_x_offset, 0.0+basket_y, 0.2), 0.1, 0.1, 0.8, 0.3);
  }

  // origianl
  // // 바구니 - 바닥
  // addBox(psi, "basket_bottom", {0.4, 0.26, 0.02}, makePose(0.0, 0.0+basket_y, 0.01));
  // // // 바구니 - 앞/뒤 벽
  // addBox(psi, "basket_wall_front", {0.4, 0.004, 0.26}, makePose(0.0, 0.12+basket_y, 0.1));
  // addBox(psi, "basket_wall_back", {0.4, 0.004, 0.26}, makePose(0.0, -0.12+basket_y, 0.1));
  // // // 바구니 - 좌/우 벽
  // addBox(psi, "basket_wall_left", {0.004, 0.22, 0.26}, makePose(0.21, 0.0+basket_y, 0.1));
  // addBox(psi, "basket_wall_right", {0.004, 0.22, 0.26}, makePose(-0.21, 0.0+basket_y, 0.1));

  // // 세탁기 - 앞판 위쪽
  addBox(psi, "front_top", {0.6860, 0.01, 0.400}, makeRotatedPose(0+wm_x, 0.255+wm_y, 1.25000, 0.00000, 0.00000, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "front_bottom", {0.6860, 0.010, 0.35}, makeRotatedPose(0+wm_x, 0.255+wm_y, 0.535, 0.00000, 0.00000, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "front_left", {0.1580, 0.010, 0.370}, makeRotatedPose(0.08600-0.35+wm_x, 0.255+wm_y, 0.885, 0.00000, 0.00000, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "front_right", {0.1580, 0.010, 0.370}, makeRotatedPose(0.61400-0.35+wm_x, 0.255+wm_y, 0.885, 0.00000, 0.00000, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "corner_filler_tr", {0.150, 0.010, 0.150}, makeRotatedPose(0.53500-0.35+wm_x, 0.255+wm_y, 1.06000, 0.00000, -0.78540, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "corner_filler_tl", {0.150, 0.010, 0.150}, makeRotatedPose(0.16500-0.35+wm_x, 0.255+wm_y, 1.06000, 0.00000, -0.78540, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "corner_filler_br", {0.150, 0.010, 0.150}, makeRotatedPose(0.53500-0.35+0.01+wm_x, 0.255+wm_y, 0.7200, 0.00000, -0.78540, 0.00000), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "corner_filler_bl", {0.150, 0.010, 0.150}, makeRotatedPose(0.16500-0.35-0.01+wm_x, 0.255+wm_y, 0.7200, 0.00000, -0.78540, 0.00000), 0.5, 0.5, 0.5, 0.3);

  addBox(psi, "stand_base", {0.686, 0.875, 0.36}, makePose(0+wm_x, 0.6875+wm_y, 0.18), 0.5, 0.5, 0.5, 0.3);  // 하단 받침대
  // 상단 몸통
  // addBox(psi, "main_body_back", {0.686, 0.02, 1.11}, makePose(0, 1.125, 0.915));
  addBox(psi, "main_body_left", {0.02, 0.875, 1.11}, makePose(-0.333+wm_x, 0.6875+wm_y, 0.915), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "main_body_right", {0.02, 0.875, 1.11}, makePose(0.333+wm_x, 0.6875+wm_y, 0.915), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "main_body_top", {0.686, 0.875, 0.02}, makePose(0+wm_x, 0.6875+wm_y, 1.465), 0.5, 0.5, 0.5, 0.3);
  addBox(psi, "main_body_bottom", {0.686, 0.875, 0.02}, makePose(0+wm_x, 0.6875+wm_y, 0.37), 0.5, 0.5, 0.5, 0.3);

  // for test
  // addBox(psi, "boxboxbox", {0.1, 0.1, 0.6}, makePose(0.3, 0, 0.6));  


  RCLCPP_INFO(node->get_logger(), "Planning Scene이 추가되었습니다.");

  // 0.2 + 

  rclcpp::shutdown();
  return 0;
}




// /////////////////////////////////////////////////////////////////////////////////////////////
// /// 책상 입니다 //
// /////////////////////////////////////////////////////////////////////////////////////////////



// #include <rclcpp/rclcpp.hpp>
// #include <geometry_msgs/msg/pose.hpp>

// #include <moveit/planning_scene_interface/planning_scene_interface.h>
// #include <moveit_msgs/msg/collision_object.hpp>
// #include <shape_msgs/msg/solid_primitive.hpp>

// #include <array>
// #include <string>
// #include <vector>

// using moveit::planning_interface::PlanningSceneInterface;

// // ------------------------
// // Pose helper
// // ------------------------
// geometry_msgs::msg::Pose makePose(double x, double y, double z)
// {
//   geometry_msgs::msg::Pose p;
//   p.position.x = x;
//   p.position.y = y;
//   p.position.z = z;
//   p.orientation.x = 0.0;
//   p.orientation.y = 0.0;
//   p.orientation.z = 0.0;
//   p.orientation.w = 1.0;  // no rotation
//   return p;
// }

// // ------------------------
// // Add box helper
// // ------------------------
// void addBox(PlanningSceneInterface &psi,
//             const std::string &name,
//             const std::array<double, 3> &size,
//             const geometry_msgs::msg::Pose &pose)
// {
//   moveit_msgs::msg::CollisionObject obj;
//   obj.header.frame_id = "world";  // 필요하면 "piper_base" 등으로 바꿔
//   obj.id = name;

//   shape_msgs::msg::SolidPrimitive primitive;
//   primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
//   primitive.dimensions.resize(3);
//   primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] = size[0];
//   primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = size[1];
//   primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] = size[2];

//   obj.primitives.push_back(primitive);
//   obj.primitive_poses.push_back(pose);
//   obj.operation = moveit_msgs::msg::CollisionObject::ADD;

//   psi.applyCollisionObject(obj);
// }



// // ======================================================
// //  addEnvironment() — 책상 + 바닥 바스켓 + 책상 위 바스켓
// // ======================================================
// void addEnvironment(PlanningSceneInterface &psi)
// {
//   // ==============================
//   // 0. 책상 (정사각형 테이블)
//   //   - 가로/세로: 77 cm → 0.77
//   //   - 높이: 73 cm → 0.73
//   //   - y = +0.40 위치
//   // ==============================
//   const double table_size_xy = 0.77;   // 77 cm
//   const double table_height  = 0.73;   // 73 cm
//   const double table_y       = -0.64;   // 40 cm

//   addBox(
//       psi,
//       "table",
//       {table_size_xy, table_size_xy, table_height},
//       makePose(0.0, table_y, table_height / 2.0));  // 바닥 기준 중앙 z


//   // ==============================
//   // 1. 바닥 바스켓
//   //   - footprint: 48 × 48 cm → 0.48 × 0.48
//   //   - 전체 높이: 30 cm → 0.30
//   //   - 벽/바닥 두께: 3 cm → 0.03
//   //   - y = -0.40
//   // ==============================
//   const double basket_floor_y = 0.40;    // 40 cm 뒤쪽

//   const double floor_basket_size_xy = 0.48;  // 48 cm
//   const double floor_basket_h_total = 0.30;  // 30 cm
//   const double floor_wall_thick     = 0.03;  // 3 cm

//   const double floor_bottom_h = floor_wall_thick;                      // 0.03
//   const double floor_wall_h   = floor_basket_h_total - floor_bottom_h; // 0.27
//   const double floor_half_xy  = floor_basket_size_xy / 2.0;           // 0.24

//   // --- 바닥 ---
//   addBox(
//       psi,
//       "basket_floor_bottom",
//       {floor_basket_size_xy, floor_basket_size_xy, floor_bottom_h},
//       makePose(0.0, basket_floor_y, floor_bottom_h / 2.0));

//   // 벽 공통 Z (바닥 위에 벽이 올라감)
//   const double floor_wall_center_z = floor_bottom_h + floor_wall_h / 2.0; // 0.03 + 0.27/2 = 0.165

//   // --- 앞/뒤 벽 ---
//   addBox(
//       psi,
//       "basket_floor_wall_front",
//       {floor_basket_size_xy, floor_wall_thick, floor_wall_h},
//       makePose(0.0, basket_floor_y + floor_half_xy, floor_wall_center_z));

//   addBox(
//       psi,
//       "basket_floor_wall_back",
//       {floor_basket_size_xy, floor_wall_thick, floor_wall_h},
//       makePose(0.0, basket_floor_y - floor_half_xy, floor_wall_center_z));

//   // --- 좌/우 벽 ---
//   addBox(
//       psi,
//       "basket_floor_wall_left",
//       {floor_wall_thick, floor_basket_size_xy, floor_wall_h},
//       makePose(floor_half_xy, basket_floor_y, floor_wall_center_z));

//   addBox(
//       psi,
//       "basket_floor_wall_right",
//       {floor_wall_thick, floor_basket_size_xy, floor_wall_h},
//       makePose(-floor_half_xy, basket_floor_y, floor_wall_center_z));



//   // ==============================
//   // 2. 책상 위 바스켓
//   //    (네가 쓰던 기존 코드 형상 그대로,
//   //     위치만 "책상 위"에 맞게 z 조정)
//   //
//   //   //   double basket_y = - (0.36+0.02);
//   //   //
//   //   //   addBox(psi, "basket_bottom", {0.34, 0.2, 0.04}, ...);
//   //   //   addBox(psi, "basket_wall_front", {0.34, 0.01, 0.25}, ...);
//   //   //   addBox(psi, "basket_wall_back",  {0.34, 0.01, 0.25}, ...);
//   //   //   addBox(psi, "basket_wall_left",  {0.01, 0.2, 0.25}, ...);
//   //   //   addBox(psi, "basket_wall_right", {0.01, 0.2, 0.25}, ...);
//   //
//   //   여기서는:
//   //     - 바닥: 0.34 x 0.2 x 0.04
//   //     - 앞/뒤 벽: 0.34 x 0.01 x 0.25
//   //     - 좌/우 벽: 0.01 x 0.2 x 0.25
//   //   위치:
//   //     - y 중심 = table_y (0.40)
//   //     - x 오프셋, y ±0.1, x = 0.12 / -0.2 는 기존 코드 유지
//   // ==============================

//   // 바스켓 바닥 크기
//   const double top_bottom_x = 0.34;
//   const double top_bottom_y = 0.20;
//   const double top_bottom_h = 0.04;   // 바닥 두께 4 cm

//   // 벽 크기
//   const double top_wall_h      = 0.14;
//   const double top_wall_thickY = 0.01;  // 앞/뒤 벽 두께
//   const double top_wall_thickX = 0.01;  // 좌/우 벽 두께

//   // z 위치 계산
//   const double top_basket_bottom_center_z = table_height + top_bottom_h / 2.0;
//   // 0.73 + 0.04/2 = 0.75

//   const double top_basket_wall_center_z   = table_height + top_bottom_h + top_wall_h / 2.0;
//   // 0.73 + 0.04 + 0.25/2 = 0.73 + 0.04 + 0.125 = 0.895
  
//   const double yy = 0.3;

//   // --- 바닥 (원래 x = -0.04 유지, y만 table_y 기준) ---
//   addBox(
//       psi,
//       "basket_top_bottom",
//       {top_bottom_x, top_bottom_y, top_bottom_h},
//       makePose(-0.04, table_y+yy, top_basket_bottom_center_z));

//   // --- 앞/뒤 벽 (원래 ±0.1 오프셋 유지) ---
//   addBox(
//       psi,
//       "basket_top_wall_front",
//       {top_bottom_x, top_wall_thickY, top_wall_h},
//       makePose(-0.04, table_y+yy + 0.10, top_basket_wall_center_z));

//   addBox(
//       psi,
//       "basket_top_wall_back",
//       {top_bottom_x, top_wall_thickY, top_wall_h},
//       makePose(-0.04, table_y+yy - 0.10, top_basket_wall_center_z));

//   // --- 좌/우 벽 (원래 x = 0.12 / -0.2 유지) ---
//   addBox(
//       psi,
//       "basket_top_wall_left",
//       {top_wall_thickX, top_bottom_y, top_wall_h},
//       makePose(0.12, table_y+yy, top_basket_wall_center_z));

//   addBox(
//       psi,
//       "basket_top_wall_right",
//       {top_wall_thickX, top_bottom_y, top_wall_h},
//       makePose(-0.20, table_y+yy, top_basket_wall_center_z));
// }



// // ======================================================
// // main()
// // ======================================================
// int main(int argc, char *argv[])
// {
//   rclcpp::init(argc, argv);
//   auto node = rclcpp::Node::make_shared("planning_scene_node");

//   PlanningSceneInterface psi;

//   RCLCPP_INFO(node->get_logger(), "Adding environment (table + baskets)...");
//   addEnvironment(psi);
//   RCLCPP_INFO(node->get_logger(), "Environment added.");

//   // 노드가 바로 종료되면 RViz에서 오브젝트가 사라질 수 있으니 유지
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }
