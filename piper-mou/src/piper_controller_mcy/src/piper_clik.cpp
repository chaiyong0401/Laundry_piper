#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "robot_controller/manipulator/base.h"
#include "robot_data/manipulator/base.h"

class PiperCLIK : public rclcpp::Node {
public:
  PiperCLIK()
  : rclcpp::Node("piper_piper"),
    control_dt_(0.01),
    is_initialized_(false),
    is_ee_target_received_(false)
  {
    // pubs / subs
    joint_ctrl_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
        "joint_ctrl_single_clik", 10);

    joint_states_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", 10,
        std::bind(&PiperCLIK::jointStatesCb, this, std::placeholders::_1));

    ee_target_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "desired_ee_pose", 10,
        std::bind(&PiperCLIK::desiredEePoseCb, this, std::placeholders::_1));

    // package paths
    const std::string robot_pkg_path =
        ament_index_cpp::get_package_share_directory("piper_with_gripper_moveit");
    const std::string desc_pkg_path =
        ament_index_cpp::get_package_share_directory("piper_description");
    const std::string urdf_path = desc_pkg_path + "/urdf/piper_description_wo_gripper.urdf";
    const std::string srdf_path = robot_pkg_path + "/config/piper.srdf";

    // RobotData / Controller
    robot_data_ = std::make_shared<RobotData::Manipulator::ManipulatorBase>(urdf_path, srdf_path, robot_pkg_path);
    controller_ = std::make_shared<RobotController::Manipulator::ManipulatorBase>(control_dt_, robot_data_);

    // Pretty print URDF info (가정: get_verbose()가 std::string 반환)
    {
      std::ostringstream oss;
      oss << "\033[1;34m\n"
          << "=================================================================\n"
          << "=================================================================\n"
          << "URDF Joint Information: Piper\n"
          << robot_data_->getVerbose()
          << "=================================================================\n"
          << "=================================================================\n\033[0m";
      RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
    }

    // states 초기화
    q_.setZero(6);
    q_desired_.setZero(6);
    qdot_.setZero(6);
    qdot_desired_.setZero(6);

    x_.setIdentity();
    x_init_.setIdentity();
    x_target_.setIdentity();

    xdot_.setZero(6);
    xdot_init_.setZero(6);

    last_print_time_ = this->now();

    Eigen::Vector6d Kp, Kv;
    Kp << 10,10,10,10,10,10;
    Kv << 1,1,1,1,1,1;
    controller_->setTaskGain(Kp,Kv);

    // timer: compute()를 control_dt 주기로 호출
    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(control_dt_),
        std::bind(&PiperCLIK::compute, this));
  }

private:
  // ===== callbacks =====
  void jointStatesCb(const sensor_msgs::msg::JointState::SharedPtr msg) {
    if (msg->position.size() < 6 || msg->velocity.size() < 6) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                           "JointState has insufficient DOF (<6). Ignored.");
      return;
    }

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      for (int i = 0; i < 6; ++i) {
        q_(i) = msg->position[i];
        qdot_(i) = msg->velocity[i];
      }
    }

    // Robot state update
    if (!robot_data_->updateState(q_, qdot_)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to update robot state.");
      return;
    }

    x_    = robot_data_->getPose("link7");
    xdot_ = robot_data_->getVelocity("link7");

    if (!is_initialized_) {
      if (x_.translation().norm() >1E-5) {
        q_desired_ = q_;
        qdot_desired_ = qdot_;
        x_target_ = x_;
        x_init_ = x_;
        xdot_init_ = xdot_;
        control_start_time_ = this->now();
        is_initialized_ = true;
      }
    }
  }

  void desiredEePoseCb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    // geometry_msgs Quaternion(x,y,z,w) -> Eigen Quaternion(w,x,y,z)
    Eigen::Quaterniond q(
        msg->pose.orientation.w,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z
    );
    Eigen::Matrix3d rot = q.normalized().toRotationMatrix();

    {
      std::lock_guard<std::mutex> lock(desired_mutex_);
      x_target_.setIdentity();
      x_target_.linear() = rot;
      x_target_.translation() << msg->pose.position.x,
                                   msg->pose.position.y,
                                   msg->pose.position.z;
    }

    // 새 목표 수신 → 현재 상태를 기준으로 cubic 시작
    is_ee_target_received_ = true;
  }

  // ===== control loop =====
  void compute() {
    if (!is_initialized_) {
      return;
    }

    // 새 목표가 들어왔으면 초기화
    if (is_ee_target_received_) {
      control_start_time_ = this->now();
      x_init_ = x_;
      xdot_init_ = xdot_;
      is_ee_target_received_ = false;
    }

    // const double now_sec   = this->now().seconds();
    // const double start_sec = control_start_time_.seconds();
    const double now_sec   = this->now().nanoseconds()*1E-9;
    const double start_sec = control_start_time_.nanoseconds()*1E-9;
    const double duration  = 2.0; // 필요시 파라미터화

    // CLIK + cubic 보간으로 qdot 계산
    // 가정: controller_.CLIK_cubic(...) -> Eigen::Matrix<double,6,1> 반환
    const Eigen::Matrix<double,6,1> xdot_target = Eigen::Matrix<double,6,1>::Zero();
    qdot_desired_ = controller_->CLIKCubic(x_target_,
                                                 xdot_target,
                                                 x_init_,
                                                 xdot_init_,
                                                 now_sec, 
                                                 start_sec, 
                                                 duration, 
                                                 "link7");
    // qdot_desired_.setZero(6);

    // simple Euler integration
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      q_desired_ += qdot_desired_ * control_dt_;
    }

    // 10 Hz 로깅
    if ((this->now() - last_print_time_).nanoseconds()*1E-9 >= 0.1) {
      last_print_time_ = this->now();

      std::ostringstream oss;
      oss << "\n====================================================================="
          << "\n[CTRL " << std::fixed << std::setprecision(3) << now_sec << "s]"
          << "\n  q           : " << q_.transpose()
          << "\n  q_desired   : " << q_desired_.transpose()
          << "\n  qdot        : " << qdot_.transpose()
          << "\n  qdot_desired: " << qdot_desired_.transpose()
          << "\n  x (4x4)     :\n" << x_.matrix()
          << "\n  xdot        : " << xdot_.transpose()
          << "\n  x_init      :\n" << x_init_.matrix()
          << "\n  xdot_init   : " << xdot_init_.transpose()
          << "\n  J           : \n" << robot_data_->getJacobian("link7")
          // << "\n  J_pinv      : \n" << robot_data_->getJacobian("link7").transpose() * (robot_data_->getJacobian("link7") * robot_data_->getJacobian("link7").transpose() + 0.01*Eigen::Matrix6d::Identity()).inverse()
          << "\n  JJT         : \n" << (robot_data_->getJacobian("link7")*robot_data_->getJacobian("link7").transpose())
          << "\n  Mani        : " << std::sqrt( (robot_data_->getJacobian("link7")*robot_data_->getJacobian("link7").transpose()).determinant())
          << "\n  x_desired   :\n" << x_target_.matrix()
          << "\n========================================================================";
      RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
    }

    // publish
    auto js = sensor_msgs::msg::JointState();
    js.header.stamp = this->now();
    js.name = {"joint1","joint2","joint3","joint4","joint5","joint6"};
    js.position.resize(6);
    js.velocity.resize(6);
    js.effort.assign(6, 0.0);

    for (int i = 0; i < 6; ++i) {
      js.position[i] = q_desired_(i);
      js.velocity[i] = qdot_desired_(i);
    }
    joint_ctrl_pub_->publish(js);
  }

private:
  // ros io
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_ctrl_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ee_target_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // robot
  std::shared_ptr<RobotData::Manipulator::ManipulatorBase> robot_data_;
  std::shared_ptr<RobotController::Manipulator::ManipulatorBase> controller_;

  // states
  std::mutex state_mutex_;
  std::mutex desired_mutex_;

  Eigen::Matrix<double,6,1> q_, q_desired_;
  Eigen::Matrix<double,6,1> qdot_, qdot_desired_;
  Eigen::Affine3d x_, x_init_, x_target_;
  Eigen::Matrix<double,6,1> xdot_, xdot_init_;

  // control
  const double control_dt_;
  bool is_initialized_;
  bool is_ee_target_received_;
  rclcpp::Time control_start_time_;
  rclcpp::Time last_print_time_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PiperCLIK>());
  rclcpp::shutdown();
  return 0;
}
