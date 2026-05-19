#include <Eigen/Geometry>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>

#include "combat_rm_interfaces/msg/target.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/ros2/ros2.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff_fyt/buff_aimer.hpp"
#include "tasks/auto_buff_fyt/buff_detector.hpp"
#include "tasks/auto_buff_fyt/buff_solver.hpp"
#include "tasks/auto_buff_fyt/buff_target.hpp"
#include "tasks/auto_buff_fyt/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace
{
constexpr auto GIMBAL_DELAY = std::chrono::milliseconds(7);
constexpr auto USB_STARTUP_SETTLE = std::chrono::milliseconds(300);
constexpr double USB_LEFT_YAW_OFFSET = 2.7;
constexpr double USB_RIGHT_YAW_OFFSET = -2.7;
constexpr double USB_LEFT_TARGET_YAW_TRIM = -0.44;
constexpr double USB_RIGHT_TARGET_YAW_TRIM = 0.2;
constexpr double USB_TARGET_PITCH_TRIM = -5.0 / 57.3;
constexpr char USB_LEFT_DEVICE[] = "video0";
constexpr char USB_RIGHT_DEVICE[] = "video2";
constexpr double USB_SETTLE_YAW_THRESH = CV_PI / 180.0;
constexpr double USB_SETTLE_PITCH_THRESH = CV_PI / 180.0;
constexpr auto USB_SETTLE_TIMEOUT = std::chrono::milliseconds(1500);

enum class TargetSource
{
  none = 0,
  main = 1,
  usb_left = 2,
  usb_right = 3
};

enum class CommandKind
{
  none = 0,
  target = 1,
  fixed_aim = 2,
  settle_aim = 3
};

enum class OmniState
{
  lost = 0,
  main_tracking = 1,
  usb_perception = 2,
  lost_from_usb_settle = 3
};

struct TargetCommand
{
  std::optional<auto_aim::Target> target = std::nullopt;
  TargetSource source = TargetSource::none;
  CommandKind kind = CommandKind::none;
  OmniState omni_state = OmniState::lost;
  float target_yaw = 0.0F;
  float target_pitch = 0.0F;
  uint64_t sequence = 0;
};

struct UsbCandidate
{
  auto_aim::Armor armor;
  std::chrono::steady_clock::time_point timestamp;
  TargetSource source;
  double distance;
};

struct UsbThreadResult
{
  std::optional<UsbCandidate> candidate = std::nullopt;
  TargetCommand target_command;
  bool ready = false;
};

double usb_yaw_offset(TargetSource source)
{
  switch (source) {
    case TargetSource::usb_left:
      return USB_LEFT_YAW_OFFSET;
    case TargetSource::usb_right:
      return USB_RIGHT_YAW_OFFSET;
    default:
      return 0.0;
  }
}

double usb_target_yaw_trim(TargetSource source)
{
  switch (source) {
    case TargetSource::usb_left:
      return USB_LEFT_TARGET_YAW_TRIM;
    case TargetSource::usb_right:
      return USB_RIGHT_TARGET_YAW_TRIM;
    default:
      return 0.0;
  }
}

bool should_track_armor(auto_aim::ArmorName name, io::GimbalMode mode)
{
  if (mode == io::GimbalMode::AUTO_AIM) return name != auto_aim::ArmorName::outpost;
  if (mode == io::GimbalMode::OUTPOST) return name == auto_aim::ArmorName::outpost;
  return false;
}


std::string armor_number(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::ArmorName::one:
      return "1";
    case auto_aim::ArmorName::two:
      return "2";
    case auto_aim::ArmorName::three:
      return "3";
    case auto_aim::ArmorName::four:
      return "4";
    case auto_aim::ArmorName::five:
      return "5";
    case auto_aim::ArmorName::outpost:
      return "8";
    case auto_aim::ArmorName::sentry:
      return "7";
    case auto_aim::ArmorName::base:
      return "0";
    case auto_aim::ArmorName::not_armor:
      return "";
  }

  return "";
}

bool is_usb_source(TargetSource source)
{
  return source == TargetSource::usb_left || source == TargetSource::usb_right;
}

auto_aim::Plan make_fixed_plan(float yaw, float pitch)
{
  return auto_aim::Plan{true, false, yaw, pitch, yaw, 0.0F, 0.0F, pitch, 0.0F, 0.0F};
}

void apply_usb_offset(auto_aim::Plan & plan, const io::GimbalState & gs, TargetSource source)
{
  if (!is_usb_source(source) || !plan.control) return;

  auto yaw_offset = gs.yaw_diff + usb_yaw_offset(source) + usb_target_yaw_trim(source);
  plan.target_yaw = tools::limit_rad(plan.target_yaw + yaw_offset);
  plan.yaw = tools::limit_rad(plan.yaw + yaw_offset);
  plan.target_pitch += USB_TARGET_PITCH_TRIM;
  plan.pitch += USB_TARGET_PITCH_TRIM;
}

TargetCommand make_usb_fixed_command(const UsbCandidate & candidate, const io::GimbalState & gs)
{
  TargetCommand command;
  command.kind = CommandKind::fixed_aim;
  command.source = candidate.source;
  command.target_yaw = tools::limit_rad(
    candidate.armor.ypd_in_world[0] + gs.yaw_diff + usb_yaw_offset(candidate.source) +
    usb_target_yaw_trim(candidate.source));
  command.target_pitch = candidate.armor.ypd_in_world[1] + USB_TARGET_PITCH_TRIM;
  return command;
}

combat_rm_interfaces::msg::Target make_target_msg(const TargetCommand & target_command)
{
  combat_rm_interfaces::msg::Target msg;
  msg.header.stamp = rclcpp::Clock(RCL_SYSTEM_TIME).now();
  msg.header.frame_id = "vision_world";

  if (!target_command.target.has_value()) {
    msg.tracking = false;
    return msg;
  }

  const auto & target = target_command.target.value();
  const auto ekf_x = target.ekf_x();
  msg.tracking = true;
  msg.id = armor_number(target.name);
  msg.position.x = ekf_x[0];
  msg.position.y = ekf_x[2];
  msg.position.z = ekf_x[4];

  return msg;
}

Eigen::Quaterniond usb_world_q(const Eigen::Quaterniond & q)
{
  // The side USB cameras do not follow gimbal pitch; keep yaw/roll and clamp pitch to zero.
  auto usb_ypr = tools::eulers(q, 2, 1, 0);
  usb_ypr[1] = 0.0;
  return Eigen::Quaterniond(tools::rotation_matrix(usb_ypr));
}

std::list<auto_aim::Armor> filter_armors(
  std::list<auto_aim::Armor> armors, auto_aim::Color enemy_color, io::GimbalMode mode)
{
  armors.remove_if([enemy_color, mode](const auto_aim::Armor & armor) {
    return armor.color != enemy_color || !should_track_armor(armor.name, mode);
  });
  return armors;
}

void solve_armors(std::list<auto_aim::Armor> & armors, auto_aim::Solver & solver)
{
  for (auto & armor : armors) solver.solve(armor);
}

std::optional<UsbCandidate> get_nearest_candidate(
  const std::list<auto_aim::Armor> & armors, std::chrono::steady_clock::time_point timestamp,
  TargetSource source)
{
  if (armors.empty()) return std::nullopt;

  auto nearest = armors.front();
  auto min_distance = nearest.xyz_in_gimbal.norm();
  for (const auto & armor : armors) {
    auto distance = armor.xyz_in_gimbal.norm();
    if (distance < min_distance) {
      min_distance = distance;
      nearest = armor;
    }
  }

  return UsbCandidate{nearest, timestamp, source, min_distance};
}

std::optional<UsbCandidate> select_nearest_candidate(
  const std::list<auto_aim::Armor> & left_armors, std::chrono::steady_clock::time_point left_t,
  const std::list<auto_aim::Armor> & right_armors, std::chrono::steady_clock::time_point right_t)
{
  auto left_candidate = get_nearest_candidate(left_armors, left_t, TargetSource::usb_left);
  auto right_candidate = get_nearest_candidate(right_armors, right_t, TargetSource::usb_right);

  if (!left_candidate.has_value()) return right_candidate;
  if (!right_candidate.has_value()) return left_candidate;

  return left_candidate->distance <= right_candidate->distance ? left_candidate : right_candidate;
}

}  // namespace

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{disable-usb    |                        | 禁用左右USB侧视相机，仅使用主海康相机}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  const bool use_usb = !cli.has("disable-usb");
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  auto yaml = tools::load(config_path);
  auto enemy_color =
    tools::read<std::string>(yaml, "enemy_color") == "red" ? auto_aim::Color::red : auto_aim::Color::blue;

  io::Gimbal gimbal(config_path);
  io::ROS2 ros2;
  std::unique_ptr<io::USBCamera> usb_left_camera;
  std::unique_ptr<io::USBCamera> usb_right_camera;
  if (use_usb) {
    usb_left_camera =
      std::make_unique<io::USBCamera>(USB_LEFT_DEVICE, config_path, "usb_left");
    usb_right_camera =
      std::make_unique<io::USBCamera>(USB_RIGHT_DEVICE, config_path, "usb_right");
    // Match `usbcamera_test` startup more closely: let the USB cameras settle before starting HikRobot.
    std::this_thread::sleep_for(USB_STARTUP_SETTLE);
  } else {
    tools::logger()->info("[standard_sentry] USB side cameras disabled.");
  }
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::YOLO usb_yolo(config_path, false);
  auto_buff_fyt::Buff_Detector buff_detector(config_path);

  auto_aim::Solver main_solver(config_path);
  auto_aim::Solver usb_left_solver(config_path);
  auto_aim::Solver usb_right_solver(config_path);
  auto_buff_fyt::Solver buff_solver(config_path);

  auto_aim::Tracker main_tracker(config_path, main_solver);
  auto_aim::Tracker usb_left_tracker(config_path, usb_left_solver);
  auto_aim::Tracker usb_right_tracker(config_path, usb_right_solver);
  auto_aim::Planner planner(config_path);
  auto_buff_fyt::Aimer buff_aimer(config_path);
  auto_buff_fyt::SmallTarget buff_small_target;
  auto_buff_fyt::BigBuffTracker buff_tracker;

  tools::ThreadSafeQueue<TargetCommand, true> target_queue(1);
  target_queue.push(TargetCommand{});

  std::atomic<bool> quit = false;
  std::atomic<bool> main_camera_has_target = false;
  std::atomic<bool> usb_settle_finished = false;
  std::mutex usb_result_mutex;
  UsbThreadResult usb_result;

  auto plan_thread = std::thread([&]() {
    std::optional<uint64_t> stopped_settle_sequence = std::nullopt;
    std::optional<uint64_t> stopped_idle_sequence = std::nullopt;
    std::optional<uint64_t> active_settle_sequence = std::nullopt;
    auto settle_start = std::chrono::steady_clock::now();

    while (!quit) {
      if (
        gimbal.mode() == io::GimbalMode::SMALL_BUFF ||
        gimbal.mode() == io::GimbalMode::BIG_BUFF)
      {
        std::this_thread::sleep_for(10ms);
        continue;
      }

      auto target_command = target_queue.front();
      auto gs = gimbal.state();
      auto plan = auto_aim::Plan{false, false, 0, 0, 0, 0, 0, 0, 0, 0};
      if (target_command.kind == CommandKind::target) {
        plan = planner.plan(target_command.target, gs.bullet_speed);
        apply_usb_offset(plan, gs, target_command.source);
      } else if (
        target_command.kind == CommandKind::fixed_aim ||
        target_command.kind == CommandKind::settle_aim)
      {
        plan = make_fixed_plan(target_command.target_yaw, target_command.target_pitch);
      }

      if (target_command.kind == CommandKind::settle_aim) {
        if (active_settle_sequence != target_command.sequence) {
          active_settle_sequence = target_command.sequence;
          settle_start = std::chrono::steady_clock::now();
        }
      } else {
        active_settle_sequence = std::nullopt;
      }

      const double settle_yaw_error = std::abs(tools::limit_rad(gs.yaw - plan.target_yaw));
      const double settle_pitch_error = std::abs(gs.pitch - plan.target_pitch);
      const bool settle_timeout =
        target_command.kind == CommandKind::settle_aim &&
        std::chrono::steady_clock::now() - settle_start > USB_SETTLE_TIMEOUT;
      const bool settle_reached =
        target_command.kind == CommandKind::settle_aim &&
        ((settle_yaw_error < USB_SETTLE_YAW_THRESH &&
          settle_pitch_error < USB_SETTLE_PITCH_THRESH) ||
         settle_timeout);

      if (settle_reached) {
        if (stopped_settle_sequence != target_command.sequence) {
          stopped_settle_sequence = target_command.sequence;
          usb_settle_finished.store(true);
        }
        plan.control = false;
      } else if (target_command.kind != CommandKind::settle_aim) {
        stopped_settle_sequence = std::nullopt;
      }

      const bool fire = target_command.source == TargetSource::main && plan.fire;
      if (target_command.kind == CommandKind::none) {
        if (stopped_idle_sequence != target_command.sequence) {
          gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
          stopped_idle_sequence = target_command.sequence;
        }
      } else {
        stopped_idle_sequence = std::nullopt;
      }

      if (!settle_reached && target_command.kind != CommandKind::none) {
        gimbal.send(
          plan.control, fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc);
      }

      std::this_thread::sleep_for(10ms);
    }
  });

  auto ros_thread = std::thread([&]() {
    while (!quit) {
      ros2.publish(gimbal.game_status());
      ros2.publish(gimbal.event_data());
      ros2.publish(gimbal.robot_status());
      ros2.publish(gimbal.hurt_data());
      ros2.publish(gimbal.sentry_info());
      ros2.publish(gimbal.rfid_status());
      ros2.publish(gimbal.robot_pos());
      ros2.publish(gimbal.ground_robot_pos());
      ros2.publish(gimbal.game_robot_hp());
      auto gs = gimbal.state();
      ros2.publish(gs.yaw, gs.pitch, gs.yaw_diff);

      ros2.spin_some();

      gimbal.send(
        ros2.getTargetMode(), ros2.getChassisStatus(), ros2.getSentryStatus(), ros2.getCmdVelX(),
        ros2.getCmdVelY(), ros2.getCmdVelZ(), ros2.getTerrainStatus(), ros2.getBumpStatus());
      std::this_thread::sleep_for(20ms);
    }
  });

  std::thread usb_thread;
  if (use_usb) {
    usb_thread = std::thread([&]() {
      while (!quit) {
        const auto mode = gimbal.mode();
        cv::Mat usb_left_img;
        cv::Mat usb_right_img;
        std::chrono::steady_clock::time_point usb_left_t;
        std::chrono::steady_clock::time_point usb_right_t;

        usb_left_camera->read(usb_left_img, usb_left_t);
        if (quit) break;
        usb_right_camera->read(usb_right_img, usb_right_t);
        if (quit) break;

        auto usb_left_q = usb_world_q(gimbal.q(usb_left_t - GIMBAL_DELAY));
        auto usb_right_q = usb_world_q(gimbal.q(usb_right_t - GIMBAL_DELAY));
        usb_left_solver.set_R_gimbal2world(usb_left_q);
        usb_right_solver.set_R_gimbal2world(usb_right_q);

        UsbThreadResult next_usb_result;
        if (main_camera_has_target.load()) {
          next_usb_result.ready = true;
          std::lock_guard<std::mutex> lock(usb_result_mutex);
          usb_result = std::move(next_usb_result);
          std::this_thread::sleep_for(10ms);
          continue;
        }

        auto left_armors = filter_armors(usb_yolo.detect(usb_left_img), enemy_color, mode);
        auto right_armors = filter_armors(usb_yolo.detect(usb_right_img), enemy_color, mode);

        solve_armors(left_armors, usb_left_solver);
        solve_armors(right_armors, usb_right_solver);
        next_usb_result.candidate =
          select_nearest_candidate(left_armors, usb_left_t, right_armors, usb_right_t);

        if (next_usb_result.candidate.has_value()) {
          std::list<auto_aim::Armor> selected_armors = {next_usb_result.candidate->armor};
          std::list<auto_aim::Target> usb_targets;
          next_usb_result.target_command =
            make_usb_fixed_command(next_usb_result.candidate.value(), gimbal.state());

          if (next_usb_result.candidate->source == TargetSource::usb_left) {
            usb_targets = usb_left_tracker.track(selected_armors, next_usb_result.candidate->timestamp, false);
          } else {
            usb_targets =
              usb_right_tracker.track(selected_armors, next_usb_result.candidate->timestamp, false);
          }

          if (!usb_targets.empty()) {
            next_usb_result.target_command.target = usb_targets.front();
            next_usb_result.target_command.source = next_usb_result.candidate->source;
            next_usb_result.target_command.kind = CommandKind::target;
          }
        }

        next_usb_result.ready = true;
        std::lock_guard<std::mutex> lock(usb_result_mutex);
        usb_result = std::move(next_usb_result);
      }
    });
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  OmniState omni_state = OmniState::lost;
  std::optional<TargetCommand> last_usb_command = std::nullopt;
  std::optional<uint64_t> lost_command_sequence = std::nullopt;
  uint64_t command_sequence = 1;

  while (!exiter.exit()) {
    camera.read(img, t);

    auto q = gimbal.q(t - GIMBAL_DELAY);
    auto gs = gimbal.state();
    auto mode = gimbal.mode();
    std::list<auto_aim::Armor> main_armors;
    std::list<auto_aim::Target> main_targets;
    std::vector<auto_buff_fyt::BigBuffCandidate> buff_candidates;
    auto_aim::Plan buff_plan{false, false, 0, 0, 0, 0, 0, 0, 0, 0};

    UsbThreadResult current_usb_result;
    TargetCommand target_command;

    if (mode == io::GimbalMode::SMALL_BUFF || mode == io::GimbalMode::BIG_BUFF) {
      target_queue.push(TargetCommand{});
      main_camera_has_target = false;
      usb_settle_finished = false;
      lost_command_sequence = std::nullopt;

      buff_solver.set_R_gimbal2world(q);
      if (mode == io::GimbalMode::SMALL_BUFF) {
        buff_tracker.reset();
        auto power_rune = buff_detector.detect(img);
        buff_solver.solve(power_rune);
        buff_small_target.get_target(power_rune, t);
        auto target_copy = buff_small_target;
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      } else {
        buff_candidates = buff_detector.detect_candidates(img);
        buff_tracker.update(buff_candidates);

        auto * locked_track = buff_tracker.locked_track();
        if (locked_track != nullptr && locked_track->power_rune.has_value()) {
          buff_solver.solve(locked_track->power_rune);
          locked_track->target.get_target(locked_track->power_rune, t);
          auto target_copy = locked_track->target;
          buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
        }
      }

      gimbal.send(
        buff_plan.control, buff_plan.fire, buff_plan.yaw, buff_plan.yaw_vel, buff_plan.yaw_acc,
        buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);
    } else {
      buff_tracker.reset();

      main_solver.set_R_gimbal2world(q);
      main_armors = filter_armors(yolo.detect(img), enemy_color, mode);
      solve_armors(main_armors, main_solver);
      main_targets = main_tracker.track(main_armors, t);
      const bool main_has_target = !main_armors.empty() || !main_targets.empty();
      main_camera_has_target = main_has_target;

      {
        std::lock_guard<std::mutex> lock(usb_result_mutex);
        current_usb_result = usb_result;
      }

      if (main_has_target) {
        omni_state = OmniState::main_tracking;
        target_command.source = TargetSource::main;
        target_command.omni_state = omni_state;
        if (!main_targets.empty()) {
          target_command.target = main_targets.front();
          target_command.kind = CommandKind::target;
        }
        last_usb_command = std::nullopt;
        lost_command_sequence = std::nullopt;
        usb_settle_finished = false;
      } else if (current_usb_result.ready) {
        if (omni_state == OmniState::lost_from_usb_settle && usb_settle_finished.load()) {
          omni_state = OmniState::lost;
          last_usb_command = std::nullopt;
          if (!lost_command_sequence.has_value()) {
            lost_command_sequence = command_sequence++;
          }
          target_command.omni_state = omni_state;
          target_command.sequence = lost_command_sequence.value();
          usb_settle_finished = false;
        } else {
          target_command = current_usb_result.target_command;
          if (target_command.kind == CommandKind::none && current_usb_result.candidate.has_value()) {
            target_command = make_usb_fixed_command(current_usb_result.candidate.value(), gs);
          }
        }

        if (target_command.kind != CommandKind::none) {
          const bool entering_usb_perception = omni_state != OmniState::usb_perception;
          omni_state = OmniState::usb_perception;
          target_command.omni_state = omni_state;
          target_command.sequence =
            entering_usb_perception || !last_usb_command.has_value()
              ? command_sequence++
              : last_usb_command->sequence;
          last_usb_command = target_command;
          lost_command_sequence = std::nullopt;
        } else if (omni_state == OmniState::usb_perception && last_usb_command.has_value()) {
          omni_state = OmniState::lost_from_usb_settle;
          target_command = last_usb_command.value();
          target_command.kind = CommandKind::settle_aim;
          target_command.omni_state = omni_state;
          target_command.sequence = command_sequence++;
          last_usb_command = target_command;
        } else if (omni_state == OmniState::lost_from_usb_settle && last_usb_command.has_value()) {
          target_command = last_usb_command.value();
          target_command.omni_state = omni_state;
        } else {
          const bool entering_lost = omni_state != OmniState::lost;
          omni_state = OmniState::lost;
          last_usb_command = std::nullopt;
          target_command.omni_state = omni_state;
          if (entering_lost || !lost_command_sequence.has_value()) {
            lost_command_sequence = command_sequence++;
          }
          target_command.sequence = lost_command_sequence.value();
        }
      } else {
        const bool entering_lost = omni_state != OmniState::lost;
        omni_state = OmniState::lost;
        last_usb_command = std::nullopt;
        target_command.omni_state = omni_state;
        if (entering_lost || !lost_command_sequence.has_value()) {
          lost_command_sequence = command_sequence++;
        }
        target_command.sequence = lost_command_sequence.value();
      }

      target_queue.push(target_command);
    }

    ros2.publish(make_target_msg(target_command));

  }

  quit = true;
  if (ros_thread.joinable()) ros_thread.join();
  if (usb_thread.joinable()) usb_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
