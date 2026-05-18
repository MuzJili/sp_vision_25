#include "usbcamera.hpp"

#include <stdexcept>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace io
{
namespace
{
double read_usb_param(
  const YAML::Node & yaml, const std::string & section, const std::string & key)
{
  if (!section.empty() && yaml[section] && yaml[section][key]) {
    return yaml[section][key].as<double>();
  }
  return tools::read<double>(yaml, key);
}
}  // namespace

USBCamera::USBCamera(
  const std::string & open_name, const std::string & config_path,
  const std::string & config_section)
: open_name_(open_name), config_section_(config_section), quit_(false), ok_(false), queue_(1),
  open_count_(0)
{
  auto yaml = tools::load(config_path);
  image_width_ = read_usb_param(yaml, config_section_, "image_width");
  image_height_ = read_usb_param(yaml, config_section_, "image_height");
  usb_exposure_ = read_usb_param(yaml, config_section_, "usb_exposure");
  usb_frame_rate_ = read_usb_param(yaml, config_section_, "usb_frame_rate");
  usb_gamma_ = read_usb_param(yaml, config_section_, "usb_gamma");
  usb_gain_ = read_usb_param(yaml, config_section_, "usb_gain");
  tools::logger()->info(
    "[USBCamera:{}] config section='{}' width={} height={} fps={} exposure={} gamma={} gain={}",
    open_name_, config_section_.empty() ? "<global>" : config_section_, image_width_, image_height_,
    usb_frame_rate_, usb_exposure_, usb_gamma_, usb_gain_);
  try_open();

  // 守护线程
  daemon_thread_ = std::thread{[this] {
    // tools::logger()->info("daemon thread start");
    while (!quit_) {
      std::this_thread::sleep_for(100ms);

      if (ok_) continue;

      if (open_count_ > 20) {
        tools::logger()->warn("Give up to open {} USB camera", this->device_name);
        quit_ = true;

        {
          std::lock_guard<std::mutex> lock(cap_mutex_);
          close();  // 先关闭摄像头
        }

        if (capture_thread_.joinable()) {
          tools::logger()->warn("Stopping capture thread");
          capture_thread_.join();
        }

        break;
      }

      if (capture_thread_.joinable()) capture_thread_.join();

      {
        std::lock_guard<std::mutex> lock(cap_mutex_);
        close();
      }
      try_open();
    }
    // tools::logger()->info("daemon thread exit");
  }};
}

USBCamera::~USBCamera()
{
  quit_ = true;
  {
    std::lock_guard<std::mutex> lock(cap_mutex_);
    close();
  }
  if (daemon_thread_.joinable()) daemon_thread_.join();
  if (capture_thread_.joinable()) capture_thread_.join();
  tools::logger()->info("USBCamera destructed.");
}

cv::Mat USBCamera::read()
{
  std::lock_guard<std::mutex> lock(cap_mutex_);
  if (!cap_.isOpened()) {
    tools::logger()->warn("Failed to read {} USB camera", this->device_name);
    return cv::Mat();
  }
  cap_ >> img_;
  return img_;
}

void USBCamera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  if (!queue_.pop(data)) {
    img = cv::Mat();
    timestamp = std::chrono::steady_clock::time_point{};
    return;
  }

  img = data.img;
  timestamp = data.timestamp;
}

void USBCamera::open()
{
  std::lock_guard<std::mutex> lock(cap_mutex_);
  std::string true_device_name = "/dev/" + open_name_;
  cap_.open(true_device_name, cv::CAP_V4L);
  if (!cap_.isOpened()) {
    tools::logger()->warn("Failed to open USB camera");
    return;
  }
  device_name = config_section_.empty() ? open_name_ : config_section_;
  sharpness_ = cap_.get(cv::CAP_PROP_SHARPNESS);
  cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, image_width_);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, image_height_);
  cap_.set(cv::CAP_PROP_FPS, usb_frame_rate_);
  cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
  cap_.set(cv::CAP_PROP_GAMMA, usb_gamma_);
  cap_.set(cv::CAP_PROP_GAIN, usb_gain_);
  cap_.set(cv::CAP_PROP_EXPOSURE, usb_exposure_);
  if (sharpness_ == 2) {
    device_name = "left";
  } else if (sharpness_ == 3) {
    device_name = "right";
  }
  tools::logger()->info("{} USBCamera opened", device_name);
  tools::logger()->info(
    "[{} USB camera] actual width={} height={} fps={} auto_exposure={} exposure={} gamma={} gain={}",
    device_name, cap_.get(cv::CAP_PROP_FRAME_WIDTH), cap_.get(cv::CAP_PROP_FRAME_HEIGHT),
    cap_.get(cv::CAP_PROP_FPS), cap_.get(cv::CAP_PROP_AUTO_EXPOSURE),
    cap_.get(cv::CAP_PROP_EXPOSURE), cap_.get(cv::CAP_PROP_GAMMA),
    cap_.get(cv::CAP_PROP_GAIN));
  // 取图线程
  capture_thread_ = std::thread{[this] {
    ok_ = true;
    std::this_thread::sleep_for(50ms);
    tools::logger()->info("[{} USB camera] capture thread started ", this->device_name);
    while (!quit_) {
      std::this_thread::sleep_for(1ms);

      cv::Mat img;
      bool success;
      {
        std::lock_guard<std::mutex> lock(cap_mutex_);
        if (!cap_.isOpened()) {
          break;
        }
        success = cap_.read(img);
      }

      if (!success) {
        tools::logger()->warn("Failed to read frame, exiting capture thread");
        break;
      }

      auto timestamp = std::chrono::steady_clock::now();
      queue_.push({img, timestamp});
    }
    ok_ = false;
  }};
}

void USBCamera::try_open()
{
  try {
    open();
    open_count_++;
  } catch (const std::exception & e) {
    tools::logger()->warn("{}", e.what());
  }
}

void USBCamera::close()
{
  if (cap_.isOpened()) {
    cap_.release();
    tools::logger()->info("USB camera released.");
  }
}

}  // namespace io
