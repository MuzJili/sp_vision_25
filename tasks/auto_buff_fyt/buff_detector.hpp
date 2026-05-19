#ifndef AUTO_BUFF_FYT__BUFF_DETECTOR_HPP
#define AUTO_BUFF_FYT__BUFF_DETECTOR_HPP

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "buff_type.hpp"
#include "buff_target.hpp"
#include "rune_detector.hpp"

namespace auto_buff_fyt
{
constexpr float BIG_BUFF_DEFAULT_R_DISTANCE_KEEP_RATIO = 0.8F;
constexpr float BIG_BUFF_DEFAULT_TRACK_MATCH_DISTANCE = 80.0F;
constexpr int BIG_BUFF_DEFAULT_MAX_TRACKS = 2;
constexpr int BIG_BUFF_DEFAULT_MAX_LOST_FRAMES = 7;

struct BigBuffCandidate
{
  PowerRune power_rune;
  cv::Point2f center;
  float confidence = 0.0F;
  float r_distance = 0.0F;
};

struct BigBuffTrack
{
  int id = 0;
  BigTarget target;
  std::optional<PowerRune> power_rune = std::nullopt;
  cv::Point2f center;
  float confidence = 0.0F;
  int lost_count = 0;
};

class BigBuffTracker
{
public:
  explicit BigBuffTracker(
    std::size_t max_tracks = BIG_BUFF_DEFAULT_MAX_TRACKS,
    float match_distance = BIG_BUFF_DEFAULT_TRACK_MATCH_DISTANCE,
    int max_lost_frames = BIG_BUFF_DEFAULT_MAX_LOST_FRAMES);

  void update(const std::vector<BigBuffCandidate> & candidates);

  void reset();

  BigBuffTrack * locked_track();

  const BigBuffTrack * locked_track() const;

  const std::vector<BigBuffTrack> & tracks() const { return tracks_; }

  std::optional<int> locked_track_id() const { return locked_track_id_; }

  std::size_t track_count() const { return tracks_.size(); }

private:
  std::vector<BigBuffTrack> tracks_;
  std::optional<int> locked_track_id_ = std::nullopt;
  int next_track_id_ = 1;
  std::size_t max_tracks_;
  float match_distance_;
  int max_lost_frames_;
};

class Buff_Detector
{
public:
  explicit Buff_Detector(const std::string & config_path);

  std::optional<PowerRune> detect(cv::Mat & bgr_img);

  std::vector<BigBuffCandidate> detect_candidates(cv::Mat & bgr_img);

  const std::vector<RuneObject> & last_objects() const { return last_objects_; }

  const std::vector<RuneObject> & last_filtered_objects() const { return last_filtered_objects_; }

  const cv::Mat & last_binary_roi() const { return last_binary_roi_; }

private:
  RuneDetector detector_;
  EnemyColor detect_color_;
  std::vector<RuneObject> last_objects_;
  std::vector<RuneObject> last_filtered_objects_;
  cv::Mat last_binary_roi_;

  void update_filtered_objects(cv::Mat & bgr_img);
};
}  // namespace auto_buff_fyt

#endif  // AUTO_BUFF_FYT__BUFF_DETECTOR_HPP
