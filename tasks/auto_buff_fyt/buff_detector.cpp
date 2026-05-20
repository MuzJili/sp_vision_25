#include "buff_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace auto_buff_fyt
{
namespace
{
EnemyColor parse_buff_detect_color(const std::string & enemy_color)
{
  return enemy_color == "blue" ? EnemyColor::RED : EnemyColor::BLUE;
}

float normalize_angle(float angle)
{
  while (angle > static_cast<float>(CV_PI)) angle -= static_cast<float>(2.0 * CV_PI);
  while (angle < static_cast<float>(-CV_PI)) angle += static_cast<float>(2.0 * CV_PI);
  return angle;
}

cv::Point2f average_fanblade_center(const RuneObject & obj)
{
  return (
           obj.pts.bottom_left + obj.pts.top_left + obj.pts.top_right + obj.pts.bottom_right) /
         4.0f;
}

std::vector<cv::Point2f> armor_points(const RuneObject & obj)
{
  return {obj.pts.bottom_left, obj.pts.top_left, obj.pts.top_right, obj.pts.bottom_right};
}

cv::Point2f four_point_center(const std::vector<cv::Point2f> & points)
{
  cv::Point2f center{0.0F, 0.0F};
  for (int i = 0; i < 4; ++i) center += points[i];
  return center / 4.0F;
}

cv::Point2f armor_center(const std::vector<cv::Point2f> & points)
{
  return std::accumulate(points.begin(), points.end(), cv::Point2f(0.0f, 0.0f)) /
         static_cast<float>(points.size());
}

std::optional<BigBuffCandidate> make_big_buff_candidate(
  const std::vector<cv::Point2f> & blade_points, const cv::Point2f & blade_center,
  const cv::Point2f & r_center, float confidence)
{
  if (blade_points.size() < 4) return std::nullopt;

  std::vector<cv::Point2f> points(blade_points.begin(), blade_points.begin() + 4);
  std::vector<FanBlade> fanblades;
  fanblades.emplace_back(points, blade_center, auto_buff::_light);

  PowerRune power_rune(fanblades, r_center, std::nullopt);
  if (power_rune.is_unsolve()) return std::nullopt;

  BigBuffCandidate candidate;
  candidate.power_rune = power_rune;
  candidate.center = blade_center;
  candidate.confidence = confidence;
  candidate.r_distance = cv::norm(r_center - four_point_center(points));
  return candidate;
}

std::vector<BigBuffCandidate> filter_big_buff_candidates(std::vector<BigBuffCandidate> candidates)
{
  if (candidates.empty()) return candidates;

  const auto max_r_it = std::max_element(
    candidates.begin(), candidates.end(), [](const BigBuffCandidate & a, const BigBuffCandidate & b) {
      return a.r_distance < b.r_distance;
    });
  const auto min_keep_r = max_r_it->r_distance * BIG_BUFF_DEFAULT_R_DISTANCE_KEEP_RATIO;

  candidates.erase(
    std::remove_if(
      candidates.begin(), candidates.end(),
      [min_keep_r](const BigBuffCandidate & candidate) {
        return candidate.r_distance < min_keep_r;
      }),
    candidates.end());

  std::sort(
    candidates.begin(), candidates.end(), [](const BigBuffCandidate & a, const BigBuffCandidate & b) {
      return a.confidence > b.confidence;
    });

  if (candidates.size() > BIG_BUFF_DEFAULT_MAX_TRACKS) {
    candidates.resize(BIG_BUFF_DEFAULT_MAX_TRACKS);
  }
  return candidates;
}
}  // namespace

Buff_Detector::Buff_Detector(const std::string & config_path) : detector_(config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  auto node = yaml["buff_fyt_detector"];
  detect_color_ = parse_buff_detect_color(yaml["enemy_color"].as<std::string>());
  max_candidates_ = node && node["max_candidates"] ? node["max_candidates"].as<int>() : 2;
  min_radius_ratio_ =
    node && node["min_radius_ratio"] ? node["min_radius_ratio"].as<float>() : 0.8f;
  lock_angle_thresh_rad_ =
    node && node["lock_angle_thresh_rad"] ? node["lock_angle_thresh_rad"].as<float>() :
                                            static_cast<float>(CV_PI / 6.0);
  if (max_candidates_ < 1) max_candidates_ = 1;
  if (min_radius_ratio_ < 0.0f) min_radius_ratio_ = 0.0f;
  if (lock_angle_thresh_rad_ < 0.0f) lock_angle_thresh_rad_ = 0.0f;
}

void Buff_Detector::update_filtered_objects(cv::Mat & bgr_img)
{
  last_objects_.clear();
  last_filtered_objects_.clear();
  last_candidates_.clear();
  last_binary_roi_ = cv::Mat::zeros(1, 1, CV_8UC3);

  if (bgr_img.empty()) return;

  cv::Mat rgb_img;
  cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);

  last_objects_ = detector_.detect(rgb_img);
  last_filtered_objects_ = last_objects_;
  last_filtered_objects_.erase(
    std::remove_if(
      last_filtered_objects_.begin(), last_filtered_objects_.end(),
      [this](const RuneObject & obj) { return obj.color != detect_color_; }),
    last_filtered_objects_.end());

  if (last_filtered_objects_.empty()) return;

  std::sort(
    last_filtered_objects_.begin(), last_filtered_objects_.end(),
    [](const RuneObject & a, const RuneObject & b) { return a.prob > b.prob; });

  const float prob_sum = std::accumulate(
    last_filtered_objects_.begin(), last_filtered_objects_.end(), 0.0f,
    [](float sum, const RuneObject & obj) { return sum + obj.prob; });
  const cv::Point2f r_prior = std::accumulate(
    last_filtered_objects_.begin(), last_filtered_objects_.end(), cv::Point2f(0.0f, 0.0f),
    [prob_sum](const cv::Point2f & p, const RuneObject & obj) {
      const float weight = prob_sum > 1e-6f ? obj.prob / prob_sum : 1.0f;
      return p + obj.pts.r_center * weight;
    });

  cv::Point2f r_center;
  if (detector_.use_r_tag()) {
    std::tie(r_center, last_binary_roi_) = detector_.detect_r_tag(bgr_img, r_prior);
  } else {
    r_center = r_prior;
  }

  std::for_each(
    last_filtered_objects_.begin(), last_filtered_objects_.end(),
    [r_center](RuneObject & obj) { obj.pts.r_center = r_center; });
}

std::optional<PowerRune> Buff_Detector::detect(cv::Mat & bgr_img)
{
  update_filtered_objects(bgr_img);

  if (last_filtered_objects_.empty()) return std::nullopt;

  filter_by_radius_ratio(last_filtered_objects_);
  if (last_filtered_objects_.empty()) {
    locked_candidate_.reset();
    return std::nullopt;
  }

  std::vector<RuneObject> inactivated_objects;
  for (const auto & obj : last_filtered_objects_) {
    if (obj.type == RuneType::INACTIVATED) inactivated_objects.push_back(obj);
  }

  auto selected_candidate = select_locked_candidate(inactivated_objects);
  if (!selected_candidate.has_value()) return std::nullopt;

  last_candidates_.push_back(selected_candidate.value());
  for (const auto & obj : inactivated_objects) {
    if (static_cast<int>(last_candidates_.size()) >= max_candidates_) break;
    if (cv::norm(fanblade_center(obj) - fanblade_center(selected_candidate.value())) < 1.0f) continue;
    last_candidates_.push_back(obj);
  }

  return to_power_rune(selected_candidate.value());
}

cv::Point2f Buff_Detector::fanblade_center(const RuneObject & obj) { return average_fanblade_center(obj); }

float Buff_Detector::radius_to_fanblade_center(const RuneObject & obj)
{
  return cv::norm(obj.pts.r_center - fanblade_center(obj));
}

float Buff_Detector::center_angle(const RuneObject & obj)
{
  const auto delta = fanblade_center(obj) - obj.pts.r_center;
  return std::atan2(delta.y, delta.x);
}

std::optional<PowerRune> Buff_Detector::to_power_rune(const RuneObject & obj)
{
  std::vector<cv::Point2f> armor_points = {
    obj.pts.bottom_left,
    obj.pts.top_left,
    obj.pts.top_right,
    obj.pts.bottom_right};
  const cv::Point2f center = fanblade_center(obj);

  std::vector<auto_buff::FanBlade> fanblades;
  fanblades.emplace_back(armor_points, center, auto_buff::_light);
  PowerRune power_rune(fanblades, obj.pts.r_center, std::nullopt);
  if (power_rune.is_unsolve()) return std::nullopt;
  return power_rune;
}

std::vector<BigBuffCandidate> Buff_Detector::detect_candidates(cv::Mat & bgr_img)
{
  update_filtered_objects(bgr_img);

  filter_by_radius_ratio(last_filtered_objects_);
  if (last_filtered_objects_.empty()) {
    locked_candidate_.reset();
    return {};
  }

  std::vector<BigBuffCandidate> candidates;
  candidates.reserve(last_filtered_objects_.size());

  for (const auto & obj : last_filtered_objects_) {
    if (obj.type != RuneType::INACTIVATED) continue;
    if (static_cast<int>(last_candidates_.size()) < max_candidates_) {
      last_candidates_.push_back(obj);
    }

    auto points = armor_points(obj);
    auto candidate =
      make_big_buff_candidate(points, armor_center(points), obj.pts.r_center, obj.prob);
    if (candidate.has_value()) candidates.emplace_back(candidate.value());
  }

  return filter_big_buff_candidates(std::move(candidates));
}

void Buff_Detector::filter_by_radius_ratio(std::vector<RuneObject> & objects) const
{
  if (objects.empty()) return;

  float max_radius = 0.0f;
  for (const auto & obj : objects) {
    max_radius = std::max(max_radius, radius_to_fanblade_center(obj));
  }

  if (max_radius <= 1e-6f) return;

  const float min_radius = max_radius * min_radius_ratio_;
  objects.erase(
    std::remove_if(
      objects.begin(), objects.end(),
      [min_radius](const RuneObject & obj) {
        return radius_to_fanblade_center(obj) < min_radius;
      }),
    objects.end());
}

std::optional<RuneObject> Buff_Detector::select_locked_candidate(
  const std::vector<RuneObject> & candidates)
{
  if (candidates.empty()) {
    locked_candidate_.reset();
    return std::nullopt;
  }

  if (locked_candidate_.has_value()) {
    float best_angle_delta = std::numeric_limits<float>::max();
    auto best_it = candidates.end();
    for (auto it = candidates.begin(); it != candidates.end(); ++it) {
      const float angle_delta = std::abs(normalize_angle(
        center_angle(*it) - center_angle(locked_candidate_.value())));
      if (angle_delta <= lock_angle_thresh_rad_ && angle_delta < best_angle_delta) {
        best_angle_delta = angle_delta;
        best_it = it;
      }
    }
    if (best_it != candidates.end()) {
      locked_candidate_ = *best_it;
      return locked_candidate_;
    }
  }

  locked_candidate_ = candidates.front();
  return locked_candidate_;
}

BigBuffTracker::BigBuffTracker(
  std::size_t max_tracks, float match_distance, int max_lost_frames)
: max_tracks_(max_tracks), match_distance_(match_distance), max_lost_frames_(max_lost_frames)
{
}

void BigBuffTracker::update(const std::vector<BigBuffCandidate> & candidates)
{
  std::vector<bool> candidate_matched(candidates.size(), false);
  std::vector<bool> track_matched(tracks_.size(), false);

  for (std::size_t candidate_i = 0; candidate_i < candidates.size(); ++candidate_i) {
    const auto & candidate = candidates[candidate_i];
    std::optional<std::size_t> best_track_i;
    auto best_distance = match_distance_;

    for (std::size_t track_i = 0; track_i < tracks_.size(); ++track_i) {
      if (track_matched[track_i]) continue;

      const auto distance = static_cast<float>(cv::norm(candidate.center - tracks_[track_i].center));
      if (distance < best_distance) {
        best_distance = distance;
        best_track_i = track_i;
      }
    }

    if (!best_track_i.has_value()) continue;

    auto & track = tracks_[best_track_i.value()];
    track.power_rune = candidate.power_rune;
    track.center = candidate.center;
    track.confidence = candidate.confidence;
    track.lost_count = 0;
    candidate_matched[candidate_i] = true;
    track_matched[best_track_i.value()] = true;
  }

  for (std::size_t track_i = 0; track_i < tracks_.size(); ++track_i) {
    if (track_matched[track_i]) continue;
    ++tracks_[track_i].lost_count;
    tracks_[track_i].power_rune = std::nullopt;
  }

  tracks_.erase(
    std::remove_if(
      tracks_.begin(), tracks_.end(),
      [this](const BigBuffTrack & track) { return track.lost_count > max_lost_frames_; }),
    tracks_.end());

  for (std::size_t candidate_i = 0; candidate_i < candidates.size(); ++candidate_i) {
    if (candidate_matched[candidate_i]) continue;

    if (tracks_.size() >= max_tracks_) {
      auto stale_track_it =
        std::max_element(tracks_.begin(), tracks_.end(), [](const auto & a, const auto & b) {
          return a.lost_count < b.lost_count;
        });
      if (stale_track_it == tracks_.end() || stale_track_it->lost_count == 0) continue;
      tracks_.erase(stale_track_it);
    }

    if (tracks_.size() >= max_tracks_) continue;

    BigBuffTrack track;
    track.id = next_track_id_++;
    track.power_rune = candidates[candidate_i].power_rune;
    track.center = candidates[candidate_i].center;
    track.confidence = candidates[candidate_i].confidence;
    tracks_.emplace_back(std::move(track));
  }

  if (
    locked_track_id_.has_value() &&
    std::none_of(tracks_.begin(), tracks_.end(), [this](const BigBuffTrack & track) {
      return track.id == locked_track_id_.value();
    }))
  {
    locked_track_id_ = std::nullopt;
  }

  if (locked_track_id_.has_value()) {
    const auto locked_track_it =
      std::find_if(tracks_.begin(), tracks_.end(), [this](const BigBuffTrack & track) {
        return track.id == locked_track_id_.value();
      });
    if (locked_track_it == tracks_.end() || locked_track_it->lost_count != 0) {
      locked_track_id_ = std::nullopt;
    }
  }

  if (!locked_track_id_.has_value()) {
    auto best_track_it = std::max_element(
      tracks_.begin(), tracks_.end(), [](const BigBuffTrack & a, const BigBuffTrack & b) {
        if ((a.lost_count == 0) != (b.lost_count == 0)) return a.lost_count != 0;
        return a.confidence < b.confidence;
      });
    if (best_track_it != tracks_.end() && best_track_it->lost_count == 0) {
      locked_track_id_ = best_track_it->id;
    }
  }
}

void BigBuffTracker::reset()
{
  tracks_.clear();
  locked_track_id_ = std::nullopt;
  next_track_id_ = 1;
}

BigBuffTrack * BigBuffTracker::locked_track()
{
  if (!locked_track_id_.has_value()) return nullptr;

  auto track_it = std::find_if(tracks_.begin(), tracks_.end(), [this](const BigBuffTrack & track) {
    return track.id == locked_track_id_.value();
  });
  return track_it == tracks_.end() ? nullptr : &(*track_it);
}

const BigBuffTrack * BigBuffTracker::locked_track() const
{
  if (!locked_track_id_.has_value()) return nullptr;

  auto track_it = std::find_if(tracks_.begin(), tracks_.end(), [this](const BigBuffTrack & track) {
    return track.id == locked_track_id_.value();
  });
  return track_it == tracks_.end() ? nullptr : &(*track_it);
}
}  // namespace auto_buff_fyt
