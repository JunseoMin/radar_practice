#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <radar_utills.hpp>
#include <cassert>

#include <pmc/pmc.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

class FeatureBasedPoseTracker
{
public:
  /**
   * @brief This class solves pose optimization problem with pmc (graph optimization)
   */
  FeatureBasedPoseTracker() = default;
  FeatureBasedPoseTracker(const double& optim_treshold);
  void setCurrentFrame(cv::Mat& current_frame);

  void showTracked();
  void solve();

  Eigen::Matrix3d getPose();
private:
  struct Keyframe {
    int id;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Eigen::Matrix3d pose; // referenced to world frame

    Keyframe(int id, const cv::Mat& img, const std::vector<cv::KeyPoint>& kps, const cv::Mat& desc, const Eigen::Matrix3d& p) 
    : id(id), image(img), keypoints(kps), descriptors(desc), pose(p){}
  };
  

  struct PoseCostFunctor {
    PoseCostFunctor(const Eigen::Vector2d& P_w, const Eigen::Vector2d& P_t)
    : P_w_(P_w), P_t_(P_t) {}

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const {  // equation(4)
      T theta = pose[2];  // Rotation
      T R[4] = {cos(theta), -sin(theta), sin(theta), cos(theta)};
      T t_x = pose[0], t_y = pose[1];
      // Transform radar keypoint P_t using pose
      T P_t_x = R[0] * T(P_t_(0)) + R[1] * T(P_t_(1)) + t_x;
      T P_t_y = R[2] * T(P_t_(0)) + R[3] * T(P_t_(1)) + t_y;
      // Compute residuals (how far P_t is from P_w)
      residuals[0] = P_t_x - T(P_w_(0));
      residuals[1] = P_t_y - T(P_w_(1));
      return true;
  }
  private:
    const Eigen::Vector2d P_w_;
    const Eigen::Vector2d P_t_;
  };

  int _key_id;
  int _best_match_id;

  double _optim_treshold;

  void _addKeyframe();
  bool _isRequireKeyframe();
  void _getMatchedKeyframe();
  void _findGoodMatch();
  void _findMaxClique();
  void _guessInitialPose();
  void _optimizePose();

  cv::Mat _current_frame;
  cv::Mat _orb_frame;

  cv::Ptr<cv::ORB> _orb;

  //these values represent current frame's
  std::vector<cv::KeyPoint> _keypoints; //query_points
  std::vector<Keyframe> _key_frames;  // map points
  cv::Mat _descriptors;

  Eigen::Matrix2d _rot;
  Eigen::Vector2d _trans;
  Eigen::Matrix3d _pose;
  Eigen::Matrix3d _initial_pose;

  Eigen::MatrixXd _G;  //consistancy matrix

  cv::BFMatcher _matcher;
  std::vector<cv::DMatch> _matches;

  pmc::input _in;

  std::vector<int> _inliers; // contains inlier feature indexes
  std::vector<Eigen::Vector2d> _p_ks; // Key frame's feature coordinate
  std::vector<Eigen::Vector2d> _p_ts; // Query frame's feature point coordinate

};
