#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Core>

#include <radar_utills.hpp>

class FeatureBasedPoseTracker
{
public:
  /**
   * @brief This class solves pose optimization problem with pmc (graph optimization)
   */
  FeatureBasedPoseTracker() = default;

private:
  std::vector<cv::Mat> _key_frames;

  cv::Mat _current_frame;
  cv::Mat _surf_frame;

  cv::Ptr<cv::ORB> _orb;
};
