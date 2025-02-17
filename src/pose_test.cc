// This code realllllllllllllllllllllllly slow..

#include <iostream>
#include "radar_utills.hpp"
#include "FeatureBasedPoseTracker.hpp"

int main(void){
  std::string filepath = "/home/junseo/datasets/MURLAn/drive-download-20250206T045839Z-002/mulan_oxford/polar_oxford_form";
  std::vector<RadarFFT> ffts;
  std::vector<RadarCartesian> carts;

  Eigen::Matrix3d pose;

  int n_iter = 200;
  int frame = 1;

  getNRadar(filepath, ffts, carts, n_iter);
  FeatureBasedPoseTracker tracker(1);

  for(const auto& cartesian : carts){
    std::cout << "current frame: " << frame << " ------------------" << "\n";
    cv::Mat cart = cartesian.cart_radar;
    tracker.setCurrentFrame(cart);
    tracker.solve();
    // tracker.visualizeInliers();
    pose = tracker.getPose();

    std::cout << "solver global pose:" << std::endl;
    std::cout << pose << std::endl;
    std::cout << "-----------------------------------" << "\n";

    cv::waitKey(0);
    frame ++;
  }
  
  return 0;
}