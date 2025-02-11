#include <radar_utills.hpp>

int main(void){
  cv::Mat radar;
  cv::Mat radar_cart;
  
  std::vector<double> azimuths;
  std::vector<int64_t> timestemps;

  getRadarOnce("/home/junseo/datasets/MURLAn/drive-download-20250206T045839Z-002/mulan_oxford/polar_oxford_form/1564718964139967887.png", timestemps, azimuths,radar);
  PolarToCartesian(azimuths,radar,radar_cart);

  // viz = eigenToCvMat<double>(radar);
  // cart_viz = eigenToCvMat<double>(radar_cart);

  cv::imshow("radar raw", radar);
  // cv::imshow("radar cart",cart_viz);

  cv::Mat resizedImg;
  cv::resize(radar_cart, resizedImg, cv::Size(1024, 1024)); // (width=800, height=600)

  cv::imshow("Radar Cart", resizedImg);

  cv::waitKey(0);

  return 0;
}