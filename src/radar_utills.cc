#include "radar_utills.hpp"

void getRadarOnce(const std::string& filepath, std::vector<int64_t>& timestamps, std::vector<double>& azimuths, cv::Mat& fft, int range_bins) {
  /**
   * @brief i'th azimuths contains 2 * M_PI / 5600.0 (rad) data and intensity is fft(i, ranges)
   */
  cv::Mat raw_data   = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
  int     n_azimuths = raw_data.rows;

  azimuths   = std::vector<double> (n_azimuths, 0); // initialize azimuths
  timestamps = std::vector<int64_t> (n_azimuths, 0);

  fft = cv::Mat::zeros(n_azimuths, range_bins, CV_32F);
  #pragma omp parallel
    for (int i = 0; i < n_azimuths; ++i) {
      uchar *byteArray = raw_data.ptr<uchar>(i);
      timestamps[i] = *((int64_t *) (byteArray));                                   // first 8 byte contains timestamp
      azimuths[i]   = *((uint16_t *) (byteArray + 8)) * 2 * M_PI / double(5600.0);  // step size of azimuths angle:  2 * pi / N_s
                                                                                    // from byte 8 ~ 9 contains azimuth value, this line saves azimuths to radian value
                                                                                    // radar output contains azimuth!!
      // valid[i]      = byteArray[10] == 255;
    for (int j = 42; j < range_bins; j++) { // memorize intensity values (byte 42 ~ end contains intensity with type 'int')
      fft.at<float>(i, j - 42) = (float) *(byteArray + 11 + j) / 255.0; // (normalize 0 ~ 1)
    }
  }

}

void PolarToCartesian(const std::vector<double>& azimuths, const cv::Mat fft, cv::Mat& cartesian) {
  int N         = azimuths.size();
  int range_max = fft.cols;
  int center    = int(range_max / 2);

  cartesian = cv::Mat::zeros(range_max, range_max, CV_32F);

  // std::cout << "cart init" << std::endl;

#pragma omp parallel for
  for (int i = 0; i < N; i ++){
    int x,y;
    double a = azimuths[i];
    for (int r = 0; r < range_max; r++){
      x = center + r * cos(a);
      y = center + r * sin(a);
      // std::cout << x << " " << y << std::endl;
      // std::cout << "max: " << range_max << std::endl;
      if (x >= 0 && x < range_max && y >= 0 && y < range_max) {
        cartesian.at<float>(y, x) = fft.at<float>(i, r) > 0.2 ? 1 : 0;  // for visualize
      }
    }
  }
}

std::vector<std::string> getPathDir(const std::string& filepath){
  std::vector<std::string> filepaths;
  
  try{
    for (const auto& entry: std::filesystem::directory_iterator(filepath)){
      if (entry.is_regular_file() && entry.path().extension() == ".png")
      {
        filepaths.emplace_back(entry.path().string());
      }
    }
  }
  catch(const std::exception& e){
    std::cout << "[Error] error during dataset interation! check dataset dir" << std::endl;
  }

  std::sort(filepaths.begin(),filepaths.end());
  return filepaths;
}

void getNRadar(const std::string& filepath, std::vector<RadarFFT>& radar_ffts, 
              std::vector<RadarCartesian>& radar_carts, const int& N){
  
  std::vector<std::string> dirs = getPathDir(filepath);
  std::vector<std::string> dirsSlice(dirs.begin(), dirs.begin() + std::min(N, static_cast<int>(dirs.size())));

  for(const auto& dir : dirsSlice){
    std::vector<int64_t> timestamp;
    std::vector<double> azimuths;
    cv::Mat FFt;
    cv::Mat cart;

    getRadarOnce(dir,timestamp,azimuths,FFt);
    RadarFFT radar_fft(timestamp, azimuths, FFt);

    PolarToCartesian(radar_fft.azimuths, radar_fft.fft, cart);
    RadarCartesian radar_cart(timestamp, azimuths, cart);

    radar_carts.emplace_back(radar_cart);
    radar_ffts.emplace_back(radar_fft);
  }

  std::cout << "[INFO] radar initialized!" << std::endl;
}

