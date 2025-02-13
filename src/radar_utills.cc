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

  std::cout << "[INFO] radar initialized!" << std::endl;
}

void PolarToCartesian(const std::vector<double>& azimuths, const cv::Mat fft, cv::Mat& cartesian) {
  int N         = azimuths.size();
  int range_max = fft.cols;
  int center    = int(range_max / 2);

  cartesian = cv::Mat::zeros(range_max,range_max, CV_32F);

  std::cout << "cart init" << std::endl;

  for (int i = 0; i < N; i ++){
    int x,y;
    double a = azimuths[i];
    for (int r = 0; r < range_max; r++){
      x = center + r * cos(a);
      y = center + r * sin(a);
      // std::cout << x << " " << y << std::endl;
      // std::cout << "max: " << range_max << std::endl;
      if (x >= 0 && x < range_max && y >= 0 && y < range_max) {
        cartesian.at<float>(y, x) = fft.at<float>(i, r) > 0.1 ? 1 : 0;  // for visualize
      }
    }
  }
}

