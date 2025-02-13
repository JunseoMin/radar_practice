#ifndef RADAR_UTILS_HPP
#define RADAR_UTILS_HPP

#define PI 3.14159265358979323846

#include <iostream>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include <filesystem>

struct RadarFFT{
  std::vector<int64_t> timestamp;
  std::vector<double> azimuths;
  cv::Mat fft;

  RadarFFT() = default;
  RadarFFT(std::vector<int64_t>& timestamp, std::vector<double>& azimuths, cv::Mat& fft)
  :timestamp(timestamp), azimuths(azimuths), fft(fft){}
};

struct RadarCartesian{
  std::vector<int64_t> timestamp;
  std::vector<double> azimuths;
  cv::Mat cart_radar;

  RadarCartesian() = default;
  RadarCartesian(std::vector<int64_t>& timestamp, std::vector<double>& azimuths, cv::Mat& cart)
  :timestamp(timestamp), azimuths(azimuths), cart_radar(cart){}
};

void getRadarOnce(const std::string& filepath, std::vector<int64_t> &timestamps, std::vector<double>& azimuths, cv::Mat& fft, int range_bins = 3768);
void PolarToCartesian(const std::vector<double>& azimuths, const cv::Mat fft, cv::Mat& cartesian);
std::vector<std::string> getPathDir(const std::string& filepath);
void getNRadar(const std::string& filepath, std::vector<RadarFFT>& radar_ffts, 
              std::vector<RadarCartesian>& radar_carts, const int& N);

#endif // RADAR_UTILS_HPP
