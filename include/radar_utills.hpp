#ifndef RADAR_UTILS_HPP
#define RADAR_UTILS_HPP

#define PI 3.14159265358979323846

#include <iostream>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

void getRadarOnce(const std::string& filepath, std::vector<int64_t> &timestamps, std::vector<double>& azimuths, cv::Mat& fft, int range_bins = 3768);
void PolarToCartesian(const std::vector<double>& azimuths, const cv::Mat fft, cv::Mat& cartesian);

#endif // RADAR_UTILS_HPP
