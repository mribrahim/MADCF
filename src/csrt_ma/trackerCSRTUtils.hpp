// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_MYCSRT_UTILS
#define OPENCV_MYCSRT_UTILS

#include <fstream>
#include <iostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>

#include <opencv4/opencv2/opencv.hpp>

namespace myCSRT
{
    
inline int modul(int a, int b)
{
    // function calculates the module of two numbers and it takes into account also negative numbers
    return ((a % b) + b) % b;
}

inline double kernel_epan(double x)
{
    return (x <= 1) ? (2.0/3.14)*(1-x) : 0;
}

cv::Mat circshift(cv::Mat matrix, int dx, int dy);
cv::Mat gaussian_shaped_labels(const float sigma, const int w, const int h);
std::vector<cv::Mat> fourier_transform_features(const std::vector<cv::Mat> &M);
cv::Mat divide_complex_matrices(const cv::Mat &A, const cv::Mat &B);
cv::Mat get_subwindow(const cv::Mat &image, const cv::Point2f center,
        const int w, const int h, cv::Rect *valid_pixels = NULL);

float subpixel_peak(const cv::Mat &response, const std::string &s, const cv::Point2f &p);
double get_max(const cv::Mat &m);
double get_min(const cv::Mat &m);

cv::Mat get_hann_win(cv::Size sz);
cv::Mat get_kaiser_win(cv::Size sz, float alpha);
cv::Mat get_chebyshev_win(cv::Size sz, float attenuation);

std::vector<cv::Mat> get_features_rgb(const cv::Mat &patch, const cv::Size &output_size);
std::vector<cv::Mat> get_features_hog(const cv::Mat &im, const int bin_size);
std::vector<cv::Mat> get_features_cn(const cv::Mat &im, const cv::Size &output_size);

cv::Mat bgr2hsv(const cv::Mat &img);

} /* namespace myCSRT */
#endif
