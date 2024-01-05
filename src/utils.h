

#ifndef DEF_utils
#define DEF_utils

#include <iostream>
#include <string>
#include <vector>
#include <sys/types.h>
#include <dirent.h>

#include <opencv4/opencv2/opencv.hpp>


void read_directory(const std::string& path, std::vector<std::string>& v, std::string extension);
std::string toUppercase(std::string &data);
std::vector<cv::Rect> readGtboxes(std::string path);
std::vector<cv::Rect> readGtboxes(std::string path);
float average(std::vector<float> const& v);
float stdDev(std::vector<float> const& v, float mean);
bool enlargeRect(cv::Rect &rect, int a=5);
void findCombinedRegions(const cv::Mat &mask, cv::Mat &maskOutput, cv::Mat &smallRegionMask,
                         std::vector<cv::Rect> &rectangles, int minArea=10, bool applyRegionGrowing = true,
                         const cv::Mat &frame =cv::Mat());
void flow2img(const cv::Mat &flow, cv::Mat &magnitude, cv::Mat &img);
#endif
