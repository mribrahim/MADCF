// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_MYCSRT_SCALE_ESTIMATION
#define OPENCV_MYCSRT_SCALE_ESTIMATION

#include <opencv4/opencv2/opencv.hpp>

namespace myCSRT
{
    
class DSST {
public:
    DSST() {};
    DSST(const cv::Mat &image, cv::Rect2f bounding_box, cv::Size2f template_size, int numberOfScales,
            float scaleStep, float maxModelArea, float sigmaFactor, float scaleLearnRate);
    ~DSST();
    void update(const cv::Mat &image, const cv::Point2f objectCenter);
    float getScale(const cv::Mat &image, const cv::Point2f objecCenter);
private:
    cv::Mat get_scale_features(cv::Mat img, cv::Point2f pos, cv::Size2f base_target_sz, float current_scale,
            std::vector<float> &scale_factors, cv::Mat scale_window, cv::Size scale_model_sz);

    cv::Size scale_model_sz;
    cv::Mat ys;
    cv::Mat ysf;
    cv::Mat scale_window;
    std::vector<float> scale_factors;
    cv::Mat sf_num;
    cv::Mat sf_den;
    float scale_sigma;
    float min_scale_factor;
    float max_scale_factor;
    float current_scale_factor;
    int scales_count;
    float scale_step;
    float max_model_area;
    float sigma_factor;
    float learn_rate;

    cv::Size original_targ_sz;
};

} /* namespace myCSRT */
#endif
