// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "tracker.hpp"
#include "../utils.h"

#include "trackerCSRTSegmentation.hpp"
#include "trackerCSRTUtils.hpp"
#include "trackerCSRTScaleEstimation.hpp"
#include "torchModel.hpp"

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <algorithm>
// using namespace cv;

namespace myCSRT
{

    MyTorchModel torchModel;
    std::vector<cv::Mat> get_features_cnn(const cv::Mat &roi, const cv::Size2i &feature_size)
    {
        return torchModel.getFeatures(roi, feature_size);
    }

    void TrackerCSRTImpl::read(const cv::FileNode &fn)
    {
        params.read(fn);
    }

    void TrackerCSRTImpl::write(cv::FileStorage &fs) // const
    {
        params.write(fs);
    }

    void TrackerCSRTImpl::setInitialMask(cv::InputArray mask)
    {
        preset_mask = mask.getMat();
    }

    bool TrackerCSRTImpl::check_mask_area(const cv::Mat &mat, const double obj_area)
    {
        double threshold = 0.05;
        double mask_area = sum(mat)[0];
        if (mask_area < threshold * obj_area)
        {
            return false;
        }
        return true;
    }

    cv::Mat TrackerCSRTImpl::calculate_response(const cv::Mat &image, const std::vector<cv::Mat> filter)
    {
        cv::Mat patch = get_subwindow(image, object_center, cvFloor(current_scale_factor * template_size.width),
                                      cvFloor(current_scale_factor * template_size.height));
        cv::resize(patch, patch, rescaled_template_size, 0, 0, cv::INTER_CUBIC);
        std::vector<cv::Mat> ftrs = get_features(patch, yf.size());
        std::vector<cv::Mat> Ffeatures = fourier_transform_features(ftrs);
        cv::Mat resp, res;
        if (params.use_channel_weights)
        {
            res = cv::Mat::zeros(Ffeatures[0].size(), CV_32FC2);
            cv::Mat resp_ch;
            cv::Mat mul_mat;
            for (size_t i = 0; i < Ffeatures.size(); ++i)
            {
                mulSpectrums(Ffeatures[i], filter[i], resp_ch, 0, true);
                res += (resp_ch * filter_weights[i]);
            }
            cv::idft(res, res, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
        }
        else
        {
            res = cv::Mat::zeros(Ffeatures[0].size(), CV_32FC2);
            cv::Mat resp_ch;
            for (size_t i = 0; i < Ffeatures.size(); ++i)
            {
                mulSpectrums(Ffeatures[i], filter[i], resp_ch, 0, true);
                res = res + resp_ch;
            }
            cv::idft(res, res, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
        }
        return res;
    }

    void TrackerCSRTImpl::update_csr_filter(const cv::Mat &image, const cv::Mat &mask)
    {
        cv::Mat patch = get_subwindow(image, object_center, cvFloor(current_scale_factor * template_size.width),
                                      cvFloor(current_scale_factor * template_size.height));
        cv::resize(patch, patch, rescaled_template_size, 0, 0, cv::INTER_CUBIC);

        std::vector<cv::Mat> ftrs = get_features(patch, yf.size());
        std::vector<cv::Mat> Fftrs = fourier_transform_features(ftrs);
        std::vector<cv::Mat> new_csr_filter = create_csr_filter(Fftrs, yf, mask);
        // calculate per channel weights
        if (params.use_channel_weights)
        {
            cv::Mat current_resp;
            double max_val;
            float sum_weights = 0;
            std::vector<float> new_filter_weights = std::vector<float>(new_csr_filter.size());
            for (size_t i = 0; i < new_csr_filter.size(); ++i)
            {
                cv::mulSpectrums(Fftrs[i], new_csr_filter[i], current_resp, 0, true);
                cv::idft(current_resp, current_resp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
                cv::minMaxLoc(current_resp, NULL, &max_val, NULL, NULL);
                sum_weights += static_cast<float>(max_val);
                new_filter_weights[i] = static_cast<float>(max_val);
            }
            // update filter weights with new values
            float updated_sum = 0;
            for (size_t i = 0; i < filter_weights.size(); ++i)
            {
                filter_weights[i] = filter_weights[i] * (1.0f - params.weights_lr) +
                                    params.weights_lr * (new_filter_weights[i] / sum_weights);
                updated_sum += filter_weights[i];
            }
            // normalize weights
            for (size_t i = 0; i < filter_weights.size(); ++i)
            {
                filter_weights[i] /= updated_sum;
            }
        }
        for (size_t i = 0; i < csr_filter.size(); ++i)
        {
            csr_filter[i] = (1.0f - params.filter_lr) * csr_filter[i] + params.filter_lr * new_csr_filter[i];
        }
        std::vector<cv::Mat>().swap(ftrs);
        std::vector<cv::Mat>().swap(Fftrs);
    }

    std::vector<cv::Mat> TrackerCSRTImpl::get_features(const cv::Mat &patch, const cv::Size2i &feature_size)
    {
        std::vector<cv::Mat> features;
        if (params.use_hog)
        {
            std::vector<cv::Mat> hog = get_features_hog(patch, cell_size);
            features.insert(features.end(), hog.begin(),
                            hog.begin() + params.num_hog_channels_used);
            
            // // std::cout << "hog: " << hog.size() << std::endl; 
            // for (int i=0; i< 1; i++){
            //     cv::Mat temp;
            //     // cv::normalize(hog[i], temp, 0, 255, cv::NORM_MINMAX);
            //     cv::imshow("HOG", hog[i]);
            //     // cv::waitKey(0);
            // }
        }
        if (params.use_color_names)
        {
            std::vector<cv::Mat> cn;
            cn = get_features_cn(patch, feature_size);
            features.insert(features.end(), cn.begin(), cn.end());
        }
        if (params.use_gray)
        {
            cv::Mat gray_m;
            cv::cvtColor(patch, gray_m, cv::COLOR_BGR2GRAY);
            cv::resize(gray_m, gray_m, feature_size, 0, 0, cv::INTER_CUBIC);
            gray_m.convertTo(gray_m, CV_32FC1, 1.0 / 255.0, -0.5);
            features.push_back(gray_m);
        }
        if (params.use_rgb)
        {
            std::vector<cv::Mat> rgb_features = get_features_rgb(patch, feature_size);
            features.insert(features.end(), rgb_features.begin(), rgb_features.end());
        }

        //    if (diff.empty()){
        //        diff = cv::Mat::zeros(cv::Size(1280,720), 0);
        //    }
        //        cv::Mat diff_m;
        //        cv::Mat patchDiff = get_subwindow(diff, object_center, cvFloor(current_scale_factor * template_size.width),
        //                                      cvFloor(current_scale_factor * template_size.height));
        //        cv::resize(patchDiff, patchDiff, feature_size, 0, 0, cv::INTER_CUBIC);
        //        patchDiff.convertTo(diff_m, CV_32FC1, 1.0/255.0);
        //        features.push_back(diff_m);

        if (params.use_cnn_torch)
        {
            std::vector<cv::Mat> cnn_features = get_features_cnn(patch, feature_size);
            features.insert(features.end(), cnn_features.begin(), cnn_features.end());
        }

        for (size_t i = 0; i < features.size(); ++i)
        {
            features.at(i) = features.at(i).mul(window);
        }
        return features;
    }

    // class ParallelLoopBodyMy
    // {
    // public:
    //     virtual ~ParallelLoopBodyMy();
    //     virtual void operator() (const Range& range) const = 0;
    // };

    class ParallelCreateCSRFilter //: public ParallelLoopBodyMy
    {
    public:
        ParallelCreateCSRFilter(
            const std::vector<cv::Mat> img_features,
            const cv::Mat Y,
            const cv::Mat P,
            int admm_iterations,
            std::vector<cv::Mat> &result_filter_) : result_filter(result_filter_)
        {
            this->img_features = img_features;
            this->Y = Y;
            this->P = P;
            this->admm_iterations = admm_iterations;
        }
        virtual void operator()(const cv::Range &range) // const CV_OVERRIDE
        {
            for (int i = range.start; i < range.end; i++)
            {
                float mu = 5.0f;
                float beta = 3.0f;
                float mu_max = 20.0f;
                float lambda = mu / 100.0f;

                cv::Mat F = img_features[i];

                cv::Mat Sxy, Sxx;
                cv::mulSpectrums(F, Y, Sxy, 0, true);
                cv::mulSpectrums(F, F, Sxx, 0, true);

                cv::Mat H;
                H = divide_complex_matrices(Sxy, (Sxx + lambda));
                idft(H, H, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
                H = H.mul(P);
                dft(H, H, cv::DFT_COMPLEX_OUTPUT);
                cv::Mat L = cv::Mat::zeros(H.size(), H.type()); // Lagrangian multiplier
                cv::Mat G;
                for (int iteration = 0; iteration < admm_iterations; ++iteration)
                {
                    G = divide_complex_matrices((Sxy + (mu * H) - L), (Sxx + mu));
                    idft((mu * G) + L, H, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
                    float lm = 1.0f / (lambda + mu);
                    H = H.mul(P * lm);
                    cv::dft(H, H, cv::DFT_COMPLEX_OUTPUT);

                    // Update variables for next iteration
                    L = L + mu * (G - H);
                    mu = std::min(mu_max, beta * mu);
                }
                result_filter[i] = H;
            }
        }

        ParallelCreateCSRFilter &operator=(const ParallelCreateCSRFilter &)
        {
            return *this;
        }

    private:
        int admm_iterations;
        cv::Mat Y;
        cv::Mat P;
        std::vector<cv::Mat> img_features;
        std::vector<cv::Mat> &result_filter;
    };

    std::vector<cv::Mat> TrackerCSRTImpl::create_csr_filter(
        const std::vector<cv::Mat> img_features,
        const cv::Mat Y,
        const cv::Mat P)
    {
        std::vector<cv::Mat> result_filter;
        result_filter.resize(img_features.size());
        ParallelCreateCSRFilter parallelCreateCSRFilter(img_features, Y, P,
                                                        params.admm_iterations, result_filter);
        cv::parallel_for_(cv::Range(0, static_cast<int>(result_filter.size())), parallelCreateCSRFilter);

        return result_filter;
    }

    cv::Mat TrackerCSRTImpl::get_location_prior(
        const cv::Rect roi,
        const cv::Size2f target_size,
        const cv::Size img_sz)
    {
        int x1 = cvRound(cv::max(cv::min(roi.x - 1, img_sz.width - 1), 0));
        int y1 = cvRound(cv::max(cv::min(roi.y - 1, img_sz.height - 1), 0));

        int x2 = cvRound(cv::min(cv::max(roi.width - 1, 0), img_sz.width - 1));
        int y2 = cvRound(cv::min(cv::max(roi.height - 1, 0), img_sz.height - 1));

        cv::Size target_sz;
        target_sz.width = target_sz.height = cvFloor(cv::min(target_size.width, target_size.height));

        double cx = x1 + (x2 - x1) / 2.;
        double cy = y1 + (y2 - y1) / 2.;
        double kernel_size_width = 1.0 / (0.5 * static_cast<double>(target_sz.width) * 1.4142 + 1);
        double kernel_size_height = 1.0 / (0.5 * static_cast<double>(target_sz.height) * 1.4142 + 1);

        cv::Mat kernel_weight = cv::Mat::zeros(1 + cvFloor(y2 - y1), 1 + cvFloor(-(x1 - cx) + (x2 - cx)), CV_64FC1);
        for (int y = y1; y < y2 + 1; ++y)
        {
            double *weightPtr = kernel_weight.ptr<double>(y);
            double tmp_y = std::pow((cy - y) * kernel_size_height, 2);
            for (int x = x1; x < x2 + 1; ++x)
            {
                weightPtr[x] = kernel_epan(std::pow((cx - x) * kernel_size_width, 2) + tmp_y);
            }
        }

        double max_val;
        cv::minMaxLoc(kernel_weight, NULL, &max_val, NULL, NULL);
        cv::Mat fg_prior = kernel_weight / max_val;
        fg_prior.setTo(0.5, fg_prior < 0.5);
        fg_prior.setTo(0.9, fg_prior > 0.9);
        return fg_prior;
    }

    cv::Mat TrackerCSRTImpl::segment_region(
        const cv::Mat &image,
        const cv::Point2f &object_center,
        const cv::Size2f &template_size,
        const cv::Size &target_size,
        float scale_factor)
    {
        cv::Rect valid_pixels;
        cv::Mat patch = get_subwindow(image, object_center, cvFloor(scale_factor * template_size.width),
                                      cvFloor(scale_factor * template_size.height), &valid_pixels);
        cv::Size2f scaled_target = cv::Size2f(target_size.width * scale_factor,
                                              target_size.height * scale_factor);
        cv::Mat fg_prior = get_location_prior(
            cv::Rect(0, 0, patch.size().width, patch.size().height),
            scaled_target, patch.size());

        std::vector<cv::Mat> img_channels;
        cv::split(patch, img_channels);
        std::pair<cv::Mat, cv::Mat> probs = Segment::computePosteriors2(img_channels, 0, 0, patch.cols, patch.rows,
                                                                        p_b, fg_prior, 1.0 - fg_prior, hist_foreground, hist_background);

        cv::Mat mask = cv::Mat::zeros(probs.first.size(), probs.first.type());
        probs.first(valid_pixels).copyTo(mask(valid_pixels));
        double max_resp = get_max(mask);
        cv::threshold(mask, mask, max_resp / 2.0, 1, cv::THRESH_BINARY);
        mask.convertTo(mask, CV_32FC1, 1.0);
        return mask;
    }

    void TrackerCSRTImpl::extract_histograms(const cv::Mat &image, cv::Rect region, Histogram &hf, Histogram &hb)
    {
        // get coordinates of the region
        int x1 = std::min(std::max(0, region.x), image.cols - 1);
        int y1 = std::min(std::max(0, region.y), image.rows - 1);
        int x2 = std::min(std::max(0, region.x + region.width), image.cols - 1);
        int y2 = std::min(std::max(0, region.y + region.height), image.rows - 1);

        // calculate coordinates of the background region
        int offsetX = (x2 - x1 + 1) / params.background_ratio;
        int offsetY = (y2 - y1 + 1) / params.background_ratio;
        int outer_y1 = std::max(0, (int)(y1 - offsetY));
        int outer_y2 = std::min(image.rows, (int)(y2 + offsetY + 1));
        int outer_x1 = std::max(0, (int)(x1 - offsetX));
        int outer_x2 = std::min(image.cols, (int)(x2 + offsetX + 1));

        // calculate probability for the background
        p_b = 1.0 - ((x2 - x1 + 1) * (y2 - y1 + 1)) /
                        ((double)(outer_x2 - outer_x1 + 1) * (outer_y2 - outer_y1 + 1));

        // split multi-channel image into the std::vector of matrices
        std::vector<cv::Mat> img_channels(image.channels());
        split(image, img_channels);
        for (size_t k = 0; k < img_channels.size(); k++)
        {
            img_channels.at(k).convertTo(img_channels.at(k), CV_8UC1);
        }

        hf.extractForegroundHistogram(img_channels, cv::Mat(), false, x1, y1, x2, y2);
        hb.extractBackGroundHistogram(img_channels, x1, y1, x2, y2,
                                      outer_x1, outer_y1, outer_x2, outer_y2);
        std::vector<cv::Mat>().swap(img_channels);
    }

    void TrackerCSRTImpl::update_histograms(const cv::Mat &image, const cv::Rect &region)
    {
        // create temporary histograms
        Histogram hf(image.channels(), params.histogram_bins);
        Histogram hb(image.channels(), params.histogram_bins);
        extract_histograms(image, region, hf, hb);

        // get histogram vectors from temporary histograms
        std::vector<double> hf_vect_new = hf.getHistogramVector();
        std::vector<double> hb_vect_new = hb.getHistogramVector();
        // get histogram vectors from learned histograms
        std::vector<double> hf_vect = hist_foreground.getHistogramVector();
        std::vector<double> hb_vect = hist_background.getHistogramVector();

        // update histograms - use learning rate
        for (size_t i = 0; i < hf_vect.size(); i++)
        {
            hf_vect_new[i] = (1 - params.histogram_lr) * hf_vect[i] +
                             params.histogram_lr * hf_vect_new[i];
            hb_vect_new[i] = (1 - params.histogram_lr) * hb_vect[i] +
                             params.histogram_lr * hb_vect_new[i];
        }

        // set learned histograms
        hist_foreground.setHistogramVector(&hf_vect_new[0]);
        hist_background.setHistogramVector(&hb_vect_new[0]);

        std::vector<double>().swap(hf_vect);
        std::vector<double>().swap(hb_vect);
    }

    cv::Point2f TrackerCSRTImpl::estimate_new_position(const cv::Mat &image, double &max_val)
    {

        cv::Mat resp = calculate_response(image, csr_filter);

        cv::Point max_loc;
        cv::minMaxLoc(resp, NULL, &max_val, NULL, &max_loc);
        // cv::Point min_loc;
        // double min_val;
        // cv::minMaxLoc(resp, &min_val, &max_val, &min_loc, &max_loc);
        // std::cout << resp.channels() << ", "<< min_val << " , " << max_val << std::endl;
        // cv::Scalar mean_val = cv::mean(resp);
        // std::cout << "resp mean: " << mean_val << std::endl;
        // std::cout << "csr_filter: " << csr_filter.size() << "  channels: "<< csr_filter[0].channels()<< std::endl;

        // std::vector<cv::Mat> channels(2);
        
        // for (int i = 0; i< 1; i++){
        //     cv::split(csr_filter[i], channels);
        //     cv::normalize(channels[0], temp, 0, 255, cv::NORM_MINMAX);
        //     std::cout << "temp shape: " << temp.size() << std::endl; 
        //     cv::Mat tempU;
        //     temp.convertTo(tempU, CV_8UC1);
        //     // cv::multiply(resp, 100, temp);
        //     cv::imshow("calculate_response", tempU);
        //     // cv::waitKey(0);
        // }

        // object_center = cv::Point2f(static_cast<float>(bounding_box.x) + original_target_size.width / 2.0f,
        //                             static_cast<float>(bounding_box.y) + original_target_size.height / 2.0f);
        // cv::Mat patch = get_subwindow(image, object_center, cvFloor(current_scale_factor * template_size.width),
        //                               cvFloor(current_scale_factor * template_size.height));
        


        // cv::Mat temp;
        // cv::normalize(resp, temp, 0, 255, cv::NORM_MINMAX);
        // // cv::absdiff(255, temp, temp);
        // cv::Mat tempU;
        // temp.convertTo(tempU, CV_8UC1);
        
        // // mean_val = cv::mean(tempU);
        // // std::cout << "normalized mean: " << mean_val << std::endl;
        // // std::cout << "max_loc: " << max_loc << std::endl;
        // // std::cout << "resp dim: " << resp.size() << std::endl;
        // // std::cout << "patch dim: " << patch.size() << "  filter dim :"<< csr_filter[0].size()<< std::endl;
       
        // int cx = max_loc.x * patch.cols / resp.cols + patch.cols / 2;
        // int cy = max_loc.y * patch.rows / resp.rows + patch.rows / 2;
        // cv::circle(patch, cv::Size(cx, cy), 2, cv::Scalar(0,0,255), -1);
        // cv::putText(patch, std::to_string(max_val), cv::Point2d(cx-10, cy-10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
        // cv::resize(tempU, tempU, cv::Size(224,224));
        // cv::resize(patch, patch, cv::Size(224,224));
        // cv::imshow("Response", tempU);
        // cv::imshow("patch", patch);

        // cv::Point2f object_center_yedek = object_center;
        // object_center = cv::Point2f(object_center.x + 100, object_center.y + 100);
        // cv::Mat patch2 = get_subwindow(image, object_center, cvFloor(current_scale_factor * template_size.width),
        //                               cvFloor(current_scale_factor * template_size.height));
        
        
        // cv::Mat resp2 = calculate_response(image, csr_filter);
        // cv::Point max_loc2;
        // double max_val2;
        // cv::minMaxLoc(resp2, NULL, &max_val2, NULL, &max_loc2);

        // cv::normalize(resp2, temp, 0, 255, cv::NORM_MINMAX);
        // // cv::absdiff(255, temp, temp);
        // temp.convertTo(tempU, CV_8UC1);
        // cx = max_loc2.x * patch2.cols / resp.cols + patch2.cols / 2;
        // cy = max_loc2.y * patch2.rows / resp.rows + patch2.rows / 2;
        // cv::circle(patch2, cv::Size(cx, cy), 2, cv::Scalar(0,0,255), -1);
        // cv::putText(patch2, std::to_string(max_val2), cv::Point2d(cx-10, cy-10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
        // cv::resize(tempU, tempU, cv::Size(224,224));
        // cv::resize(patch2, patch2, cv::Size(224,224));
        // cv::imshow("Response-2", tempU);
        // cv::imshow("patch-2", patch2);

        // object_center = object_center_yedek;

        if (max_val < params.psr_threshold)
            return cv::Point2f(-1, -1); // target "lost"

        // take into account also subpixel accuracy
        float col = ((float)max_loc.x) + subpixel_peak(resp, "horizontal", max_loc);
        float row = ((float)max_loc.y) + subpixel_peak(resp, "vertical", max_loc);
        if (row + 1 > (float)resp.rows / 2.0f)
        {
            row = row - resp.rows;
        }
        if (col + 1 > (float)resp.cols / 2.0f)
        {
            col = col - resp.cols;
        }
        // calculate x and y displacements
        cv::Point2f new_center = object_center + cv::Point2f(current_scale_factor * (1.0f / rescale_ratio) * cell_size * (col),
                                                             current_scale_factor * (1.0f / rescale_ratio) * cell_size * (row));
        // sanity checks
        if (new_center.x < 0)
            new_center.x = 0;
        if (new_center.x >= image_size.width)
            new_center.x = static_cast<float>(image_size.width - 1);
        if (new_center.y < 0)
            new_center.y = 0;
        if (new_center.y >= image_size.height)
            new_center.y = static_cast<float>(image_size.height - 1);

        return new_center;
    }

    // *********************************************************************
    // *                        Update API function                        *
    // *********************************************************************
    bool TrackerCSRTImpl::update(const cv::Mat &image_, cv::Rect2d &boundingBox, const cv::Mat &homoMat)
    {
        cv::Mat image;
        if (image_.channels() == 1) // treat gray image as color image
            cv::cvtColor(image_, image, cv::COLOR_GRAY2BGR);
        else
            image = image_;

        double max_val;
        object_center = estimate_new_position(image, max_val);
       
        // float THRESH_DISTANCE = std::max(bounding_box.width,bounding_box.height)/2;
        float averageResponse = 0, maxResponse = 0;
        max_val_history.push_back(max_val);
        if (max_val_history.size() > HISTORY_COUNT)
        {
            max_val_history.erase(max_val_history.begin());
        }
        maxResponse = *std::max_element(max_val_history.begin(), max_val_history.end());
        averageResponse = average(max_val_history);
        float stdDevResponse = stdDev(max_val_history, averageResponse);
        if (maxResponse < averageResponse)
        {
            maxResponse = averageResponse;
        }
        // std::cout << "max val:" << max_val << "  averageResponse: " << averageResponse << "  maxResponse: " << maxResponse << std::endl;

        cv::Mat temp, framePreviousWarped;
        cv::warpPerspective(framePrevious, framePreviousWarped, homoMat, image.size());
        cv::absdiff(image, framePreviousWarped, diff);
        //    cv::imshow("diff image", diff);
        cv::cvtColor(diff, temp, cv::COLOR_BGR2GRAY);
        cv::threshold(temp, diff, 40, 255, cv::THRESH_BINARY);
        // cv::imshow("diff mask", diff);
        if (counter_check >= pointsGrid.size())
        {
            counter_check = 0;
        }

        cv::Rect rect_temp = cv::Rect(object_center.x - 50, object_center.y - 50, 100, 100);
        if (rect_temp.x < 0)
        {
            rect_temp.x = 0;
        }
        if (rect_temp.y < 0)
        {
            rect_temp.y = 0;
        }
        if (rect_temp.x + rect_temp.width >= image.cols || rect_temp.y + rect_temp.height >= image.rows)
        {
            rect_temp = cv::Rect2f();
        }
        cv::Mat diff_patch = diff(rect_temp);

        if (max_val < THRESH_RESPONSE)
        {

            int px = pointsGrid.at(counter_check).x;
            int py = pointsGrid.at(counter_check).y;
            cv::Rect rect = cv::Rect(px - 50, py - 50, 100, 100);
            if (rect.x >= 0 && rect.y >= 0)
            {
                cv::Mat roi = diff(rect);
                if (cv::countNonZero(roi) > 10)
                {
                    object_center.x = px;
                    object_center.y = py;

                    double max_val_temp;
                    cv::Point2f object_center_temp = estimate_new_position(image, max_val_temp);
                    if (max_val_temp > max_val)
                    {
                        points_history.clear();
                        max_val_history.clear();
                        max_val = max_val_temp;
                        object_center = object_center_temp;
                        // std::cout<<"update position: " << object_center << "  max_val:"<< max_val<< std::endl;
                        // cv::Mat roi = image(rect);
                        // cv::imshow("roi", roi);
                        // cv::waitKey(0);
                    }
                }
            }

            counter_check++;
        }

        if (max_val < THRESH_RESPONSE)
        {
            image.copyTo(framePrevious);
            return false;
        }

        current_scale_factor = dsst.getScale(image, object_center);
        // update bouding_box according to new scale and location
        bounding_box.x = object_center.x - current_scale_factor * original_target_size.width / 2.0f;
        bounding_box.y = object_center.y - current_scale_factor * original_target_size.height / 2.0f;
        bounding_box.width = current_scale_factor * original_target_size.width;
        bounding_box.height = current_scale_factor * original_target_size.height;

        // update tracker
        if (params.use_segmentation)
        {
            cv::Mat hsv_img = bgr2hsv(image);
            update_histograms(hsv_img, bounding_box);
            filter_mask = segment_region(hsv_img, object_center,
                                         template_size, original_target_size, current_scale_factor);
            cv::resize(filter_mask, filter_mask, yf.size(), 0, 0, cv::INTER_NEAREST);
            if (check_mask_area(filter_mask, default_mask_area))
            {
                cv::dilate(filter_mask, filter_mask, erode_element);
            }
            else
            {
                filter_mask = default_mask;
            }
        }
        else
        {
            filter_mask = default_mask;
        }

        // if (dist > THRESH_DISTANCE) {
        if (max_val > THRESH_RESPONSE)
        {
            update_csr_filter(image, filter_mask);
            dsst.update(image, object_center);
        }

        boundingBox = bounding_box;
        image.copyTo(framePrevious);
        return true;
    }

    // *********************************************************************
    // *                        Init API function                          *
    // *********************************************************************
    bool TrackerCSRTImpl::init(cv::Mat &image_, const cv::Rect2d &boundingBox)
    {
        cv::Mat image;
        if (image_.channels() == 1) // treat gray image as color image
            cvtColor(image_, image, cv::COLOR_GRAY2BGR);
        else
            image = image_;

        image.copyTo(framePrevious);

        int gridSizeW = 100;
        int gridSizeH = 100;
        for (int i = 0; i < image.cols / gridSizeW; ++i)
        {
            for (int j = 0; j < image.rows / gridSizeH; ++j)
            {
                pointsGrid.push_back(cv::Vec2f(i * gridSizeW + gridSizeW / 2.0, j * gridSizeH + gridSizeH / 2.0));
            }
        }

        current_scale_factor = 1.0;
        image_size = image.size();
        bounding_box = boundingBox;
        cell_size = cvFloor(std::min(4.0, std::max(1.0, static_cast<double>(
                                                            cvCeil((bounding_box.width * bounding_box.height) / 400.0)))));
        original_target_size = cv::Size(bounding_box.size());

        template_size.width = static_cast<float>(cvFloor(original_target_size.width + params.padding *
                                                                                          sqrt(original_target_size.width * original_target_size.height)));
        template_size.height = static_cast<float>(cvFloor(original_target_size.height + params.padding *
                                                                                            sqrt(original_target_size.width * original_target_size.height)));
        template_size.width = template_size.height =
            (template_size.width + template_size.height) / 2.0f;
        rescale_ratio = sqrt(pow(params.template_size, 2) / (template_size.width * template_size.height));
        if (rescale_ratio > 1)
        {
            rescale_ratio = 1;
        }
        rescaled_template_size = cv::Size2i(cvFloor(template_size.width * rescale_ratio),
                                            cvFloor(template_size.height * rescale_ratio));
        object_center = cv::Point2f(static_cast<float>(boundingBox.x) + original_target_size.width / 2.0f,
                                    static_cast<float>(boundingBox.y) + original_target_size.height / 2.0f);

        yf = gaussian_shaped_labels(params.gsl_sigma,
                                    rescaled_template_size.width / cell_size, rescaled_template_size.height / cell_size);
        if (params.window_function.compare("hann") == 0)
        {
            window = get_hann_win(cv::Size(yf.cols, yf.rows));
        }
        else if (params.window_function.compare("cheb") == 0)
        {
            window = get_chebyshev_win(cv::Size(yf.cols, yf.rows), params.cheb_attenuation);
        }
        else if (params.window_function.compare("kaiser") == 0)
        {
            window = get_kaiser_win(cv::Size(yf.cols, yf.rows), params.kaiser_alpha);
        }
        else
        {
            std::cout << "Not a valid window function" << std::endl;
            return false;
        }

        cv::Size2i scaled_obj_size = cv::Size2i(cvFloor(original_target_size.width * rescale_ratio / cell_size),
                                                cvFloor(original_target_size.height * rescale_ratio / cell_size));
        // set dummy mask and area;
        int x0 = std::max((yf.size().width - scaled_obj_size.width) / 2 - 1, 0);
        int y0 = std::max((yf.size().height - scaled_obj_size.height) / 2 - 1, 0);
        default_mask = cv::Mat::zeros(yf.size(), CV_32FC1);
        default_mask(cv::Rect(x0, y0, scaled_obj_size.width, scaled_obj_size.height)) = 1.0f;
        default_mask_area = static_cast<float>(sum(default_mask)[0]);

        // initalize segmentation
        if (params.use_segmentation)
        {
            cv::Mat hsv_img = bgr2hsv(image);
            hist_foreground = Histogram(hsv_img.channels(), params.histogram_bins);
            hist_background = Histogram(hsv_img.channels(), params.histogram_bins);
            extract_histograms(hsv_img, bounding_box, hist_foreground, hist_background);
            filter_mask = segment_region(hsv_img, object_center, template_size,
                                         original_target_size, current_scale_factor);
            // update calculated mask with preset mask
            if (preset_mask.data)
            {
                cv::Mat preset_mask_padded = cv::Mat::zeros(filter_mask.size(), filter_mask.type());
                int sx = std::max((int)cvFloor(preset_mask_padded.cols / 2.0f - preset_mask.cols / 2.0f) - 1, 0);
                int sy = std::max((int)cvFloor(preset_mask_padded.rows / 2.0f - preset_mask.rows / 2.0f) - 1, 0);
                preset_mask.copyTo(preset_mask_padded(
                    cv::Rect(sx, sy, preset_mask.cols, preset_mask.rows)));
                filter_mask = filter_mask.mul(preset_mask_padded);
            }
            erode_element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));
            cv::resize(filter_mask, filter_mask, yf.size(), 0, 0, cv::INTER_NEAREST);
            if (check_mask_area(filter_mask, default_mask_area))
            {
                dilate(filter_mask, filter_mask, erode_element);
            }
            else
            {
                filter_mask = default_mask;
            }
        }
        else
        {
            filter_mask = default_mask;
        }

        // initialize filter
        cv::Mat patch = get_subwindow(image, object_center, cvFloor(current_scale_factor * template_size.width),
                                      cvFloor(current_scale_factor * template_size.height));
        cv::resize(patch, patch, rescaled_template_size, 0, 0, cv::INTER_CUBIC);
        std::vector<cv::Mat> patch_ftrs = get_features(patch, yf.size());
        std::vector<cv::Mat> Fftrs = fourier_transform_features(patch_ftrs);
        csr_filter = create_csr_filter(Fftrs, yf, filter_mask);

        if (params.use_channel_weights)
        {
            cv::Mat current_resp;
            filter_weights = std::vector<float>(csr_filter.size());
            float chw_sum = 0;
            for (size_t i = 0; i < csr_filter.size(); ++i)
            {
                cv::mulSpectrums(Fftrs[i], csr_filter[i], current_resp, 0, true);
                cv::idft(current_resp, current_resp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
                double max_val;
                cv::minMaxLoc(current_resp, NULL, &max_val, NULL, NULL);
                chw_sum += static_cast<float>(max_val);
                filter_weights[i] = static_cast<float>(max_val);
            }
            for (size_t i = 0; i < filter_weights.size(); ++i)
            {
                filter_weights[i] /= chw_sum;
            }
        }

        // initialize scale search
        dsst = DSST(image, bounding_box, template_size, params.number_of_scales, params.scale_step,
                    params.scale_model_max_area, params.scale_sigma_factor, params.scale_lr);

        //    model=Ptr<TrackerCSRTModel>(new TrackerCSRTModel(params));
        isInit = true;
        return true;
    }

    TrackerCSRTImpl::Params::Params()
    {
        use_channel_weights = true;
        use_segmentation = true;
        use_cnn_torch = false;
        use_hog = true;
        use_color_names = true;
        use_gray = true;
        use_rgb = false;
        window_function = "hann";
        kaiser_alpha = 3.75f;
        cheb_attenuation = 45;
        padding = 3.0f;
        template_size = 200;
        gsl_sigma = 1.0f;
        hog_orientations = 9;
        hog_clip = 0.2f;
        num_hog_channels_used = 18;
        filter_lr = 0.02f;
        weights_lr = 0.02f;
        admm_iterations = 4;
        number_of_scales = 33;
        scale_sigma_factor = 0.250f;
        scale_model_max_area = 512.0f;
        scale_lr = 0.025f;
        scale_step = 1.020f;
        histogram_bins = 16;
        background_ratio = 2;
        histogram_lr = 0.04f;
        psr_threshold = 0.035f;
    }

    void TrackerCSRTImpl::Params::read(const cv::FileNode &fn)
    {
        *this = TrackerCSRTImpl::Params();
        if (!fn["padding"].empty())
            fn["padding"] >> padding;
        if (!fn["template_size"].empty())
            fn["template_size"] >> template_size;
        if (!fn["gsl_sigma"].empty())
            fn["gsl_sigma"] >> gsl_sigma;
        if (!fn["hog_orientations"].empty())
            fn["hog_orientations"] >> hog_orientations;
        if (!fn["num_hog_channels_used"].empty())
            fn["num_hog_channels_used"] >> num_hog_channels_used;
        if (!fn["hog_clip"].empty())
            fn["hog_clip"] >> hog_clip;
        if (!fn["use_hog"].empty())
            fn["use_hog"] >> use_hog;
        if (!fn["use_cnn_torch"].empty())
            fn["use_cnn_torch"] >> use_hog;
        if (!fn["use_color_names"].empty())
            fn["use_color_names"] >> use_color_names;
        if (!fn["use_gray"].empty())
            fn["use_gray"] >> use_gray;
        if (!fn["use_rgb"].empty())
            fn["use_rgb"] >> use_rgb;
        if (!fn["window_function"].empty())
            fn["window_function"] >> window_function;
        if (!fn["kaiser_alpha"].empty())
            fn["kaiser_alpha"] >> kaiser_alpha;
        if (!fn["cheb_attenuation"].empty())
            fn["cheb_attenuation"] >> cheb_attenuation;
        if (!fn["filter_lr"].empty())
            fn["filter_lr"] >> filter_lr;
        if (!fn["admm_iterations"].empty())
            fn["admm_iterations"] >> admm_iterations;
        if (!fn["number_of_scales"].empty())
            fn["number_of_scales"] >> number_of_scales;
        if (!fn["scale_sigma_factor"].empty())
            fn["scale_sigma_factor"] >> scale_sigma_factor;
        if (!fn["scale_model_max_area"].empty())
            fn["scale_model_max_area"] >> scale_model_max_area;
        if (!fn["scale_lr"].empty())
            fn["scale_lr"] >> scale_lr;
        if (!fn["scale_step"].empty())
            fn["scale_step"] >> scale_step;
        if (!fn["use_channel_weights"].empty())
            fn["use_channel_weights"] >> use_channel_weights;
        if (!fn["weights_lr"].empty())
            fn["weights_lr"] >> weights_lr;
        if (!fn["use_segmentation"].empty())
            fn["use_segmentation"] >> use_segmentation;
        if (!fn["histogram_bins"].empty())
            fn["histogram_bins"] >> histogram_bins;
        if (!fn["background_ratio"].empty())
            fn["background_ratio"] >> background_ratio;
        if (!fn["histogram_lr"].empty())
            fn["histogram_lr"] >> histogram_lr;
        if (!fn["psr_threshold"].empty())
            fn["psr_threshold"] >> psr_threshold;
        CV_Assert(number_of_scales % 2 == 1);
        CV_Assert(use_gray || use_color_names || use_hog || use_rgb);
    }

    void TrackerCSRTImpl::Params::write(cv::FileStorage &fs) const
    {
        fs << "padding" << padding;
        fs << "template_size" << template_size;
        fs << "gsl_sigma" << gsl_sigma;
        fs << "hog_orientations" << hog_orientations;
        fs << "num_hog_channels_used" << num_hog_channels_used;
        fs << "hog_clip" << hog_clip;
        fs << "use_hog" << use_hog;
        fs << "use_cnn_torch" << use_cnn_torch;
        fs << "use_color_names" << use_color_names;
        fs << "use_gray" << use_gray;
        fs << "use_rgb" << use_rgb;
        fs << "window_function" << window_function;
        fs << "kaiser_alpha" << kaiser_alpha;
        fs << "cheb_attenuation" << cheb_attenuation;
        fs << "filter_lr" << filter_lr;
        fs << "admm_iterations" << admm_iterations;
        fs << "number_of_scales" << number_of_scales;
        fs << "scale_sigma_factor" << scale_sigma_factor;
        fs << "scale_model_max_area" << scale_model_max_area;
        fs << "scale_lr" << scale_lr;
        fs << "scale_step" << scale_step;
        fs << "use_channel_weights" << use_channel_weights;
        fs << "weights_lr" << weights_lr;
        fs << "use_segmentation" << use_segmentation;
        fs << "histogram_bins" << histogram_bins;
        fs << "background_ratio" << background_ratio;
        fs << "histogram_lr" << histogram_lr;
        fs << "psr_threshold" << psr_threshold;
    }

} /* namespace myCSRT */