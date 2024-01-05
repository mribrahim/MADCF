//
// Created by ibrahim on 2/14/21.
//

#ifndef PERCEPTION_MYTRACKER_H
#define PERCEPTION_MYTRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "csrt_ma/tracker.hpp"


enum trackerType{
    trackerOPENCV,
    trackerNEWCSRT
};

enum openCVTrackerMethod{
    CSRT,
    KCF,
    TLD
};

class MyTracker {

public:
    MyTracker();
    void setTrackerMethod(std::string method);
    void add(cv::Mat &frame, cv::Rect &bbox);
    void add( cv::Mat &frame, std::vector<cv::Rect> &bboxes);
    void update( cv::Mat &frame, const cv::Mat &homoMat);
    std::vector<cv::Rect2d> getBboxes();
    std::string getMethodName();
private:

    cv::Ptr<cv::Tracker> createTracker(cv::Mat &frame, cv::Rect &bbox);

    trackerType selectedType;
    openCVTrackerMethod selectedOpencvTracker;
    int MAX_TRACKER_COUNT = 5;
    std::vector<cv::Ptr<cv::Tracker>> trackersOPENCV;
    std::vector<myCSRT::TrackerCSRTImpl> trackersNewCSRT;
    myCSRT::TrackerCSRTImpl::Params parametersOURS;

    std::vector<cv::Rect2d> boxesOPENCV;
};


#endif //PERCEPTION_MYTRACKER_H
