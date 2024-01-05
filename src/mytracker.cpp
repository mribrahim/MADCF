//
// Created by ibrahim on 2/14/21.
//

#include "mytracker.h"
#include "utils.h"
#include <opencv2/tracking/tracking_legacy.hpp>

using namespace std;

MyTracker::MyTracker() {

     parametersOURS.use_cnn_torch = true;
     parametersOURS.use_hog = false;    
     parametersOURS.use_color_names = false;
     parametersOURS.use_gray = false;

    //  parametersOURS.use_cnn_torch = false;
    //  parametersOURS.use_hog = true;    
    //  parametersOURS.use_color_names = true;
    //  parametersOURS.use_gray = true;

}

void MyTracker::setTrackerMethod(string method) {

    if ( "CSRT" == toUppercase(method)){
        selectedOpencvTracker = CSRT;
        selectedType = trackerOPENCV;
    }
    else if ( "KCF" == toUppercase(method)){
        selectedOpencvTracker = KCF ;
        selectedType = trackerOPENCV;
    }
    else if ( "TLD" == toUppercase(method)){
        selectedOpencvTracker = TLD ;
        selectedType = trackerOPENCV;
    }
    else if ( "MADCF" == toUppercase(method)){
        selectedType = trackerNEWCSRT;
    }
    else{
        cout << "Please select a proper tracker method, options: CSRT, KCF, TLD, MADCF" << endl;
        exit(0);
    }
}

cv::Ptr<cv::Tracker> MyTracker::createTracker(cv::Mat &frame, cv::Rect &bbox){
    cv::Ptr<cv::Tracker> tracker;

    if ( CSRT == selectedOpencvTracker){
        // tracker = cv::TrackerCSRT::create();
        tracker = cv::legacy::upgradeTrackingAPI(cv::legacy::TrackerCSRT::create());
    }
    else if ( KCF == selectedOpencvTracker){
        // tracker = cv::TrackerKCF::create();
        tracker = cv::legacy::upgradeTrackingAPI(cv::legacy::TrackerKCF::create());
    }
    else if ( TLD == selectedOpencvTracker){
        // tracker = cv::TrackerTLD::create();
        tracker = cv::legacy::upgradeTrackingAPI(cv::legacy::TrackerTLD::create());
    }

    tracker->init(frame, bbox);
    return tracker;
}

void MyTracker::add(cv::Mat &frame, cv::Rect &bbox) {
    if ( trackerOPENCV == selectedType && trackersOPENCV.size() >= MAX_TRACKER_COUNT) {
        return;
    }

    if ( trackerOPENCV == selectedType) {
        for (const cv::Rect &temp: boxesOPENCV) {
            cv::Rect r = temp & bbox;
            if (false == r.empty()) {
                return;
            }
        }
        trackersOPENCV.push_back(createTracker(frame, bbox));
    }
    else if ( trackerNEWCSRT == selectedType) {
        for (const cv::Rect &temp: boxesOPENCV) {
            cv::Rect r = temp & bbox;
            if (false == r.empty()) {
                return;
            }
        }
        myCSRT::TrackerCSRTImpl x;
        x.setParams(parametersOURS);
        x.init(frame, bbox);
        trackersNewCSRT.push_back(x);
    }
}

void MyTracker::add(cv::Mat &frame, std::vector<cv::Rect> &bboxes) {

    for(cv::Rect &bbox: bboxes) {
        add(frame, bbox);
    }
}

void MyTracker::update(cv::Mat &frame, const cv::Mat &homoMat) {

    if ( trackerOPENCV == selectedType) {
        boxesOPENCV.clear();
        for (auto it = trackersOPENCV.begin(); it != trackersOPENCV.end();){
            cv::Rect box;
            bool ret = (*it)->update(frame, box);
            if (!ret){
                cout<<  "delete CSRT tracker"<< endl;
                it = trackersOPENCV.erase(it);
            }
            else{
                ++it;
                boxesOPENCV.push_back(box);
            }
        }
    }
    if ( trackerNEWCSRT == selectedType) {
        boxesOPENCV.clear();
        for (auto it = trackersNewCSRT.begin(); it != trackersNewCSRT.end();){
            cv::Rect2d box;
            bool ret = (*it).update(frame, box, homoMat);
            // if (!ret){
            //     cout<<  "delete NEW CSRT tracker"<< endl;
            //     it = trackersNewCSRT.erase(it);
            // }
            // else{
            //     ++it;
            //     boxesOPENCV.push_back(box);
            // }
            if (ret){
                boxesOPENCV.push_back(box);
            }
            ++it;
        }
    }
}

vector<cv::Rect2d> MyTracker::getBboxes() {

    if ( trackerOPENCV == selectedType || trackerNEWCSRT == selectedType ) {
        return boxesOPENCV;
    }
    vector<cv::Rect2d> temp;
    return temp;
}

string MyTracker::getMethodName()
{
    if ( trackerOPENCV == selectedType ){
        if (CSRT == selectedOpencvTracker){
            return "CSRT";
        }
        else if (KCF == selectedOpencvTracker){
            return "KCF";
        }
        else if (TLD == selectedOpencvTracker){
            return "TLD";
        }
    }
    else if (trackerNEWCSRT == selectedType){
            return "MADCF";
    }

    return "None";
}
