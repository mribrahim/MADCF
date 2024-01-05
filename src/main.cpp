#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>

#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>

#include "utils.h"
#include "mytracker.h"
#include "readData.h"

using namespace std;
using namespace cv;

cv::Mat findHomographyMatrix(const cv::Mat &prevGray, const cv::Mat &currentGray);
cv::Mat makeHomoGraphy(int *pnMatch, int nCnt);

int MAX_COUNT;
std::vector<uchar> status;
std::vector<cv::Point2f>  pointsPrev, pointsCurrent;

int main(int argc, char* argv[]) {
    
    string path;
    string sequence = "Pexels-Shuraev-trekking";
    string method = "MADCF";
    for (int i = 0; i < argc; ++i)
    {
        if (0 == strcmp("-s", argv[i]))
        {
            sequence = argv[i+1];
        }
        else if (0 == strcmp("-m", argv[i]))
        {
            method = argv[i+1];
        }
        else if (0 == strcmp("-p", argv[i]))
        {
            path = argv[i+1];
        }
        else if (0 == strcmp("-h", argv[i]))
        {
            cout << "Use following parameters to select dataset path, sequence and method" 
            "\n -p for PESMOD dataset path \n sequence with -s (-s Pexels-Welton)  \n method with -m (-m MADCF/CSRT/KCF/TLD/)" << endl;
            exit(0);
        }
    }


    path += sequence + "/images/";
    string pathAnno = "../PESMOD/" + sequence + "/annotations/";

    vector<Rect> gtBoxes = readGtboxes(pathAnno);
    if (sequence.find("Zaborski") != string::npos)
        gtBoxes.push_back(Rect(980, 580, 40, 60)); // Zaborski
    else if (sequence.find("Elliot") != string::npos)
        gtBoxes.push_back(Rect(810, 400, 15, 15)); // Elliot-road
    else if (sequence.find("Marian") != string::npos)
        gtBoxes.push_back(Rect(1800, 490, 20, 15)); // Marian
    else if (sequence.find("Welton") != string::npos)
        gtBoxes.push_back(Rect(1205, 579, 30, 20)); // Welton
    else if (sequence.find("Wolfgang") != string::npos)
        gtBoxes.push_back(Rect(720, 395, 25, 20)); // Wolfgang
    else if (sequence.find("trekking") != string::npos)
        gtBoxes.push_back(Rect(1340, 585, 25, 30)); // Shuraev-trekking
    else if (sequence.find("Zaborski") != string::npos)
        gtBoxes.push_back(Rect(1460, 920, 25, 40)); // Grisha-snow


    int width = 1920, height = 1080;
    int gridSizeW = 32;
    int gridSizeH = 24;
    MAX_COUNT =  (width / gridSizeW + 1) * (height / gridSizeH + 1);

    for (int i = 0; i < width / gridSizeW - 1; ++i) {
        for (int j = 0; j < height / gridSizeH - 1; ++j) {
            pointsPrev.push_back(Vec2f(i * gridSizeW + gridSizeW / 2.0, j * gridSizeH + gridSizeH / 2.0));
        }
    }

    ReadData readData(path, "jpg");
    MyTracker myTracker;
    myTracker.setTrackerMethod(method);


    bool processOneStep = false;
    bool isStopped = false;

    Mat frame, frameGray, frameGrayPrev, homoMat;
    int counter = 0;
    char keyboard;
    while(true) {

        int64 startTime = cv::getTickCount();

        if (processOneStep){
            keyboard = waitKey(0);
            if (83 != keyboard){
                processOneStep = false;
            }
        }else{
            keyboard = waitKey(5);
        }
        if (keyboard == 27)
        {
            break;
        }
        if ('s' == keyboard)
        {
            isStopped = !isStopped;
            processOneStep = false;
        }
        else if ( 83 == keyboard )
        {
            processOneStep = true;
        }

        if (isStopped){
            continue;
        }

        Mat frame, frameShow;
        bool ret = readData.read(frame);
        if (!ret){
            break;
        }
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        frame.copyTo(frameShow);
        
        if (!frameGrayPrev.empty()) {
            homoMat = findHomographyMatrix(frameGrayPrev, frameGray);
        }
        else{
            homoMat = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        }

        if (32 == keyboard){
            Rect r = selectROI(frame);
            destroyWindow("ROI selector");
            myTracker.add(frame, r);
        }

        myTracker.update(frame, homoMat);
        for (Rect2d bbox:myTracker.getBboxes()){
           rectangle(frameShow, bbox, Scalar (0,0,255), 2, 1);
           putText(frameShow, myTracker.getMethodName(),Point2d(bbox.x+ 5, bbox.y+20), FONT_HERSHEY_PLAIN, 1.2, Scalar (0,0,255), 2);
        }


        if (gtBoxes.size()>0 && counter<gtBoxes.size()) {
            Rect rectGT = gtBoxes.at(counter);
            putText(frameShow, "GT", Point2d(rectGT.x, rectGT.y - 20), FONT_HERSHEY_PLAIN, 1.2, Scalar(0, 255, 0), 2);
            rectangle(frameShow, rectGT, Scalar(0, 255, 0), 2, 1);

            // Start tracker
            if (0 == counter) {
                myTracker.add(frame, rectGT);
            }
        }


        resize(frameShow, frameShow, Size(1280,720));
        putText(frameShow, to_string(counter), Point2d(30, 30), FONT_HERSHEY_PLAIN, 1.2, Scalar(0, 255, 0), 2);
        imshow("image", frameShow);
        frameGray.copyTo(frameGrayPrev);

        counter ++;
        if ( 0 == counter % 50) {
            double secs = (cv::getTickCount() - startTime) / cv::getTickFrequency();
            cout << "elapsed time: " << secs << endl;
        }

    }       

    return 0;
}



Mat findHomographyMatrix(const Mat &prevGray, const Mat &currentGray)
{
    int* nMatch = (int*)alloca(sizeof(int) * MAX_COUNT);
    int count;
    int flags = 0;
    int i =0, k=0;
    if (!pointsPrev.empty())
    {
        TermCriteria criteria = TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 20, 0.03);
        calcOpticalFlowPyrLK(prevGray, currentGray, pointsPrev, pointsCurrent, status, noArray(), Size(15, 15), 2, criteria, flags);

        for (i = k = 0; i < status.size(); i++) {
            if (!status[i]) {
                continue;
            }

            nMatch[k++] = i;
        }
        count = k;
    }
    if (k >= 10) {
        // Make homography matrix with correspondences
        return makeHomoGraphy(nMatch, count);
        //homoMat = findHomography(points0, points1, RANSAC, 1);
    } else {
        return (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
}

Mat makeHomoGraphy(int *pnMatch, int nCnt)
{
    vector<Point2f> pt1;
    vector<Point2f> pt2;
    for (int i = 0; i < nCnt; ++i)
    {
        pt1.push_back(pointsPrev[pnMatch[i]]);
        pt2.push_back(pointsCurrent[pnMatch[i]]);

    }
    return findHomography(pt1, pt2, RANSAC, 1);
}