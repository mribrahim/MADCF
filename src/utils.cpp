#include <fstream>
#include <iterator>
#include <numeric>

#include "utils.h"

using namespace std;
using namespace cv;

void read_directory(const std::string& path, std::vector<std::string>& v, string extension)
{
    DIR* dirp = opendir(path.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {

        string filePath = path + dp->d_name;
        if (std::string::npos == filePath.find(extension))
            continue;

        v.push_back(dp->d_name);
    }
    closedir(dirp);
}

template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

string toUppercase(string &data)
{
    std::for_each(data.begin(), data.end(), [](char & c) {
            c = ::toupper(c);
        });
    return data;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

vector<Rect> readGtboxes(string path) {

    vector<Rect> bboxes;
    ifstream file(path);
    string strLine;
    while (getline(file, strLine)) {        
        vector<string> result = split(strLine, ',');
        if ("NaN" == result.at(0)){
            bboxes.push_back(Rect2d());
        }
        else{
            bboxes.push_back(Rect(stoi(result.at(0)), stoi(result.at(1)), stoi(result.at(2)), stoi(result.at(3))));
        }
    }

    return bboxes;
}

float average(vector<float> const& v){
    if(v.empty()){
        return 0;
    }

    float sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

float stdDev(vector<float> const& v, float mean){
    if(v.empty()){
        return 0;
    }

    float sum = 0;
    for (float e : v)
    {
       sum += pow(e-mean, 2);
    }
    
    return sqrt(sum / v.size());
}

bool enlargeRect(cv::Rect &rect, int a)
{
    rect.x -=a;
    rect.y -=a;
    rect.width += (a*2);
    rect.height += (a*2);
    return true;
}

void findCombinedRegions(const Mat &mask, Mat &maskRegionOutput, Mat &maskSmallregions, vector<Rect> &rectangles, int minArea, bool applyRegionGrowing, const Mat &frame)
{
    Mat frameTemp;
    if (!frame.empty()) {
        frame.copyTo(frameTemp);
    }

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );
    maskSmallregions = Mat::zeros( mask.size(), CV_8UC1 );

    for( size_t i = 0; i< contours.size(); i++ )
    {
        if (contourArea(contours[i]) <= minArea)
        {
            continue;
        }

        Rect rect = boundingRect(contours[i]);

        if (applyRegionGrowing && !enlargeRect(rect))
        {
            continue;
        }
        rectangle(maskSmallregions, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), 1, FILLED);
        if (!frame.empty()) {
            rectangle(frameTemp, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(255,0,0));
        }
    }

    maskRegionOutput = Mat::zeros( mask.size(), CV_8UC1 );
    findContours( maskSmallregions, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Rect rect = boundingRect(contours[i]);


        rectangles.push_back(rect);

        rectangle(maskRegionOutput, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), 1, FILLED);
        if (!frame.empty()) {
            rectangle(frameTemp, Point(rect.x, rect.y), Point(rect.x+rect.width, rect.y+rect.height), Scalar(0, 255,0));
        }
    }

}


void flow2img(const Mat &flow,Mat &magnitude, Mat &img)
{
    Mat angle, magn_norm;
    Mat flow_parts[2];
    split(flow, flow_parts);
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));
    //build hsv image
    Mat _hsv[3], hsv, hsv8;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, img, COLOR_HSV2BGR);
}
