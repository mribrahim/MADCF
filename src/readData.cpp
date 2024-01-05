//
// Created by ibrahim on 21.10.2021.
//

#include "readData.h"
#include "utils.h"
#include <experimental/filesystem>

using namespace std;
using namespace cv;


ReadData::~ReadData() {
    if (capture) {
        delete capture;
    }
}


ReadData::ReadData(string _path, string extension, int index) {

    i = index;
    path = _path;
    ended = false;
    if (extension.size()==0)
    {
        if (string::npos != path.find(".mp4") || string::npos != path.find(".mov") || string::npos != path.find(".avi"))
        {
            srcMode = VIDEO;
            capture = new VideoCapture(path);
            if (capture->isOpened()){
                cout << "opening video: " << path << endl;
                capture->set(CAP_PROP_POS_FRAMES, index);
            }
            else{
                if (experimental::filesystem::exists(path))
                {
                    cout<<"!!! File exists in the given path, but could not be opened: " << path << endl;
                }else {
                    cout << "!!! please check video path: " << path << endl;
                }
                ended = true;
            }
        }
        else if (string::npos != path.find(".jpg") || string::npos != path.find(".png") || string::npos != path.find(".jpeg"))
        {
            srcMode = FILENAME;
        }
    }
    else{
        srcMode = FOLDER;
        read_directory(path, fileList, extension);
        sort(fileList.begin(), fileList.end());
    }
}

bool ReadData::read(Mat &frame) {

    if (ended){
        return false;
    }

    if (FOLDER == srcMode){
        if (i>=fileList.size()){
            return false;
        }
        frame = imread(path + fileList.at(i));
        i++;
    }
    else if (FILENAME == srcMode) {
        frame = imread(path);
        ended = true;
    }
    else if (VIDEO == srcMode) {
        return capture->read(frame);
    }

    if (frame.empty())
    {
        return false;
    }
    return true;
}