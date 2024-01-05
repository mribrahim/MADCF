//
// Created by ibrahim on 21.10.2021.
//

#ifndef TEPEGOZ_READDATA_H
#define TEPEGOZ_READDATA_H

#include "opencv4/opencv2/opencv.hpp"
#include <iostream>

enum DataType { FILENAME, FOLDER, VIDEO};

class ReadData {
public:
   ReadData(std::string _path, std::string extension = std::string(), int index = 0);
   ~ReadData();
   bool read(cv::Mat &frame);
private:
    DataType srcMode;
    cv::VideoCapture *capture;
    std::string path;
    int i;
    bool ended;
    std::vector<std::string> fileList;
};


#endif //TEPEGOZ_READDATA_H
