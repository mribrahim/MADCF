
#ifndef DEF_MyTorchModel
#define DEF_MyTorchModel

#include <opencv4/opencv2/opencv.hpp>

#include <torch/torch.h>

class MyTorchModel
{
    public:
        MyTorchModel();
        std::vector<cv::Mat> getFeatures(const cv::Mat &frame_roi, const cv::Size2i &feature_size);
    private:
        std::string modelName;
        torch::jit::Module modelTorch;
        torch::Device deviceTorch = torch::kCPU;

        torch::Tensor imgToTensor(const cv::Mat &img);

        std::vector<double> norm_mean = {0.485, 0.456, 0.406};
        std::vector<double> norm_std = {0.229, 0.224, 0.225};
};

#endif //DEF_MyTorchModel