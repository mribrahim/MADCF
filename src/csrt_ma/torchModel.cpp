#include <vector>
#include "torchModel.hpp"

#include <torch/script.h>

MyTorchModel::MyTorchModel()
{
    // modelName = "../model-sand.pt";
    modelName = "../resnet.pt";
    modelTorch = torch::jit::load(modelName);
    modelTorch.eval();

    if (torch::cuda::is_available()) {
        deviceTorch = torch::kCUDA;
        modelTorch.to(deviceTorch);
    }
}

std::vector<cv::Mat> MyTorchModel::getFeatures(const cv::Mat &frame_roi, const cv::Size2i &feature_size)
{   
    std::vector<torch::jit::IValue> inputs;
    cv::Mat inputMat;

    if (std::string::npos != modelName.find("sand.pt")){
        cv::resize(frame_roi, inputMat, cv::Size(200,300), cv::INTER_CUBIC);
    }
    else{
         cv::resize(frame_roi, inputMat, cv::Size(200,200), cv::INTER_CUBIC);
    }
    // cv::imshow("CNN frame_roi", frame_roi);

    torch::Tensor in = imgToTensor(inputMat);
    inputs.push_back(in.to(deviceTorch));
    torch::Tensor output1 = modelTorch.forward(inputs).toTensor().cpu();

    int index = 0;
    std::vector<cv::Mat> features;
    int featureChannel = output1.size(1);
    int featureHeight = output1.size(2);
    int featureWidth = output1.size(3);

//    if (featureChannel > 25) // model.pt
//    {
//        featureWidth = 52;
//        featureHeight = 52;
//        featureChannel = 16;
//    }

    for (int i=0; i< featureChannel; i++){
//        cv::Mat image = cv::Mat(cv::Size(200,300), CV_32FC1, output1.data_ptr<float>(), cv::Mat::AUTO_STEP);
        cv::Mat image(featureHeight, featureWidth, CV_32FC1);
        std::memcpy((void *)image.data, output1.data_ptr() + index, featureHeight * featureWidth * sizeof(float));
//        cv::imshow("CNN feature", image);
//        cv::waitKey(0);
        cv::Mat matFeature;
//        cv::normalize(image, matFeature, 1);
//        image.convertTo(matFeature, CV_32FC1, 1.0/255.0, -0.5);
        cv::resize(image, matFeature, feature_size);
        features.push_back(matFeature);
        index += (featureHeight*featureWidth * sizeof(float));
    }
//     double min_val, max_val;
//     cv::minMaxLoc(features.at(0), &min_val, &max_val, NULL, NULL);
    //  std::cout << "CNN:" << min_val << "," << max_val << std::endl;
//    cv::imshow("CNN feature", features.at(1));
//    cv::waitKey(0);
    return features;
}

torch::Tensor MyTorchModel::imgToTensor(const cv::Mat &imgInput)
{
    cv::Mat img;
//    cv::resize(img, img, cv::Size(64,64));

    if (imgInput.channels()==1)
        cv::cvtColor(imgInput, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(imgInput, img, cv::COLOR_BGR2RGB);

    img.convertTo( img, CV_32FC3, 1/255.0 );

    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, c10::kFloat);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor.unsqueeze_(0);

    img_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(img_tensor);

    return img_tensor;
}