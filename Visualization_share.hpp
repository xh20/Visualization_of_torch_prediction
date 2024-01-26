//
// Created by lz on 04.06.23.
//

#ifndef HAR_VISUALIZATION_HPP
#define HAR_VISUALIZATION_HPP

#endif //HAR_VISUALIZATION_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
//#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <cmath>

using namespace std;

class Visualizer{
private:
    cv::Mat lblBarLeft = cv::Mat(cv::Size(848, 40), CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat probBarLeft = cv::Mat(cv::Size(848, 40), CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat lblBarRight = cv::Mat(cv::Size(848, 40), CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat probBarRight = cv::Mat(cv::Size(848, 40), CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat spaceBar = cv::Mat(cv::Size(848, 10), CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat arrowMat = cv::Mat(cv::Size(848, 10), CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat legend;
    int num_classes = 9;
    map<int, cv::Vec3b> colorMap;
    torch::Tensor l_probability, l_predictions;
    torch::Tensor r_probability, r_predictions;
    bool isBarCreated = false;
    int nFrames;
    int height = 480;
    int width = 848;
    int barStartCol = 100;
    int index_old = 100;

    int windowSize;
    vector<double> thresholds;
    std::shared_ptr<std::vector<std::shared_ptr<cv::Mat>>> dispImages_ptr;

    void createBarRT(const int&);

public:
    explicit Visualizer(const string&);
    ~Visualizer();
    void run(const std::shared_ptr<std::vector<std::shared_ptr<cv::Mat>>>& colorImages_ptr);

    void plotRealTimeResults(const cv::Mat*,
                             const torch::Tensor&, const torch::Tensor&,
                             const int&);
    string saveDir;

//    vector<int> lHands;
//    vector<int> rHands;
};

Visualizer::Visualizer() {
    nFrames = 6*windowSize;
    cv::namedWindow("Display: Results", CV_WINDOW_NORMAL);
    cv::setWindowProperty("Display: Results", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    // B G R
    /***
     * {-1, "unknown"}, {0, "idle"}, {1, "approach"}, {2, "retreat"},
        {3, "take/carry"}, {4, "place"}, {5, "hold"}, {6, "pour"},
        {7, "screw"}, {8, "fold/unfold"}
     */
    colorMap[-1] = cv::Vec3b(60, 60, 60);
    colorMap[0] = cv::Vec3b(0, 192, 192);
    colorMap[1] = cv::Vec3b(189, 114, 0);
    colorMap[2] = cv::Vec3b(25, 83, 217);
//    colorMap[2] = cv::Vec3b(32, 177, 237);
    colorMap[3] = cv::Vec3b(142, 47, 126);
    colorMap[4] = cv::Vec3b(48, 172, 119);
    colorMap[5] = cv::Vec3b(238, 172, 77);
    colorMap[6] = cv::Vec3b(47, 20, 162);
    colorMap[7] = cv::Vec3b(0, 0, 255);
    colorMap[8] = cv::Vec3b(0, 128, 0);
//    colorMap[9] = cv::Vec3b(255, 0, 0);
    int fontFace = cv::FONT_HERSHEY_COMPLEX;
    cv::putText(lblBarLeft, "left hand", cv::Point(5, 25), fontFace,
            0.5, cv::Scalar(100, 100, 0), 1, cv::LINE_AA);
    cv::putText(lblBarRight, "right hand", cv::Point(5, 25), fontFace,
            0.5, cv::Scalar(100, 100, 0), 1, cv::LINE_AA);

    cv::putText(probBarLeft, "left prob", cv::Point(5, 25), fontFace,
                        0.5, cv::Scalar(100, 100, 0), 1, cv::LINE_AA);
    cv::putText(probBarRight, "right prob", cv::Point(5, 25), fontFace,
                0.5, cv::Scalar(100, 100, 0), 1, cv::LINE_AA);
}

Visualizer::~Visualizer() = default;

void Visualizer::createBarRT(const int& index){
    auto probL = l_probability[0].item<double>();
    auto probR = r_probability[0].item<double>();

    auto maxL = max(probL, 255.0);
    auto minL = min(probL, 0.0);
    auto maxR = max(probR, 255.0);
    auto minR = min(probR, 0.0);

    probL = 255.0*(maxL - probL)/(maxL- minL);
    probR = 255.0*(maxR - probR)/(maxR- minR);

    lblBarLeft.colRange(index_old, index) = colorMap[l_predictions[0].item<int>()];
    lblBarRight.colRange(index_old, index) = colorMap[r_predictions[0].item<int>()];
    probBarLeft.colRange(index_old, index) = cv::Vec3b(probL, probL, probL);
    probBarRight.colRange(index_old, index) = cv::Vec3b(probR, probR, probR);
}

void Visualizer::plotRealTimeResults(const cv::Mat* dispImages_ptr,
                             const torch::Tensor& probLeft, const torch::Tensor& probRight,
                             const int& frameIndex){

    torch::Tensor probL, probR, predL, predR;
    tie(l_probability, predL) = max(probLeft, 0);
    tie(r_probability, predR) = max(probRight, 0);
    tie(probL, l_predictions) = max(torch::softmax(probLeft, 0), 0);
    tie(probR, r_predictions) =  max(torch::softmax(probRight, 0), 0);

    height = dispImages_ptr->rows;
    width = dispImages_ptr->cols;
    if(frameIndex>=nFrames){
        nFrames = frameIndex;
    }
    int index = (int) round(frameIndex*(width-barStartCol)/nFrames) + barStartCol;
    createBarRT(index);
    cv::Mat concatenatedMat;
    cv::vconcat(dispImages_ptr->clone(), arrowMat.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, spaceBar.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, lblBarLeft.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, spaceBar.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, probBarLeft.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, spaceBar.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, spaceBar.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, lblBarRight.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, spaceBar.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, probBarRight.clone(), concatenatedMat);
    int heightConcat = concatenatedMat.rows;

/* If you need it , create a legend as the same as the creatBarRT function
    cv::vconcat(concatenatedMat, spaceBar.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, spaceBar.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, legend.clone(), concatenatedMat);
    cv::vconcat(concatenatedMat, spaceBar.clone(), concatenatedMat);
*/
    int arrowStart = index;
    cv::Point startPoint(arrowStart, height + 10);
    cv::Point endPoint(arrowStart, heightConcat);
    cv::line(concatenatedMat, startPoint, endPoint, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Display: Results", concatenatedMat);
    index_old = index;

    int key = cv::waitKey(1);
    if(key==27)
    {
        cvDestroyWindow("Display: Results");
        exit(-1);
    }
}

