#pragma once
#include <opencv2/core/mat.hpp>

using namespace cv;


float filter(Mat source, Mat mask, int row, int col);
bool isInside(Mat img, int row, int col);
Mat convolution(Mat source, Mat mask);
void harrisCorners(Mat source);
