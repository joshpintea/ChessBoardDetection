#pragma once
#include <opencv2/core/mat.hpp>
using namespace cv;

enum HOUGH_TRANSFORMATION_IMPL
{
	OPEN_CV_H,
	PIZZA_H
};

class HoughTransformation
{
public:
	void transform(Mat edges, int thresh, int t, std::vector<Point>& points, HOUGH_TRANSFORMATION_IMPL impl);
};
