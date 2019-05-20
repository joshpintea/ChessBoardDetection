#pragma once
#include <opencv2/core/mat.hpp>
using namespace cv;

enum PERSPECTIVE_TRANSFORM_IMPL
{
	OPEN_CV,
	PIZZA
};

class PerspectiveProjection
{
public:
	Mat getPerspectiveTransform(cv::Point2f sourcePoints[4], cv::Point2f destinationPoints[4], PERSPECTIVE_TRANSFORM_IMPL impl) const;
	Mat perspectiveProjection(Mat tr, Mat source, PERSPECTIVE_TRANSFORM_IMPL impl, Size size);
	void getBorderBoxes(std::vector<Point> points, Point& tl, Point& tr, Point& bl, Point& br);

private:
	bool compareSum(Point p1, Point p2);
	bool compareDiff(Point p1, Point p2);
};
