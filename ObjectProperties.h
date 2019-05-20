#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>

class ObjectProperties
{
public:
	int id;
	int rows;
	int cols;
	int colMin;
	int colMax;
	int rowMin;
	int rowMax;
	int surface;
	cv::Point center;
	float perimeter = 0;
	int *h;
	int *v;
	float aspectRatio;
	float thinnessRatio;

	float fi;

	ObjectProperties();
	ObjectProperties(int rows, int cols, int id);
	std::vector<cv::Point> pixels;
	std::vector<cv::Point> pixelsOnEdge;

	void add(int row, int col, bool isOnEdge = false);
	void computeFinalProperties();
	void print();
	cv::Mat drawContourPixels(cv::Mat img);
	bool operator < (const ObjectProperties &obj);
	bool operator == (const ObjectProperties &obj);
private:
	int rowSum = 0;
	int colSum = 0;
	void setDetails(int row, int col);
};
