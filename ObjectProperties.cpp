#include "stdafx.h"
#include "ObjectProperties.h"
#include "common.h"
#include <math.h>

ObjectProperties::ObjectProperties()
{
	
}
ObjectProperties::ObjectProperties(int rows, int cols, int id)
{
	this->colMin = -1;
	this->colMax = -1;
	this->rowMin = -1;
	this->rowMax = -1;
	this->id = id;

	this->surface = 0;
	this->perimeter = 0.0f;

	this->h = static_cast<int*>(calloc(cols, sizeof(int)));
	this->v = static_cast<int*>(calloc(rows, sizeof(int)));
}

void ObjectProperties::add(int row, int col, bool isOnEdge)
{
	cv::Point point(row, col);
	this->pixels.push_back(point);
	this->setDetails(row, col);


	this->surface++;
	this->rowSum += row;
	this->colSum += col;

	if (isOnEdge)
	{
		this->perimeter++;
		this->pixelsOnEdge.push_back(point);
	}

	this->h[col]++;
	this->v[row]++;
}


void ObjectProperties::setDetails(int row, int col)
{
	if (col < this->colMin || this->colMin == -1)
	{
		this->colMin = col;
	}

	if (col > this->colMax || this->colMax == -1)
	{
		this->colMax = col;
	}

	if (row < this->rowMin || this->rowMin == -1)
	{
		this->rowMin = row;
	}

	if (row > this->rowMax || this->rowMax == -1)
	{
		this->rowMax = row;
	}
}

void ObjectProperties::computeFinalProperties()
{
	this->perimeter *= (PI / 4.0);
	if (this->surface > 0)
	{
		this->center.x = static_cast<int>(this->rowSum / this->surface);
		this->center.y = static_cast<int>(this->colSum / this->surface);
	}

	this->aspectRatio = static_cast<float>(this->colMax - this->colMin + 1) / static_cast<float>(this->rowMax - this->rowMin + 1);
	this->thinnessRatio = (4 * PI) * (this->surface / static_cast<float>(this->perimeter * this->perimeter));

	int num = 0;
	int nm1 = 0;
	int nm2 = 0;

	for(cv::Point point: this->pixels)
	{
		num += ((point.x - this->center.x) * (point.y - this->center.y));
		nm1 += ((point.y - this->center.y) * (point.y - this->center.y));
		nm2 += ((point.x - this->center.x) * (point.x - this->center.x));
	}

	float tan = atan2(2.0 * num, nm1- nm2);

	this->fi = tan / 2;
}


cv::Mat ObjectProperties::drawContourPixels(cv::Mat img) {
	cv::Mat image(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

	// for (cv::Point point: this->pixelsOnEdge)
	// {
	// 	image.at<Vec3b>(point.x, point.y) = img.at<Vec3b>(point.x, point.y);
	// }

	return image;
}

bool ObjectProperties::operator<(const ObjectProperties& obj)
{
	return this->id < obj.id;
}

bool ObjectProperties::operator==(const ObjectProperties& obj)
{
	return this->id = obj.id;
}


void ObjectProperties::print()
{
	//std::cout << std::endl << this->colMax - this-> colMin << " " << this->colMin << " " << this->rowMax - this->rowMin << " " << this->rowMin << std::endl;

	std::cout << std::endl << "Id " << this->id << std::endl;
	std::cout << "Surface " << this->surface << std::endl;
	std::cout << "Perimeter " << this->perimeter << std::endl;
	std::cout << "Aspect ratio " << this->aspectRatio<< std::endl;
	std::cout << "Thinness ratio " << this->thinnessRatio << std::endl;
	std::cout << "Center " << this->center << std::endl;
	std::cout << "Fi " << this->fi << std::endl;
}

