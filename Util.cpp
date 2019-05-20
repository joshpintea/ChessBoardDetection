#include "stdafx.h"
#include "Util.h"
#include <opencv2/shape/hist_cost.hpp>
#include <opencv2/highgui.hpp>


bool isInside(Mat img, int row, int col)
{
	return (row >= 0 && row < img.rows) && (col >= 0 && col < img.cols);
}


float filter(Mat source, Mat mask, int row, int col)
{
	int hs_r = (mask.rows / 2);
	int hs_c = (mask.cols / 2);


	float out = 0;

	for (int u = 0; u < hs_r; ++u)
	{
		for (int v = 0; v < hs_c; ++v)
		{
			int r = row + u - hs_r;
			int c = col + v - hs_c;
			if (isInside(source, r, c))
			{
				out += (mask.at<float>(u, v) * source.at<float>(r, c));
			}
		}
	}
	return out;
}


Mat convolution(Mat source, Mat mask)
{
	Mat destination(source.rows, source.cols, CV_32FC1);

	int hs_r = mask.rows / 2;
	int hs_c = mask.cols / 2;
	for (int r = hs_r; r < source.rows - hs_r; ++r)
	{
		for (int c = hs_c; c < source.cols - hs_c; c++)
		{
			destination.at<float>(r, c) = filter(source, mask, r, c);
		}
	}

	return destination;
}


void harrisCorners(Mat source)
{
	Mat gray;

	cvtColor(source, gray, COLOR_BGR2GRAY);

	imshow("Img gray", gray);
	imshow("Original image", source);

	float sobel_x[9] = { -1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f };
	float sobel_y[9] = { -1.0f, -2.0f, -1.0f, 0.0, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f };

	Mat mask_x(3, 3, CV_32FC1, sobel_x);
	Mat mask_y(3, 3, CV_32FC1, sobel_y);

	Mat pxFilter = convolution(gray, mask_x);
	Mat pyFilter = convolution(gray, mask_y);
}

