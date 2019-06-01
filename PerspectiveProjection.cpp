#include "stdafx.h"
#include "PerspectiveProjection.h"
#include <opencv2/opencv.hpp>
#include <algorithm>



Mat PerspectiveProjection::getPerspectiveTransform(cv::Point2f sourcePoints[4], cv::Point2f destinationPoints[4], PERSPECTIVE_TRANSFORM_IMPL impl) const
{
	Mat m;

	switch (impl)
	{
	case OPEN_CV:
		m = cv::getPerspectiveTransform(sourcePoints, destinationPoints);
		break;
	case PIZZA: // @To do
		cv::Point2f sourceMock[4];
		sourceMock[0] = Point(sourcePoints[0].y + 10, sourcePoints[0].x + 10);
		sourceMock[1] = Point(sourcePoints[1].y + 10, sourcePoints[1].x + 60);
		sourceMock[2] = Point(sourcePoints[2].y + 60, sourcePoints[2].x + 10);
		sourceMock[3] = Point(sourcePoints[3].y + 60, sourcePoints[3].x + 60);
		m = cv::getPerspectiveTransform(sourceMock, destinationPoints);
		break;
	}

	return m;
}


Mat PerspectiveProjection::perspectiveProjection(Mat tr, Mat source, PERSPECTIVE_TRANSFORM_IMPL impl, Size size)
{
	Mat dest;
	switch (impl)
	{
		case OPEN_CV:
			warpPerspective(source, dest, tr, size);
			break;
		case PIZZA:
			Mat mInv = tr.inv();

			Mat mapx(size.height, size.width, CV_32FC1);
			Mat mapy(size.height, size.width, CV_32FC1);

			dest = Mat(size.height, size.width, source.type());
			for (int r = 0; r < mapx.rows; r++)
			{
				for (int c = 0; c < mapx.cols; c++)
				{
					float xp = (mInv.at<double>(0, 0) * r + mInv.at<double>(0, 1) * c + mInv.at<double>(0, 2)) / (mInv.at<
						double>(2, 0) * r + mInv.at<double>(2, 1) * c + 1);
					float yp = (mInv.at<double>(1, 0) * r + mInv.at<double>(1, 1) * c + mInv.at<double>(1, 2)) / (mInv.at<
						double>(2, 0) * r + mInv.at<double>(2, 1) * c + 1);

					mapx.at<float>(r, c) = xp;
					mapy.at<float>(r, c) = yp;

					dest.at<Vec3b>(r, c) = source.at<Vec3b>((int)xp, (int)yp);
				}
			}


			break;
	}
	return dest;
}

bool compareSumPers(Point p1, Point p2)
{
	return (p1.x + p1.y) < (p2.x + p2.y);
}

bool compareDiffPers(Point p1, Point p2)
{
	return (p1.x - p1.y) < (p2.x - p2.y);
}

/**
 * tl is the point with the smallest sum of his coordinates
 * br is the point with the highest sum of his coordinates
 * bl is the point with the smallest dif of his coordinates
 * br is the point with the highest diff of his coordinates
 */
void PerspectiveProjection::getBorderBoxes(std::vector<Point> points, Point& tl, Point& tr, Point& bl, Point& br)
{
	std::sort(points.begin(), points.end(), compareSumPers);

	tl.x = points[0].x;
	tl.y = points[0].y;
	br.x = points[points.size() - 1].x;
	br.y = points[points.size() - 1].y;

	std::sort(points.begin(), points.end(), compareDiffPers);


	bl.x = points[0].x;
	bl.y = points[0].y;
	tr.x = points[points.size() - 1].x;
	tr.y = points[points.size() - 1].y;
}
