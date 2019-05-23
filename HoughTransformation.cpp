#include "stdafx.h"
#include "common.h"
#include "HoughTransformation.h"

using namespace cv;
using namespace std;

void polarToCartesian(double rho, int theta, Point& p1, Point& p2)
{
	int x0 = cvRound(rho * cos(theta));
	int y0 = cvRound(rho * sin(theta));

	p1.x = cvRound(x0 + 1000 * (-sin(theta)));
	p1.y = cvRound(y0 + 1000 * (cos(theta)));

	p2.x = cvRound(x0 - 1000 * (-sin(theta)));
	p2.y = cvRound(y0 - 1000 * (cos(theta)));
}

bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
                  Point2f& r)
{
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x * d2.y - d1.y * d2.x;
	if (abs(cross) < /*EPS*/2.0)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}

void HoughTransformation::transform(Mat edges, int thresh, int t, std::vector<Point>& points,
				HOUGH_TRANSFORMATION_IMPL impl)
{
	std::vector<Vec4i> lines;

	switch (impl)
	{
		case OPEN_CV_H:
			HoughLinesP(edges, lines, 1, CV_PI / 180, 200, 10, 250);
			for (auto l : lines)
			{
				Point p1(l[0], l[1]);
				Point p2(l[2], l[3]);

				if (isInside(edges, p1.y, p1.x))
				{
					points.push_back(p1);
				}
				if (isInside(edges, p2.y, p2.x))
				{
					points.push_back(p2);
				}
			}
			
			break;
		case PIZZA_H:
			int maxDistance = hypot(edges.rows, edges.cols);
			vector<vector<int>> votes(2 * maxDistance, vector<int>(t + 1, 0));
			int rho, i, j, theta;
			for (i = 0; i < edges.rows; ++i)
			{
				for (j = 0; j < edges.cols; ++j)
				{
					if (edges.at<uchar>(i, j) > 254)
					{
						for (theta = 0; theta <= t; theta += 1)
						{
							rho = round(j * cos(theta - 90) + i * sin(theta - 90)) + maxDistance;
							votes[rho][theta]++;
						}
					}
				}
			}

			vector<pair<Point, Point>> lin;
			for (i = 0; i < votes.size(); ++i)
			{
				for (j = 0; j < votes[i].size(); ++j)
				{
					if (votes[i][j] >= thresh)
					{
						rho = i - maxDistance;
						theta = j - 90;

						Point p1, p2;
						// line(dest, p1, p2, Scalar(255), 1, LINE_8, 0);
						polarToCartesian(rho, theta, p1, p2);
						lin.push_back(make_pair(p1, p2));
					}
				}
			}

			// imshow("dest", dest);

			bool intersect;
			for (i = 0; i < lin.size() - 1; i++)
			{
				for (int j = i + 1; j < lin.size(); j++)
				{
					Point2f p;
					intersect = intersection(lin[i].first, lin[i].second, lin[j].first, lin[j].second, p);
					if (intersect && isInside(edges, p.y, p.x))
					{
						points.push_back(p);
					}
				}
			}
			break;
	}
}
