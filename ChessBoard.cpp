#include "stdafx.h"
#include "ChessBoard.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/videostab/ring_buffer.hpp>
#include "PerspectiveProjection.h"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <queue>

void ChessBoard::setPoints(vector<Point> points)
{
	this->points = points;
}

void ChessBoard::trainDescriptorHorizontal(vector<vector<int>> &desc, string path, int count)
{
	int r, c, i;
	for (i = 1; i <= count; i++)
	{
		string filename = path + std::to_string(i) + ".jpg";
		Mat gauss, dst, img2;
		Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		resize(img, img, Size(64, 128));
		Rect r2(0, 32, 64, 96);
		Mat img3 = img(r2).clone();
		resize(img3, img2, Size(80, 160));

		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(img2, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);

		vector<int> d(dst.rows, 0);

		for (r = 0; r < dst.rows; r++)
		{
			for (c = 0; c < dst.cols; c++)
			{
				if (dst.at<uchar>(r, c) > 250)
				{
					d[r]++;
				}
			}
		}

		desc.push_back(d);
	}
}

void showHistogram2(const std::string& name, int* hist, const int hist_cols, const int hist_height,
	bool histScale = true)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;

	if (histScale)
	{
		scale = (double)hist_height / max_hist;
	}

	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++)
	{
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));

		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


void ChessBoard::train()
{
	// train horizontal classification method
	trainDescriptorHorizontal(this->horizontalHistPawn, basePath + "pawn/", 32);
	trainDescriptorHorizontal(this->horizontalHistRook, basePath + "rook/", 32);
	trainDescriptorHorizontal(this->horizontalHistEmpty, basePath + "empty/", 32);
	trainDescriptorHorizontal(this->horizontalHistKing, basePath + "king/", 16);
	trainDescriptorHorizontal(this->horizontalHistBishop, basePath + "bishop/", 32);
	trainDescriptorHorizontal(this->horizontalHistKnight, basePath + "knight/", 32);
	trainDescriptorHorizontal(this->horizontalHistQueen, basePath + "queen/", 32);
}

void ChessBoard::reconstructChessBoard(Mat img, CHESSBOARD_RECONSTRUCT_SOLVER solver, CLASSIFICATION cls)
{
	int r, c, i;
	Size sizeImg(640, 640);
	PerspectiveProjection perspectiveProjectionUtil;

	Point tl, tr, bl, br;
	perspectiveProjectionUtil.getBorderBoxes(points, tl, tr, bl, br);

	Point2f sourcePoints[4] = { tl, tr, bl, br };
	Point2f destinationPoints1[4] = {
		Point2f(0, 0), Point2f(sizeImg.height, 0), Point2f(0, sizeImg.width), Point2f(sizeImg.height, sizeImg.width)
	};

	Mat projectionMatrix1 = perspectiveProjectionUtil.
		getPerspectiveTransform(sourcePoints, destinationPoints1, OPEN_CV);
	Mat imgProjected1 = perspectiveProjectionUtil.perspectiveProjection(projectionMatrix1, img, OPEN_CV, sizeImg);
	cvtColor(imgProjected1, this->imgProjectedGray, CV_BGR2GRAY);

	for (int i =0 ; i < 8; i++)
	{
		vector<Piece> row;
		for (int i = 0; i < 8; i++)
		{
			Piece p;
			row.push_back(p);
		}
		this->chessBoardPieces.push_back(row);
	}

	imshow("Img", imgProjectedGray);

	ofstream fout(this->basePath + "file.txt");
	queue<Point> queue;

	for (r = 0; r < 8; r++)
	{
		for (c = 0; c < 8; c++)
		{
			Rect rr(r * 80, c * 80, 80, 80);
			Mat p = this->imgProjectedGray(rr).clone();

			int hist[256] = { 0 };
			int negative = 0;
			int positive = 0;
			int widthNegative = 0;
			int widthPositive = 0;
			for (int rr = r * 80 + 1; rr < (r+1)*80; rr++)
			{
				for (int cc = c * 80 + 1; cc < (c+1)*80; cc++)
				{
					hist[imgProjectedGray.at<uchar>(rr, cc)]++;
				}
			}

			for (int i = 0; i < 256; i++)
			{
				if (i < 70 && hist[i] > 50)
				{
					negative += hist[i];
					widthNegative++;
				}

				if (i > 185 && hist[i] > 50)
				{
					positive += hist[i];
					widthPositive++;
				}
			}

			chessBoardPieces[r][c].positive = positive;
			chessBoardPieces[r][c].negative = negative;
			chessBoardPieces[r][c].widthPositive = widthPositive;
			chessBoardPieces[r][c].widthNegative = widthNegative;



			// printf("(%d, %d), (%d, %d), (%d, %d)\n",r,c, positive, widthPositive, negative, widthNegative);
			// std::cout << positive << " " << negative << "\n";
			// showHistogram2("asa", hist, 256, 256, true);
			// imshow("img", p);

			// waitKey();

			// sum /= (80 * 80);
			// this->chessBoardPieces[r][c].sumColor = sum;
		}
	}
	int even = -1;
	for (r = 0; r < 8; r++)
	{
		for (c = 0; c < 8; c++)
		{
			Piece p = chessBoardPieces[r][c];
			if (even != -1)
			{
				chessBoardPieces[r][c].backgroundColor = ((r + c) % 2 == 0) ? even : (1 - even);
			}

			if (p.negative > 4000 && p.positive < 500 && p.widthNegative < 20 && p.widthPositive < 3 )
			{
				chessBoardPieces[r][c].type = "empty";
				chessBoardPieces[r][c].backgroundColor = 0;
				if (even == -1)
				{
					even = ((r + c) % 2 == 0) ? 0 : 1;
				}
			}

			if (p.positive > 4000 && p.negative < 500 && p.widthPositive < 20 && p.widthNegative < 3)
			{
				chessBoardPieces[r][c].type = "empty";
				chessBoardPieces[r][c].backgroundColor = 1;
				if (even == -1)
				{
					even = ((r + c) % 2 == 0) ? 1 : 0;
				}
			}

			if (even == -1)
			{
				queue.push(Point(r, c));
			} else
			{
				Rect rr(r * 80 + 20, c * 80 + 20, 40, 40);
				Mat p = this->imgProjectedGray(rr).clone();

				int white = 0;
				int black = 0;
				for (int i = 0; i < p.rows; i++)
				{
					for (int j = 0 ; j < p.cols; j++)
					{
						if (p.at<uchar>(i,j) < 70)
						{
							black++;
						} else if (p.at<uchar>(i,j) > 185)
						{
							white++;
						}
					}
				}
				Rect rrr(r * 80, c * 80, 80, 80);
				Mat imrr = this->imgProjectedGray(rrr).clone();
				imshow("p", p);
				imshow("imsd", imrr);
				// std::cout << "white: " << white << " black " << black << " backcolor:" << chessBoardPieces[r][c].backgroundColor << std::endl;
				if (white > black)
				{
					cout << 1 << std::endl;
					this->chessBoardPieces[r][c].color = 1;
				} else // black > white ; back = 1 => color = 0 else color 1
				{
					cout << 0 << std::endl;
					this->chessBoardPieces[r][c].color = 0;
				}
				waitKey();
			}
		}
	}
	if (even == -1)
	{
		even = rand() % 2; // 1 in a million :))
	}
	/*
	while (!queue.empty())
	{
		Point p = queue.front();
		queue.pop();
		chessBoardPieces[p.x][p.y].backgroundColor = ((p.x + p.y) % 2 == 0) ? even : 1 - even;
		Rect rr(p.x * 80 + 10, p.y * 80 + 10, 40, 40);
		Mat pp = this->imgProjectedGray(rr).clone();

		int white = 0;
		int black = 0;
		for (int i = 0; i < pp.rows; i++)
		{
			for (int j = 0; j < pp.cols; j++)
			{
				if (pp.at<uchar>(i, j) < 70)
				{
					black++;
				}
				else if (pp.at<uchar>(i,j) > 185)
				{
					white++;
				}
			}
		}

		imshow("p", pp);
		std::cout << "white: " << white << " black " << black << " backcolor:" << chessBoardPieces[p.x][p.y].backgroundColor << std::endl;
		waitKey();

		if (white > black)
		{
			chessBoardPieces[p.x][p.y].color = 1;
		}
		else
		{
			chessBoardPieces[p.x][p.y].color = 0;
		}
	}*/

	for (r = 0; r < 8; r++)
	{
		for (c = 0; c < 8; c++)
		{
			fout << this->chessBoardPieces[r][c].toString() << " ";
		}
		fout << std::endl;
	}
	// this->reconstructChessBoardMethod1(img, cls);
}


void ChessBoard::reconstructChessBoardMethod1(Mat img, CLASSIFICATION cls)
{
	int r, c, i;
	Size sizeImg(640, 640);
	PerspectiveProjection perspectiveProjectionUtil;

	Point tl, tr, bl, br;
	perspectiveProjectionUtil.getBorderBoxes(points, tl, tr, bl, br);

	Point2f sourcePoints[4] = { tl, tr, bl, br };
	Point2f destinationPoints1[4] = {
		Point2f(0, 0), Point2f(sizeImg.height, 0), Point2f(0, sizeImg.width), Point2f(sizeImg.height, sizeImg.width)
	};

	Mat mInv = perspectiveProjectionUtil.getPerspectiveTransform(destinationPoints1, sourcePoints, OPEN_CV);

	// matrix 9 x 9
	vector<vector<Point>> chessBoardCorners;
	for (i = 0; i <= 9; i++)
	{
		vector<Point> row;
		chessBoardCorners.push_back(row);
	}

	// compute corners into the initial image
	for (c = 0; c <= sizeImg.width; c += 80)
	{
		for (r = 0; r <= sizeImg.height; r += 80)
		{
			float xp = (mInv.at<double>(0, 0) * r + mInv.at<double>(0, 1) * c + mInv.at<double>(0, 2)) / (mInv.at<
				double>(2, 0) * r + mInv.at<double>(2, 1) * c + 1);
			float yp = (mInv.at<double>(1, 0) * r + mInv.at<double>(1, 1) * c + mInv.at<double>(1, 2)) / (mInv.at<
				double>(2, 0) * r + mInv.at<double>(2, 1) * c + 1);

			chessBoardCorners[c / 80].push_back(Point((int)xp, (int)yp));
		}
	}

	vector<vector<Mat>> board;
	for (i = 0; i < 9; i++)
	{
		vector<Mat> ss;
		board.push_back(ss);
	}

	for (r = 1; r < 9; r++)
	{
		for (c = 1; c < 9; c++)
		{
			Point tll = chessBoardCorners[r - 1][c - 1];
			Point trr = chessBoardCorners[r - 1][c];
			Point bll = chessBoardCorners[r][c - 1];
			Point brr = chessBoardCorners[r][c];

			int width = brr.x - bll.x;
			int height = bll.y - tll.y + 5;

			for (int h = 140; h > height; h -= 20)
			{
				if ((bll.y - h) >= 0)
				{
					height = h;
					break;
				}
			}
			Rect rr(bll.x, bll.y - height, width, height);
			Mat i = img(rr).clone();

			board[r - 1].push_back(i);
		}
	}

	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			string res;
			switch (cls)
			{
			case HOG:
				res = classificationHog(board[i][j]);
				break;
			case HORIZONTAL:
				res = classificationHorizontal(board[i][j]);
				break;
			default:
				res = "SS";
				break;
			}

			cout << res << " ";
		}
		cout << endl;
	}
}

string ChessBoard::classificationHog(Mat img)
{
	return "PAWN";
}

string ChessBoard::classificationHorizontal(Mat img1)
{
	Mat res1, res2, gauss1, gauss2, dst1, dst2;
	resize(img1, res1, Size(80, 160));

	double k = 0.4;
	int pH = 50;
	int pL = (int)k * pH;

	GaussianBlur(res1, gauss1, Size(5, 5), 0.8, 0.8);
	Canny(gauss1, dst1, pL, pH, 3);

	vector<int> imgHorizontal(dst1.rows, 0);

	for (int r = 0; r < dst1.rows; r++)
	{
		for (int c = 0; c < dst1.cols; c++)
		{
			if (dst1.at<uchar>(r, c) > 250)
			{
				imgHorizontal[r]++;
			}
		}
	}

	int minValue = 100000000;
	string piece;

	int sum;
	for (int i = 0; i < horizontalHistBishop.size(); i++)
	{
		sum = 0;
		for (int c = 0; c < dst1.rows; c++)
		{
			sum += std::abs(imgHorizontal[c] - horizontalHistBishop[i][c]);
		}
		if (sum < minValue)
		{
			minValue = sum;
			piece = "Bishop";
		}
	}

	for (int i = 0; i < horizontalHistEmpty.size(); i++)
	{
		sum = 0;
		for (int c = 0; c < dst1.rows; c++)
		{
			sum += std::abs(imgHorizontal[c] - horizontalHistEmpty[i][c]);
		}
		if (sum < minValue)
		{
			minValue = sum;
			piece = "Empty";
		}
	}
	for (int i = 0; i < horizontalHistKing.size(); i++)
	{
		sum = 0;
		for (int c = 0; c < dst1.rows; c++)
		{
			sum += std::abs(imgHorizontal[c] - horizontalHistKing[i][c]);
		}
		if (sum < minValue)
		{
			minValue = sum;
			piece = "King";
		}
	}

	for (int i = 0; i < horizontalHistKnight.size(); i++)
	{
		sum = 0;
		for (int c = 0; c < dst1.rows; c++)
		{
			sum += std::abs(imgHorizontal[c] - horizontalHistKnight[i][c]);
		}
		if (sum < minValue)
		{
			minValue = sum;
			piece = "Knight";
		}
	}
	for (int i = 0; i < horizontalHistPawn.size(); i++)
	{
		sum = 0;
		for (int c = 0; c < dst1.rows; c++)
		{
			sum += std::abs(imgHorizontal[c] - horizontalHistPawn[i][c]);
		}
		if (sum < minValue)
		{
			minValue = sum;
			piece = "Pawn";
		}
	}
	for (int i = 0; i < horizontalHistQueen.size(); i++)
	{
		sum = 0;
		for (int c = 0; c < dst1.rows; c++)
		{
			sum += std::abs(imgHorizontal[c] - horizontalHistQueen[i][c]);
		}
		if (sum < minValue)
		{
			minValue = sum;
			piece = "Queen";
		}
	}

	for (int i = 0; i < horizontalHistRook.size(); i++)
	{
		sum = 0;
		for (int c = 0; c < dst1.rows; c++)
		{
			sum += std::abs(imgHorizontal[c] - horizontalHistRook[i][c]);
		}
		if (sum < minValue)
		{
			minValue = sum;
			piece = "Rook";
		}
	}

	return piece;
}