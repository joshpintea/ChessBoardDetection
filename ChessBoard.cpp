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
#include <opencv2/objdetect.hpp>
#include <vector>
#include <algorithm>


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
	trainDescriptorHorizontal(this->horizontalHistPawn, basePath + "pawn/", 140);
	trainDescriptorHorizontal(this->horizontalHistRook, basePath + "rook/", 32);
	trainDescriptorHorizontal(this->horizontalHistEmpty, basePath + "empty/", 63);
	trainDescriptorHorizontal(this->horizontalHistKing, basePath + "king/", 16);
	trainDescriptorHorizontal(this->horizontalHistBishop, basePath + "bishop/", 32);
	trainDescriptorHorizontal(this->horizontalHistKnight, basePath + "knight/", 32);
	trainDescriptorHorizontal(this->horizontalHistQueen, basePath + "queen/", 32);

	//hog
	trainBishopDescriptor();
	trainKingDescriptor();
	trainPawnDescriptor();
	trainEmptyDescriptor();
	trainKnightDescriptor();
	trainQueenDescriptor();
	trainRookDescriptor();

}

void ChessBoard::colorClassification(Mat img)
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

	queue<Point> queue;

	for (c = 0; c < 8; c++)
	{
		for (r = 0; r < 8; r++)
		{
			Rect rr(r * 80, c * 80, 80, 80);
			Mat p = this->imgProjectedGray(rr).clone();

			int hist[256] = { 0 };
			int negative = 0;
			int positive = 0;
			int widthNegative = 0;
			int widthPositive = 0;
			for (int rr = c * 80 + 1; rr < (c+1)*80; rr++)
			{
				for (int cc = r * 80 + 1; cc < (r+1)*80; cc++)
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
		}
	}
	int even = -1;
	for (c = 0; c < 8; c++)
	{
		for (r = 0; r < 8; r++)
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
						if (p.at<uchar>(i,j) < 128)
						{
							black++;
						} else if (p.at<uchar>(i,j) > 128)
						{
							white++;
						}
					}

				}
				Rect rrr(r * 80, c * 80, 80, 80);
				Mat pp = this->imgProjectedGray(rrr).clone();
				imshow("p", p);
				imshow("src", pp);
				cout << black << "  " << white << "\n";
				waitKey();
				if (white > black)
				{
					this->chessBoardPieces[r][c].color = 1;
				} else // black > white ; back = 1 => color = 0 else color 1
				{
					this->chessBoardPieces[r][c].color = 0;
				}
			}
		}
	}
	if (even == -1)
	{
		even = rand() % 2; // 1 in a million :))
	}

}


void ChessBoard::reconstructChessBoard(Mat img, CHESSBOARD_RECONSTRUCT_SOLVER solver, CLASSIFICATION cls)
{

	this->colorClassification(img);
	for (int r = 0; r < 8; r++)
	{
		for (int c = 0; c < 8; c++)
		{
			cout << this->chessBoardPieces[r][c].toString() << " ";
		}
		cout << endl;
	}

	if (solver == METHOD1) {
		reconstructChessBoardMethod1(img, cls);
	} else {
		reconstructChessBoardMethod2(img, cls);
	}



	Mat result(640, 640, CV_8UC3);

	for (int r = 0; r < 8; r++)
	{
		for (int c = 0; c < 8; c++)
		{
			string filePath = (this->chessBoardPieces[r][c].backgroundColor == 0) ? this->darkDir : this->whiteDir;
			filePath = filePath + this->chessBoardPieces[r][c].type + "_";
			filePath = filePath + ((this->chessBoardPieces[r][c].color == 0) ? "dark" : "white") + ".png";
			Mat piece = imread(filePath, IMREAD_COLOR);
			//imshow("pp",piece);
			//cout << chessBoardPieces[r][c].toString()<<" ";
			resize(piece, piece, Size(80, 80));
			for(int i=r*80; i<(r+1)*80; i++)
			{
				for(int j=c*80; j<(c+1)*80; j++)
				{
					result.at<Vec3b>(i, j) = piece.at<Vec3b>(i - r * 80, j - c * 80);
				}
			}
			//waitKey();
		}
	}
	imshow("reconstr"+cls, result);
	waitKey();
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

			for (int h = 2 * width; h > height; h -= 20)
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
			if (this->chessBoardPieces[i][j].type == "empty")
			{
				continue;
			}
			switch (cls)
			{
			case HOG:
				res = classificationHog(board[i][j]);
				break;
			case HORIZONTAL:
				res = classificationHorizontal(board[i][j]);
				break;
			default:
				res = "empty";
				break;
			}

			this->chessBoardPieces[i][j].type = res;
			//cout << res << " ";
		}
		//cout << endl;
	}
	
	for (int r = 0; r < 8 ; r++)
	{
		for (int c = 0; c < 8; c++)
		{
			cout << this->chessBoardPieces[r][c].toString() << " ";
		}
		cout << endl;
	}
}

void ChessBoard::reconstructChessBoardMethod2(Mat img, CLASSIFICATION cls)
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

	for (r = 0; r < 8; r++)
	{
		for (c = 0; c < 8; c++)
		{
			Rect rr(r * 80, c * 80, 80, 80);
			Mat piece = imgProjected1(rr).clone();

			string res;
			if (this->chessBoardPieces[r][c].type == "empty")
			{
				continue;
			}
			switch (cls)
			{
			case HOG:
				res = classificationHog(piece);
				break;
			case HORIZONTAL:
				res = classificationHorizontal(piece);
				break;
			default:
				res = "empty";
				break;
			}

			this->chessBoardPieces[r][c].type = res;
		}
	}

	for (int r = 0; r < 8; r++)
	{
		for (int c = 0; c < 8; c++)
		{
			cout << this->chessBoardPieces[r][c].toString() << " ";
		}
		cout << endl;
	}
}


bool myCompare6(pair<float, int> a, pair<float, int> b)
{
	return a.first < b.first;
}



string ChessBoard::decide(float b, float p, float kn, float k, float q, float e, float r)
{
	string cat[] = { "bishop", "pawn", "knight", "king", "queen", "empty", "rook" };

	vector<pair<float, int>> pairs;
	pairs.push_back(make_pair(b, 0));
	pairs.push_back(make_pair(p, 1));
	pairs.push_back(make_pair(kn, 2));
	pairs.push_back(make_pair(k, 3));
	pairs.push_back(make_pair(q, 4));
	pairs.push_back(make_pair(e, 5));
	pairs.push_back(make_pair(r, 6));
	return cat[std::min_element(pairs.begin(), pairs.end(), myCompare6)->second];
}


float ChessBoard::vectors_distance(vector<float> a, vector<float> b)
{
	float sum = 0;

	for (int i = 0; i < a.size(); i++)
	{
		sum += ((a[i] - b[i]) * (a[i] - b[i]));
	}

	return sqrt(sum);
}

int len = 35;
string ChessBoard::classificationHog(Mat sourceImg)
{
	Mat img;
	resize(sourceImg, img, Size(64, 128));

	//imwrite(this->basePath + "/projres/" + to_string(len) + ".jpg", sourceImg);
	//len++;
	vector<float> descriptor;
	vector<float> descriptor2;
	Rect r3(0, 48, 64, 80);
	Mat img2 = img(r3).clone();
	HOGDescriptor h(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
	HOGDescriptor h2(Size(64, 80), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
	h.compute(img, descriptor);
	h2.compute(img2, descriptor2);

	float b = 0;
	for (int i = 0; i < bishopDescriptors.size(); i++)
	{
		b += vectors_distance(descriptor2, bishopDescriptors[i]);
	}
	float k = 0;
	for (int i = 0; i < kingDescriptors.size(); i++)
	{
		k += vectors_distance(descriptor2, kingDescriptors[i]);
	}
	float p = 0;
	for (int i = 0; i < pawnDescriptors.size(); i++)
	{
		p += vectors_distance(descriptor2, pawnDescriptors[i]);
	}
	float e = 0;
	for (int i = 0; i < emptyDescriptors.size(); i++)
	{
		e += vectors_distance(descriptor2, emptyDescriptors[i]);
	}
	float kn = 0;
	for (int i = 0; i < knightDescriptors.size(); i++)
	{
		kn += vectors_distance(descriptor2, knightDescriptors[i]);
	}
	float q = 0;
	for (int i = 0; i < queenDescriptors.size(); i++)
	{
		q += vectors_distance(descriptor2, queenDescriptors[i]);
	}
	float r = 0;
	for (int i = 0; i < rookDescriptors.size(); i++)
	{
		r += vectors_distance(descriptor2, rookDescriptors[i]);
	}
	string res = decide(b / bishopDescriptors.size(), p / pawnDescriptors.size(), kn / knightDescriptors.size(),
		k / kingDescriptors.size(), q / queenDescriptors.size(), e / emptyDescriptors.size(),
		r / rookDescriptors.size());

	return res;
	//return "PAWN";
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
			piece = "bishop";
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
			piece = "empty";
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
			piece = "king";
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
			piece = "knight";
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
			piece = "pawn";
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
			piece = "queen";
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
			piece = "rook";
		}
	}

	return piece;
}

void ChessBoard::trainBishopDescriptor()
{
	for (int i = 1; i <= 32; i++)
	{
		string filename = basePath + "bishop/";
		filename += std::to_string(i);
		filename += ".jpg";
		Mat img1 = imread(filename);
		resize(img1, img1, Size(64, 128));
		vector<float> descriptor1;
		vector<float> descriptor2;
		vector<float> descriptor3;
		vector<float> descriptor4;
		vector<float> descriptor5;
		HOGDescriptor h(Size(64, 80), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		/*
		h.compute(img1, descriptor1);
		bishopDescriptors.push_back(descriptor1);


		Rect r1(0, 16, 64, 112);
		Mat img2 = img1(r1).clone();
		resize(img2, img2, Size(64, 128));
		h.compute(img2, descriptor2);
		bishopDescriptors.push_back(descriptor2);

		Rect r2(0, 32, 64, 96);
		Mat img3 = img1(r2).clone();
		resize(img3, img3, Size(64, 128));
		h.compute(img3, descriptor3);
		bishopDescriptors.push_back(descriptor3);
		*/
		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		//resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		bishopDescriptors.push_back(descriptor4);
		/*
		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		bishopDescriptors.push_back(descriptor5);
		*/
	}
}

void ChessBoard::trainKingDescriptor()
{
	for (int i = 1; i <= 16; i++)
	{
		string filename = basePath + "king/";
		filename += std::to_string(i);
		filename += ".jpg";
		Mat img1 = imread(filename);
		resize(img1, img1, Size(64, 128));
		vector<float> descriptor1;
		vector<float> descriptor2;
		vector<float> descriptor3;
		vector<float> descriptor4;
		vector<float> descriptor5;
		HOGDescriptor h(Size(64, 80), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		/*
		h.compute(img1, descriptor1);
		kingDescriptors.push_back(descriptor1);


		Rect r1(0, 16, 64, 112);
		Mat img2 = img1(r1).clone();
		resize(img2, img2, Size(64, 128));
		h.compute(img2, descriptor2);
		kingDescriptors.push_back(descriptor2);

		Rect r2(0, 32, 64, 96);
		Mat img3 = img1(r2).clone();
		resize(img3, img3, Size(64, 128));
		h.compute(img3, descriptor3);
		kingDescriptors.push_back(descriptor3);
		*/
		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		//resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		kingDescriptors.push_back(descriptor4);
		/*
		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		kingDescriptors.push_back(descriptor5);
		*/
	}
}

void ChessBoard::trainPawnDescriptor()
{
	for (int i = 1; i <= 140; i++)
	{
		string filename = basePath + "pawn/";
		filename += std::to_string(i);
		filename += ".jpg";
		Mat img1 = imread(filename);
		resize(img1, img1, Size(64, 128));
		vector<float> descriptor1;
		vector<float> descriptor2;
		vector<float> descriptor3;
		vector<float> descriptor4;
		vector<float> descriptor5;
		HOGDescriptor h(Size(64, 80), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		/*
		h.compute(img1, descriptor1);
		pawnDescriptors.push_back(descriptor1);


		Rect r1(0, 16, 64, 112);
		Mat img2 = img1(r1).clone();
		resize(img2, img2, Size(64, 128));
		h.compute(img2, descriptor2);
		pawnDescriptors.push_back(descriptor2);

		Rect r2(0, 32, 64, 96);
		Mat img3 = img1(r2).clone();
		resize(img3, img3, Size(64, 128));
		h.compute(img3, descriptor3);
		pawnDescriptors.push_back(descriptor3);
		*/
		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		//resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		pawnDescriptors.push_back(descriptor4);
		/*
		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		pawnDescriptors.push_back(descriptor5);
		*/
	}
}

void ChessBoard::trainEmptyDescriptor()
{
	for (int i = 1; i <= 63; i++)
	{
		string filename = basePath + "empty/";
		filename += std::to_string(i);
		filename += ".jpg";
		Mat img1 = imread(filename);
		resize(img1, img1, Size(64, 128));
		vector<float> descriptor1;
		vector<float> descriptor2;
		vector<float> descriptor3;
		vector<float> descriptor4;
		vector<float> descriptor5;
		HOGDescriptor h(Size(64, 80), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		/*
		h.compute(img1, descriptor1);
		emptyDescriptors.push_back(descriptor1);


		Rect r1(0, 16, 64, 112);
		Mat img2 = img1(r1).clone();
		resize(img2, img2, Size(64, 128));
		h.compute(img2, descriptor2);
		emptyDescriptors.push_back(descriptor2);

		Rect r2(0, 32, 64, 96);
		Mat img3 = img1(r2).clone();
		resize(img3, img3, Size(64, 128));
		h.compute(img3, descriptor3);
		emptyDescriptors.push_back(descriptor3);
		*/
		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		//resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		emptyDescriptors.push_back(descriptor4);
		/*
		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		emptyDescriptors.push_back(descriptor5);
		*/
	}
}

void ChessBoard::trainKnightDescriptor()
{
	for (int i = 1; i <= 32; i++)
	{
		string filename = basePath + "knight/";
		filename += std::to_string(i);
		filename += ".jpg";
		Mat img1 = imread(filename);
		resize(img1, img1, Size(64, 128));
		vector<float> descriptor1;
		vector<float> descriptor2;
		vector<float> descriptor3;
		vector<float> descriptor4;
		vector<float> descriptor5;
		HOGDescriptor h(Size(64, 80), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		/*
		h.compute(img1, descriptor1);
		knightDescriptors.push_back(descriptor1);


		Rect r1(0, 16, 64, 112);
		Mat img2 = img1(r1).clone();
		resize(img2, img2, Size(64, 128));
		h.compute(img2, descriptor2);
		knightDescriptors.push_back(descriptor2);

		Rect r2(0, 32, 64, 96);
		Mat img3 = img1(r2).clone();
		resize(img3, img3, Size(64, 128));
		h.compute(img3, descriptor3);
		knightDescriptors.push_back(descriptor3);
		*/
		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		//resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		knightDescriptors.push_back(descriptor4);
		/*
		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		knightDescriptors.push_back(descriptor5);
		*/
	}
}

void ChessBoard::trainQueenDescriptor()
{
	for (int i = 1; i <= 16; i++)
	{
		string filename = basePath + "queen/";
		filename += std::to_string(i);
		filename += ".jpg";
		Mat img1 = imread(filename);
		resize(img1, img1, Size(64, 128));
		vector<float> descriptor1;
		vector<float> descriptor2;
		vector<float> descriptor3;
		vector<float> descriptor4;
		vector<float> descriptor5;
		HOGDescriptor h(Size(64, 80), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		/*
		h.compute(img1, descriptor1);
		queenDescriptors.push_back(descriptor1);


		Rect r1(0, 16, 64, 112);
		Mat img2 = img1(r1).clone();
		resize(img2, img2, Size(64, 128));
		h.compute(img2, descriptor2);
		queenDescriptors.push_back(descriptor2);

		Rect r2(0, 32, 64, 96);
		Mat img3 = img1(r2).clone();
		resize(img3, img3, Size(64, 128));
		h.compute(img3, descriptor3);
		queenDescriptors.push_back(descriptor3);
		*/
		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		//resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		queenDescriptors.push_back(descriptor4);
		/*
		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		queenDescriptors.push_back(descriptor5);
		*/
	}
}

void ChessBoard::trainRookDescriptor()
{
	for (int i = 1; i <= 32; i++)
	{
		string filename = basePath + "rook/";
		filename += std::to_string(i);
		filename += ".jpg";
		Mat img1 = imread(filename);
		resize(img1, img1, Size(64, 128));
		vector<float> descriptor1;
		vector<float> descriptor2;
		vector<float> descriptor3;
		vector<float> descriptor4;
		vector<float> descriptor5;
		HOGDescriptor h(Size(64, 80), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		/*
		h.compute(img1, descriptor1);
		rookDescriptors.push_back(descriptor1);


		Rect r1(0, 16, 64, 112);
		Mat img2 = img1(r1).clone();
		resize(img2, img2, Size(64, 128));
		h.compute(img2, descriptor2);
		rookDescriptors.push_back(descriptor2);

		Rect r2(0, 32, 64, 96);
		Mat img3 = img1(r2).clone();
		resize(img3, img3, Size(64, 128));
		h.compute(img3, descriptor3);
		rookDescriptors.push_back(descriptor3);
		*/
		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		//resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		rookDescriptors.push_back(descriptor4);
		/*
		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		rookDescriptors.push_back(descriptor5);
		*/
	}
}
