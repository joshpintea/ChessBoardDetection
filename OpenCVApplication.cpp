// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>
#include "time.h"
#include <fstream>
#include <iostream>
#include <experimental/filesystem>
#include "PerspectiveProjection.h"
#include "HoughTransformation.h"
#include <unordered_map>
#include <functional>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR); // Read the image

	if (!src.data) // Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = MAX_PATH - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360; // lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1]; // lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2]; // lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened())
	{
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0); // waits a key press to advance to the next frame
		if (c == 27)
		{
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break; //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
	                 (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10); // waits a key press to advance to the next frame
		if (c == 27)
		{
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break; //ESC pressed
		}
		if (c == 115)
		{
			//'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}
}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
		       x, y,
		       (int)(*src).at<Vec3b>(y, x)[2],
		       (int)(*src).at<Vec3b>(y, x)[1],
		       (int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int hist_cols, const int hist_height,
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

void showHistogramRow(const std::string& name, int* hist, const int histCols, const int histHeight,
                      bool imgScale = true)
{
	Mat imgHist(histHeight, histCols, CV_8UC3, CV_RGB(255, 255, 255));
	int max_hist = 0;
	for (int i = 0; i < histHeight; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	if (imgScale)
	{
		scale = (double)histCols / max_hist;
	}
	int baseline = histCols - 1;

	for (int y = 0; y < histHeight; y++)
	{
		Point p1 = Point(0, y);
		Point p2 = Point(cvRound(hist[y] * scale), y);

		line(imgHist, p1, p2, CV_RGB(255, 0, 255));
	}

	imshow(name, imgHist);
}


void cornerHarrisDemo()
{
	char path[MAX_PATH];
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.02;

	int thresh = 200;
	std::cin >> thresh;
	while (openFileDlg(path))
	{
		Mat src = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		Mat source;
		resizeImg(src, source, 600, true);

		Mat dst = Mat::zeros(source.size(), CV_32FC1);
		cornerHarris(source, dst, blockSize, apertureSize, k);

		Mat dst_norm, dst_norm_scaled;
		normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		convertScaleAbs(dst_norm, dst_norm_scaled);

		for (int i = 0; i < dst_norm.rows; i++)
		{
			for (int j = 0; j < dst_norm.cols; j++)
			{
				if ((int)dst_norm.at<float>(i, j) > thresh)
				{
					circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 5, 8, 0);
				}
			}
		}

		imshow("Corners window", dst_norm_scaled);
		waitKey();
	}
}

Mat transform_perspective(Mat source, cv::Point2f sourcePoints[4], int s)
{
	Point2f destinationPoints[4];
	destinationPoints[0] = Point2f(0, 0);
	destinationPoints[1] = Point2f(s, 0);
	destinationPoints[2] = Point2f(0, s);
	destinationPoints[3] = Point2f(s, s);


	// std::cout <<"Source points 1\n";
	// for (int i = 0 ; i < 4; i++)
	// {
	// 	std::cout << sourcePoints[i] << " ";
	// }
	// std::cout <<"\nDestination points\n";
	//
	// for (int i =0 ; i < 4 ; i++)
	// {
	// 	std::cout << destinationPoints[i] << " ";
	// }
	// std::cout << "\n";

	Mat transformMatrix = getPerspectiveTransform(sourcePoints, destinationPoints);
	// std::cout << "1:" << transformMatrix << std::endl;
	// std::cout << "Transformation matrix" << std::endl;
	//
	// std::cout << transformMatrix;
	// for (int i =0 ; i < transformMatrix.rows; i++)
	// {
	// 	for (int j = 0 ; j < transformMatrix.cols; j++)
	// 	{
	// 		std::cout << transformMatrix.at<float>(i, j) << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::cout << std::endl;
	Mat dst;

	warpPerspective(source, dst, transformMatrix, Size(s, s));
	return dst;
}

bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2)
{
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i > j);
}



void testHoughTransformation()
{
	char path[MAX_PATH];

	HoughTransformation houghTransformationUtil;

	while (openFileDlg(path))
	{
		Mat imgResized, gray, edges, dest, grayWithoutNoises;
		Mat img = imread(path, IMREAD_COLOR);
		resizeImg(img, imgResized, 640, true);
		cvtColor(imgResized, gray, COLOR_BGR2GRAY);

		dest = imgResized.clone();
		// remove noises
		grayWithoutNoises = filterGaussianNoises(gray, 5);

		// apply edge filter
		Canny(grayWithoutNoises, edges, 100, 200);

		std::vector<Point> lines;
		houghTransformationUtil.transform(edges, 130, 180, lines, PIZZA_H);
		for (auto l : lines)
		{
			circle(imgResized, l, 2, Scalar(40), 2, 8, 0);
		}
		imshow("edges", edges);
		imshow("img", imgResized);

		// harrisCorners(imgResized, 20.0f);
		waitKey();
	}
}

void harrisCornersTest()
{
	char path[MAX_PATH];

	while (openFileDlg(path))
	{
		Mat imgResized;
		Mat img = imread(path, IMREAD_COLOR);
		resizeImg(img, imgResized, 600, true);

		harrisCorners(imgResized, 20.0f);
		waitKey();
	}
}

void test()
{
	char path[MAX_PATH];

	while (openFileDlg(path))
	{
		Mat imgResized;
		Mat img = imread(path, IMREAD_COLOR);

		resizeImg(img, imgResized, 600, true);

		for (int i = 0; i < imgResized.rows; i++)
		{
			for (int j = 0; j < imgResized.cols; j++)
			{
				Vec3b color = imgResized.at<Vec3b>(i, j);

				if (color[2] > 55 && color[2] < 120)
				{
					imgResized.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}
			}
		}

		imshow("Img without red chanel", imgResized);
		// harrisCorners(imgResized, 20.0f);
		waitKey();
	}
}

void testCannyEdgeDetection()
{
	char path[MAX_PATH];

	while (openFileDlg(path))
	{
		Mat imgResized;
		Mat img = imread(path, IMREAD_COLOR);
		resizeImg(img, imgResized, 600, true);

		imshow("Original image", imgResized);

		Mat gray;
		cvtColor(imgResized, gray, COLOR_BGR2GRAY);

		Mat byCanny = cannyAlgorithm(gray);
		imshow("canny", byCanny);
		waitKey();
	}
}


float vectors_distance(vector<float> a,  vector<float> b)
{
	float sum = 0;

	for(int i=0; i<a.size(); i++)
	{
		sum += ((a[i] - b[i])*(a[i] - b[i]));
	}

	return  sqrt(sum);
}




vector<vector<float>> bishopDescriptors;
vector<vector<float>> kingDescriptors;
vector<vector<float>> pawnDescriptors;
vector<vector<float>> emptyDescriptors;
vector<vector<float>> knightDescriptors;
vector<vector<float>> queenDescriptors;
vector<vector<float>> rookDescriptors;

string basePath = "D:/MyWorkSpace/Image Processing/ChessBoardDetection/ChessBoardDetection/Images/training_images/";

void trainBishopDescriptor()
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
		HOGDescriptor h(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		h.compute(img1, descriptor1);
		bishopDescriptors.push_back(descriptor1);

		/*
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

		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		bishopDescriptors.push_back(descriptor4);

		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		bishopDescriptors.push_back(descriptor5);
		*/
	}
}

void trainKingDescriptor()
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
		HOGDescriptor h(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		h.compute(img1, descriptor1);
		kingDescriptors.push_back(descriptor1);

		/*
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

		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		kingDescriptors.push_back(descriptor4);

		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		kingDescriptors.push_back(descriptor5);
		*/
	}
}

void trainPawnDescriptor()
{
	for (int i = 1; i <= 128; i++)
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
		HOGDescriptor h(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		h.compute(img1, descriptor1);
		pawnDescriptors.push_back(descriptor1);

		/*
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

		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		pawnDescriptors.push_back(descriptor4);

		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		pawnDescriptors.push_back(descriptor5);
		*/
	}
}

void trainEmptyDescriptor()
{
	for (int i = 1; i <= 64; i++)
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
		HOGDescriptor h(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		h.compute(img1, descriptor1);
		emptyDescriptors.push_back(descriptor1);

		/*
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

		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		emptyDescriptors.push_back(descriptor4);

		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		emptyDescriptors.push_back(descriptor5);
		*/
	}
}

void trainKnightDescriptor()
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
		HOGDescriptor h(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		h.compute(img1, descriptor1);
		knightDescriptors.push_back(descriptor1);

		/*
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

		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		knightDescriptors.push_back(descriptor4);

		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		knightDescriptors.push_back(descriptor5);
		*/
	}
}

void trainQueenDescriptor()
{
	for (int i = 1; i <= 32; i++)
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
		HOGDescriptor h(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		h.compute(img1, descriptor1);
		queenDescriptors.push_back(descriptor1);

		/*
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

		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		queenDescriptors.push_back(descriptor4);

		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		queenDescriptors.push_back(descriptor5);
		*/
	}
}

void trainRookDescriptor()
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
		HOGDescriptor h(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
		h.compute(img1, descriptor1);
		rookDescriptors.push_back(descriptor1);

		/*
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

		Rect r3(0, 48, 64, 80);
		Mat img4 = img1(r3).clone();
		resize(img4, img4, Size(64, 128));
		h.compute(img4, descriptor4);
		rookDescriptors.push_back(descriptor4);

		Rect r4(0, 64, 64, 64);
		Mat img5 = img1(r4).clone();
		resize(img5, img5, Size(64, 128));
		h.compute(img5, descriptor5);
		rookDescriptors.push_back(descriptor5);
		*/
	}
}

void hog(Mat sourceImg)
{
	//Mat img = imread("Images/old_training_images/bp/20160529_214403.jpg",CV_LOAD_IMAGE_COLOR);
	//Mat img1 = imread("Images/old_training_images/bp/20160529_214403.jpg");
	//Mat img2 = imread("Images/old_training_images/wr/20160529_214026.jpg");
	//resize(img1, img1,Size(64,128));
	//resize(img2, img2,Size(64,128));
	/*
	img.convertTo(img, CV_32F);
	Mat gx, gy;
	Sobel(img, gx, CV_32F, 1, 0, 1);
	Sobel(img, gy, CV_32F, 0, 1, 1);
	
	Mat mag, angle;
	cartToPolar(gx, gy, mag, angle, 1);

	for(int i=0; i<angle.rows; i++)
	{
		for(int j=0; j<angle.cols; j++)
		{
			if(angle.at<float>(i,j)>=180)
			{
				angle.at<float>(i, j) -= 180;
			}
		}
	}
	float maximus = 0;
	for (int i = 0; i < mag.rows; i++)
	{
		for (int j = 0; j < mag.cols; j++)
		{
			if(mag.at<float>(i, j)>maximus)
			{
				maximus = img.at<float>(i, j);
			}
		}
		cout << maximus<<"\n";
	}

	unordered_map<int, vector<float>> histograms;
	for(int i=0; i<img.rows; i+=8)
	{
		for(int j=0; j<img.cols; j+=8)
		{
			vector<float> hist(9,0);
			//nucleu
			for (int x = 0; x < 8; x++)
			{
				for (int y = 0; y < 8; y++)
				{
					int poz = (int)(angle.at<float>(i + x, j + y)/20)%9;
					float dif = angle.at<float>(i + x, j + y) - poz * 20;
					//cout << dif << "\n";
					dif = dif * 100 / 20;
					hist[poz%9] += mag.at<float>(i + x, j + y)*(100 - dif) / 100;
					hist[(poz+1)%9] += mag.at<float>(i + x, j + y)*dif / 100;
					if(i==8 && j==24)
					{
						cout << mag.at<float>(i + x, j + y)<<" ";
					}
				}
				if (i == 8 && j == 24)
				{
					cout << "\n";
				}
			}
			
			histograms[i/8+j/8] = hist;
		}
	}

	

	for(auto it : histograms[0])
	{
		//cout << it << " ";
	}

	//cout << img.rows << " " << img.cols;
	imshow("image", img);
	imshow("gx", gx);
	imshow("gy", gy);
	imshow("mag", mag);
	waitKey(0);
	*/

	
	Mat img;
	resize(sourceImg, img, Size(64, 128));
	imshow("img", img);
	vector<float> descriptor;
	HOGDescriptor h(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 0, -1, 0, 0.2, 0);
	h.compute(img, descriptor);
	
	float b=0;
	for(int i=0; i<bishopDescriptors.size(); i++)
	{
		b+=vectors_distance(descriptor, bishopDescriptors[i]);
	}
	float k = 0;
	for (int i = 0; i < kingDescriptors.size(); i++)
	{
		k+=vectors_distance(descriptor, kingDescriptors[i]);
	}
	float p = 0;
	for (int i = 0; i < pawnDescriptors.size(); i++)
	{
		p+=vectors_distance(descriptor, pawnDescriptors[i]);
	}
	float e = 0;
	for (int i = 0; i < emptyDescriptors.size(); i++)
	{
		e += vectors_distance(descriptor, emptyDescriptors[i]);
	}
	float kn = 0;
	for (int i = 0; i < knightDescriptors.size(); i++)
	{
		kn += vectors_distance(descriptor, knightDescriptors[i]);
	}
	float q = 0;
	for (int i = 0; i < queenDescriptors.size(); i++)
	{
		q += vectors_distance(descriptor, queenDescriptors[i]);
	}
	float r = 0;
	for (int i = 0; i < rookDescriptors.size(); i++)
	{
		r += vectors_distance(descriptor, rookDescriptors[i]);
	}
	cout << "bishop: " << b / bishopDescriptors.size() << "\n";
	cout << "king: " << k / kingDescriptors.size() << "\n";
	cout << "pawn: " << p / pawnDescriptors.size()<< "\n";
	cout << "empty: " << e / emptyDescriptors.size() << "\n";
	cout << "knight: " << kn / knightDescriptors.size() << "\n";
	cout << "queen: " << q / queenDescriptors.size() << "\n";
	cout << "rook: " << r / rookDescriptors.size() << "\n";
}


int thresh = 200;
Mat imgResized, edges, gray;

bool compareVertical(pair<Point2f, Point2f> p1, pair<Point2f, Point2f> p2)
{
	return (p1.first.x + p1.second.x) < (p2.first.x + p2.first.x);
}

bool compareOrizontal(pair<Point2f, Point2f> p1, pair<Point2f, Point2f> p2)
{
	return (p1.first.y + p1.second.y) < (p2.first.y + p2.first.y);
}

bool intersection2(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
	Point2f& r)
{
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x * d2.y - d1.y * d2.x;
	if (abs(cross) < /*EPS*/1.0) {
		return false;
	}

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}

void onTrackBarTresh(int, void*)
{
	std::vector<Point> points;
	std::vector<Vec4i> lines;
	Size sizeImg(640, 640);
	Canny(gray, edges, 50, thresh);

	HoughLinesP(edges, lines, 1, CV_PI / 180, thresh, 10, 250);

	Mat imgs = imgResized.clone();

	vector<pair<Point2f, Point2f>> verticalLines;
	vector<pair<Point2f, Point2f>> orizontalLines;
	int *cols = (int*)calloc((int)edges.cols, sizeof(int));
	int *rows = (int*)calloc((int)edges.rows, sizeof(int));
	bool d = false;
	for (auto l : lines)
	{
		Point2f p1(l[0], l[1]);
		Point2f p2(l[2], l[3]);

		float m = (p2.y - p1.y) / (p2.x - p1.x);

		if (std::abs(m) < 0.1)
		{
			d = false;
			for (int y = p1.y - 5; y < p1.y + 5; y++)
			{
				if (y >= 0 && y < edges.rows) {
					d |= rows[y];
				}
			}
			if (!d) {
				orizontalLines.push_back(make_pair(p1, p2));
			}
		}
		else if (std::abs(m) > 3)
		{
			d = false;
			for (int x = p1.x - 5; x < p1.x + 5; x++)
			{
				if (x >= 0 && x < edges.cols) {
					d |= cols[x];
				}
			}
			if (!d) {
				verticalLines.push_back(make_pair(p1, p2));
			}
		}
	}
	sort(verticalLines.begin(), verticalLines.end(), compareVertical);
	sort(orizontalLines.begin(), orizontalLines.end(), compareOrizontal);

	Point2f tl, tr, bl, br;
	int sizeO = orizontalLines.size();
	int sizeV = verticalLines.size();

	intersection2(verticalLines[1].first, verticalLines[1].second, orizontalLines[1].first, orizontalLines[1].second, tl);
	intersection2(verticalLines[1].first, verticalLines[1].second, orizontalLines[sizeO - 1].first, orizontalLines[sizeO - 1].second, bl);
	intersection2(verticalLines[sizeV - 1].first, verticalLines[sizeV - 1].second, orizontalLines[1].first, orizontalLines[1].second, tr);
	intersection2(verticalLines[sizeV - 1].first, verticalLines[sizeV - 1].second, orizontalLines[sizeO - 1].first, orizontalLines[sizeO - 1].second, br);

	for (auto p : verticalLines)
	{
		line(imgs, p.first, p.second, Scalar(0, 0, 255), 3, LINE_AA);
	}

	for (auto p : orizontalLines)
	{
		line(imgs, p.first, p.second, Scalar(255, 0, 0), 3, LINE_AA);
	}
	// // destination points for OPENCV Method
	Point2f destinationPoints1[4] = {
		Point2f(0, 0), Point2f(sizeImg.height, 0), Point2f(0, sizeImg.width), Point2f(sizeImg.height, sizeImg.width)
	};

	PerspectiveProjection perspectiveProjectionUtil;
	Point2f sourcePoints[4] = { tl, tr, bl, br };
	// projection with opencv implementation
	Mat projectionMatrix1 = perspectiveProjectionUtil.getPerspectiveTransform(sourcePoints, destinationPoints1, OPEN_CV);
	Mat imgProjected1 = perspectiveProjectionUtil.perspectiveProjection(projectionMatrix1, imgResized, OPEN_CV, sizeImg);


	vector<vector<Mat>> board;

	for (int i = 0; i < 8; i++)
	{
		vector<Mat> m;
		board.push_back(m);
	}

	int s = 0;
	char imageName[512];
	// extragere piese de pe tabla de sah.
	for (int y = 0; y < 7; y++) {
		for (int x = 0; x < 8; x++)
		{
			if (y == 0)
			{
				Rect r1(x * 80, y * 80, 80, 80);
				Mat img2 = imgProjected1(r1).clone();
				board[0].push_back(img2);

				Rect r2(x * 80, 0, 80, 160);
				Mat img3 = imgProjected1(r2).clone();
				board[1].push_back(img3);
			}
			else
			{
				Rect r1(x * 80, y * 80, 80, 160);
				Mat img2 = imgProjected1(r1).clone();

				board[y + 1].push_back(img2);
			}

		}
	}


	
	for (int i = 0; i < 8 ; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			cout << i << " " << j << std::endl << std::endl;
			hog(board[i][j]);
			cout << std::endl;
			cout << std::endl;
		}
	}

	imshow("Parallel lines", imgs);
	imshow("Chessboard projection", imgProjected1);
}

void chessBoardDetection()
{
	char path[MAX_PATH];
	openFileDlg(path);

	Mat grayWithoutNoises;
	Mat source = imread(path, IMREAD_COLOR);
	
	// resize image
	resizeImg(source, imgResized, 1024, true);
	
	// convert to gray scale
	cvtColor(imgResized, gray, COLOR_BGR2GRAY);

	// train hog 
	trainBishopDescriptor();
	trainKingDescriptor();
	trainPawnDescriptor();
	trainEmptyDescriptor();
	trainKnightDescriptor();
	trainQueenDescriptor();
	trainRookDescriptor();

	char TrackbarName[50];
	sprintf(TrackbarName, "Treshhold x %d", 255);
	
	namedWindow("Parallel lines", 1);
	createTrackbar(TrackbarName, "Parallel lines", &thresh, 255, onTrackBarTresh);
	
	onTrackBarTresh(thresh, 0);
	//
	waitKey();
}

void trainHog()
{
	trainBishopDescriptor();
	trainKingDescriptor();
	trainPawnDescriptor();
	trainEmptyDescriptor();
	trainKnightDescriptor();
	trainQueenDescriptor();
	trainRookDescriptor();
}

vector<Point> points;

void extractBoard(Mat img)
{
	Size sizeImg(640, 640);
	PerspectiveProjection perspectiveProjectionUtil;

	Point tl, tr, bl, br;
	perspectiveProjectionUtil.getBorderBoxes(points, tl, tr, bl, br);

	Point2f sourcePoints[4] = { tl, tr, bl, br };
	Point2f destinationPoints1[4] = {
		Point2f(0, 0), Point2f(sizeImg.height, 0), Point2f(0, sizeImg.width), Point2f(sizeImg.height, sizeImg.width)
	};

	Mat projectionMatrix1 = perspectiveProjectionUtil.getPerspectiveTransform(sourcePoints, destinationPoints1, OPEN_CV);
	Mat imgProjected1 = perspectiveProjectionUtil.perspectiveProjection(projectionMatrix1, img, OPEN_CV, sizeImg);


	vector<vector<Mat>> board;

	for (int i = 0; i < 8; i++)
	{
		vector<Mat> m;
		board.push_back(m);
	}

	int s = 0;
	char imageName[512];
	// extragere piese de pe tabla de sah.
	for (int y = 0; y < 7; y++) {
		for (int x = 0; x < 8; x++)
		{
			if (y == 0)
			{
				Rect r1(x * 80, y * 80, 80, 80);
				Mat img2 = imgProjected1(r1).clone();
				board[0].push_back(img2);

				Rect r2(x * 80, 0, 80, 160);
				Mat img3 = imgProjected1(r2).clone();
				board[1].push_back(img3);
			}
			else
			{
				Rect r1(x * 80, y * 80, 80, 160);
				Mat img2 = imgProjected1(r1).clone();

				board[y + 1].push_back(img2);
			}

		}
	}



	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			cout << i << " " << j << std::endl << std::endl;
			hog(board[i][j]);
			cout << std::endl;
			cout << std::endl;
		}
	}

	imshow("Board projected", imgProjected1);
}

void onMouseClick(int event, int x, int y, int flags, void *param)
{
	if (event == CV_EVENT_LBUTTONDOWN && points.size() != 4)
	{
		Mat &img = *((Mat*)(param)); // 1st cast it back, then deref
		circle(img, Point(x, y), 5, Scalar(0, 255, 0), 1);

		points.push_back(Point(x, y));

		imshow("Original image", img);

		if (points.size() == 4)
		{
			extractBoard(img);
		}
	}
}

void chessBoardReconstruct()
{
	char path[MAX_PATH];
	openFileDlg(path);

	Mat grayWithoutNoises, imgResized;
	Mat source = imread(path, IMREAD_COLOR);
	points.clear();
	// resize image
	resizeImg(source, imgResized, 1024, true);
	namedWindow("Original image");
	imshow("Original image", imgResized);
	setMouseCallback("Original image", onMouseClick, &imgResized);
	waitKey();
}


int main()
{
	int op;

	srand(time(NULL));

	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 11 - Chess board detection\n");
		printf(" 12 - Harris corner detection\n");
		printf(" 13 - Canny edge detection\n");
		printf(" 14 -Remove red chanel\n");
		printf(" 15 - Test Hough Transformation\n");
		printf(" 16 - Chessboard reconstruction\n");
		printf(" 17 - Train hog\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
			break;
		case 10:
			cornerHarrisDemo();
			break;
		case 11:
			chessBoardDetection();
			break;
		case 12:
			harrisCornersTest();
			break;

		case 13:
			testCannyEdgeDetection();
			break;
		case 14:
			test();
			break;
		case 15:
			testHoughTransformation();
			break;
		case 16:
			// hog();
			chessBoardReconstruct();
			break;
		case 17:
			// hog();
			trainHog();
			break;
		}
	}
	while (op != 0);

	return 0;
}
