// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>
#include <time.h>
#include <fstream>
#include <iostream>
#include <experimental/filesystem>
#include "PerspectiveProjection.h"

#define WEAK 128
#define STRONG 255

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
	while (openFileDlg(path)) {
	
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

Mat convolutieGaussSeparat(Mat img, int w) {
	float sigma = w / 6.0;
	vector<float> Gx;
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			if (i == w / 2) {
				Gx.push_back((1 / (sqrt(2 * PI)*sigma))*exp(-((j - w / 2)*(j - w / 2)) / (2 * sigma*sigma)));
			}
			else {
				Gx.push_back(0);
			}
		}
	}

	vector<float> Gy;
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			if (j == w / 2) {
				Gy.push_back((1 / (sqrt(2 * PI)*sigma))*exp(-((i - w / 2)*(i - w / 2)) / (2 * sigma*sigma)));
			}
			else {
				Gy.push_back(0);
			}
		}
	}

	Mat newImg = img.clone();
	Mat newImg2 = img.clone();
	int k = (w - 1) / 2;
	for (int i = k; i < img.rows - k; i++) {
		for (int j = k; j < img.cols - k; j++) {
			float suma = 0;
			for (int u = 0; u < w; u++) {
				for (int v = 0; v < w; v++) {
					suma += (Gx[u*w + v] * img.at<uchar>(i + u - k, j + v - k));
				}
			}
			newImg.at<uchar>(i, j) = suma;
		}
	}

	for (int i = k; i < img.rows - k; i++) {
		for (int j = k; j < img.cols - k; j++) {
			float suma = 0;
			for (int u = 0; u < w; u++) {
				for (int v = 0; v < w; v++) {
					suma += (Gy[u*w + v] * newImg.at<uchar>(i + u - k, j + v - k));
				}
			}
			newImg2.at<uchar>(i, j) = suma;
		}
	}
	return newImg2;
}

Mat cannyBoti(Mat src) {
	Mat temp = src.clone();
	Mat modul = Mat::zeros(src.size(), CV_8UC1);
	Mat directie = Mat::zeros(src.size(), CV_8UC1);

	temp = convolutieGaussSeparat(src, 5);
	vector<int> nucleux;
	nucleux.push_back(-1);
	nucleux.push_back(0);
	nucleux.push_back(1);
	nucleux.push_back(-2);
	nucleux.push_back(0);
	nucleux.push_back(2);
	nucleux.push_back(-1);
	nucleux.push_back(0);
	nucleux.push_back(1);

	vector<int> nucleuy;
	nucleuy.push_back(1);
	nucleuy.push_back(2);
	nucleuy.push_back(1);
	nucleuy.push_back(0);
	nucleuy.push_back(0);
	nucleuy.push_back(0);
	nucleuy.push_back(-1);
	nucleuy.push_back(-2);
	nucleuy.push_back(-1);

	int w = 3;
	int k = 1;
	int dir;
	for (int i = k; i < temp.rows - k; i++) {
		for (int j = k; j < temp.cols - k; j++) {
			int sumax = 0;
			int sumay = 0;
			for (int u = 0; u < w; u++) {
				for (int v = 0; v < w; v++) {
					sumax += (nucleux[u*w + v] * temp.at<uchar>(i + u - k, j + v - k));
					sumay += (nucleuy[u*w + v] * temp.at<uchar>(i + u - k, j + v - k));
				}
			}
			modul.at<uchar>(i, j) = sqrt(sumax*sumax + sumay * sumay) / 5.65;
			float teta = atan2((float)sumay, (float)sumax);
			if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8)) dir = 0;
			if ((teta > PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) dir = 1;
			if ((teta > -PI / 8 && teta < PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8) dir = 2;
			if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta < -PI / 8)) dir = 3;
			directie.at<uchar>(i, j) = dir;
		}
	}

	Mat subtiere = modul.clone();

	for (int i = 0; i < modul.rows; i++) {
		for (int j = 0; j < modul.cols; j++) {
			int m = modul.at<uchar>(i, j);
			switch (directie.at<uchar>(i, j))
			{
			case 2:
				if (j - 1 >= 0) {
					if (modul.at<uchar>(i, j - 1) > modul.at<uchar>(i, j)) {
						m = 0;
					}
				}
				if (j + 1 < modul.cols) {
					if (modul.at<uchar>(i, j + 1) > modul.at<uchar>(i, j)) {
						m = 0;
					}
				}
				break;
			case 1:
				if (j + 1 < modul.cols && i - 1 >= 0) {
					if (modul.at<uchar>(i - 1, j + 1) > modul.at<uchar>(i, j)) {
						m = 0;
					}
				}
				if (j - 1 >= 0 && i + 1 < modul.rows) {
					if (modul.at<uchar>(i + 1, j - 1) > modul.at<uchar>(i, j)) {
						m = 0;
					}
				}
				break;
			case 0:
				if (i - 1 >= 0) {
					if (modul.at<uchar>(i - 1, j) > modul.at<uchar>(i, j)) {
						m = 0;
					}
				}
				if (i + 1 < modul.rows) {
					if (modul.at<uchar>(i + 1, j) > modul.at<uchar>(i, j)) {
						m = 0;
					}
				}
				break;
			case 3:
				if (j + 1 < modul.cols && i + 1 < modul.rows) {
					if (modul.at<uchar>(i + 1, j + 1) > modul.at<uchar>(i, j)) {
						m = 0;
					}
				}
				if (j - 1 >= 0 && i - 1 >= 0) {
					if (modul.at<uchar>(i - 1, j - 1) > modul.at<uchar>(i, j)) {
						m = 0;
					}
				}
				break;
			default:
				break;
			}
			subtiere.at<uchar>(i, j) = m;
		}
	}


	for (int i = k; i < subtiere.rows; i++) {
		subtiere.at<uchar>(i, 0) = 0;
		subtiere.at<uchar>(i, subtiere.cols - 1) = 0;
	}

	for (int i = k; i < subtiere.cols; i++) {
		subtiere.at<uchar>(0, i) = 0;
		subtiere.at<uchar>(subtiere.rows - 1, i) = 0;
	}


	int hist[256] = { 0 };

	for (int i = k; i < subtiere.rows; i++) {
		for (int j = k; j < subtiere.cols; j++) {
			hist[subtiere.at<uchar>(i, j)]++;
		}
	}

	int nrNonMuchie = (1 - 0.1)*(subtiere.rows*subtiere.cols - hist[0]);
	int sumahist = 0, prag = 0;
	for (int i = 1; i < 256; i++) {
		sumahist += hist[i];
		if (sumahist > nrNonMuchie) {
			prag = i;
			break;
		}
	}

	int pH = prag;
	int pL = 0.4*pH;

	modul = subtiere.clone();

	for (int i = 0; i < modul.rows; i++) {
		for (int j = 0; j < modul.cols; j++) {
			if (modul.at<uchar>(i, j) < pL) {
				modul.at<uchar>(i, j) = 0;
			}
			else
				if (modul.at<uchar>(i, j) > pH) {
					modul.at<uchar>(i, j) = STRONG;
				}
				else
				{
					modul.at<uchar>(i, j) = WEAK;
				}
		}
	}

	int difX[8] = { 0, -1, -1, -1,  0,  1, 1, 1 };
	int difY[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

	Mat visited(modul.rows, modul.cols, CV_8UC1, Scalar(0));
	queue <Point> que;
	for (int i = 0; i < modul.rows; i++) {
		for (int j = 0; j < modul.cols; j++) {
			if (modul.at<uchar>(i, j) == STRONG && visited.at<uchar>(i, j) == 0) {
				que.push(Point(j, i));
				visited.at<uchar>(i, j) = 1;
			}
			while (!que.empty()) {
				Point oldest = que.front();
				int jj = oldest.x;
				int ii = oldest.y;
				que.pop();
				for (int p = 0; p < 8; p++) {
					if (ii + difX[p] >= 0 && ii + difX[p] < modul.rows && jj + difY[p] >= 0 && jj + difY[p] < modul.cols) {
						if (modul.at<uchar>(ii + difX[p], jj + difY[p]) == WEAK) {
							modul.at<uchar>(ii + difX[p], jj + difY[p]) = STRONG;
							que.push(Point(jj + difY[p], ii + difX[p]));
							visited.at<uchar>(ii + difX[p], jj + difY[p]) = 1;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < modul.rows; i++) {
		for (int j = 0; j < modul.cols; j++) {
			if (modul.at<uchar>(i, j) == WEAK) {
				modul.at<uchar>(i, j) = 0;
			}
		}
	}


	return modul;
}

void chessBoardDetection()
{
	char path[MAX_PATH];
	int thresh = 200;

	Size sizeImg(640, 640);
	PerspectiveProjection perspectiveProjectionUtil;

	while(openFileDlg(path))
	{
		Mat gray, imgResized, edges, grayWithoutNoises;
		Mat source = imread(path, IMREAD_COLOR);

		// resize image
		resizeImg(source, imgResized, 1024, true);
		// Mat imgResized2 = removeRedColor(imgResized, 50, 105);

		// convert to gray scale
		cvtColor(imgResized, gray, COLOR_BGR2GRAY);

		// remove noises
		//grayWithoutNoises = filterGaussianNoises(gray, 5);

		// apply edge filter
		//Canny(grayWithoutNoises, edges, 100, thresh);
		grayWithoutNoises = cannyBoti(gray);
		/*
		// Create a vector to store lines of the image
		std::vector<Vec4i> lines;
		Point tl, tr, bl, br;
		HoughLinesP(edges, lines, 1, CV_PI / 180, thresh, 10, 250);
		std::vector<Point> points;
		for (auto l: lines)
		{
			Point p1(l[0], l[1]);
			Point p2(l[2], l[3]);

			if (isInside(imgResized, p1.y, p1.x)) {
				points.push_back(p1);
			}
			if (isInside(imgResized, p2.y, p2.x)) {
				points.push_back(p2);
			}
		}

		// destination points for OPENCV Method
		Point2f destinationPoints1[4] = { Point2f(0,0), Point2f(sizeImg.height, 0), Point2f(0, sizeImg.width) ,Point2f(sizeImg.height, sizeImg.width) };
		// destination points for our implementation
		Point2f destinationPoints2[4] = { Point2f(0,0), Point2f(0, sizeImg.width), Point2f(sizeImg.height, 0) ,Point2f(sizeImg.height, sizeImg.width) };

		// compute border points of the chess board
		perspectiveProjectionUtil.getBorderBoxes(points, tl, tr, bl, br);
		Point2f sourcePoints[4] = { tl, tr, bl, br };

		// projection with opencv implementation
		Mat projectionMatrix1 = perspectiveProjectionUtil.getPerspectiveTransform(sourcePoints, destinationPoints1, OPEN_CV);
		Mat imgProjected1 = perspectiveProjectionUtil.perspectiveProjection(projectionMatrix1, imgResized, OPEN_CV, sizeImg);

		// projection with out implementation
		Mat projectionMatrix2 = perspectiveProjectionUtil.getPerspectiveTransform(sourcePoints, destinationPoints2, PIZZA);
		Mat imgProjected2 = perspectiveProjectionUtil.perspectiveProjection(projectionMatrix2, imgResized, PIZZA, sizeImg);


		// Start drawing Margins
		Mat boardMargins(imgResized.rows, imgResized.cols, CV_8UC1, Scalar(255));
		circle(boardMargins, bl, 2, Scalar(0), 2, 8, 0);
		circle(boardMargins, br, 2, Scalar(40), 2, 8, 0);
		circle(boardMargins, tl, 2, Scalar(0), 2, 8, 0);
		circle(boardMargins, tr, 2, Scalar(120), 2, 8, 0);
		imshow("Board margins", boardMargins);
		// End drawing margins
		*/
		imshow("hi", grayWithoutNoises);
		//imshow("Gray", gray);
		//imshow("Edges", edges);
		//imshow("Image resized", imgResized);
		//imshow("Img projection1-OPENCV", imgProjected1);
		//imshow("Img projection2-PIZZA", imgProjected2);
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
			for (int j = 0 ; j < imgResized.cols; j++)
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

	while(openFileDlg(path))
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
		}
	}
	while (op != 0);

	return 0;
}
