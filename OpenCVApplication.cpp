// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>
#include "time.h"
#include <fstream>
#include <iostream>
#include <experimental/filesystem>
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


	// std::cout <<"Source points \n";
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


void detectLines()
{
	char path[MAX_PATH];
	int thresh = 200;
	int size = 640;


	while(openFileDlg(path))
	{
		Mat gray, imgResized;
		Mat source = imread(path, IMREAD_COLOR);
		resizeImg(source, imgResized, 1024, true);
		std::cout << imgResized.at<Vec3b>(0, 0);
		Mat imgResized2 = removeRedColor(imgResized, 50, 105);
		cvtColor(imgResized2,gray, COLOR_BGR2GRAY);
		Mat grayWithotNoises = filterGaussianNoises(gray, 5);

		Mat edges;
		Canny(grayWithotNoises, edges, 100, thresh);
		
		// Create a vector to store lines of the image
		std::vector<Vec4i> lines;

		Point tl,tr, bl, br;
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

		
		getBorderBoxes(points, tl, tr, bl, br);

		Mat newImg(imgResized.rows, imgResized.cols, CV_8UC1, Scalar(255));
		Point2f sourcePoints[4] = { tl, tr, bl, br };

		Mat transform = transform_perspective(imgResized, sourcePoints, size);
		Mat t = perspectiveProjection(imgResized, sourcePoints, size);

		circle(newImg, bl, 2, Scalar(0), 2, 8, 0);
		circle(newImg, br, 2, Scalar(40), 2, 8, 0);
		circle(newImg, tl, 2, Scalar(80), 2, 8, 0);
		circle(newImg, tr, 2, Scalar(120), 2, 8, 0);

		std::cout << tl << " " << tr << " " << bl << " " << br << std::endl;
		// // Apply Hough Transform

		// // Draw lines on the image
		// for (size_t i = 0; i < lines.size(); i++) {
		// 	Vec4i l = lines[i];
		// 	circle(imgResized, Point(l[0], l[1]), 2, Scalar(0),2, 8, 0);
		// 	circle(imgResized, Point(l[2], l[3]), 2, Scalar(0),2, 8, 0);
		// 	// line(imgResized, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 3, LINE_AA);
		// }
		// Show result image
		imshow("Gray", gray);
		imshow("Edges", edges);
		imshow("Margin", newImg);
		imshow("Result Image", imgResized);
		imshow("Transform", transform);
		imshow("Transform2", t);
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
			detectLines();
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
