#include "stdafx.h"

#include "common.h"
#include <CommDlg.h>
#include <ShlObj.h>
#include "time.h"
#include "limits"
#include <fstream>


FileGetter::FileGetter(char* folderin, char* ext)
{
	strcpy(folder, folderin);
	char folderstar[MAX_PATH];
	if (!ext) strcpy(ext, "*");
	sprintf(folderstar, "%s\\*.%s", folder, ext);
	hfind = FindFirstFileA(folderstar, &found);
	hasFiles = !(hfind == INVALID_HANDLE_VALUE);
	first = 1;
	//skip .
	//FindNextFileA(hfind,&found);		
}

int FileGetter::getNextFile(char* fname)
{
	if (!hasFiles)
		return 0;
	//skips .. when called for the first time
	if (first)
	{
		strcpy(fname, found.cFileName);
		first = 0;
		return 1;
	}
	else
	{
		chk = FindNextFileA(hfind, &found);
		if (chk)
			strcpy(fname, found.cFileName);
		return chk;
	}
}

int FileGetter::getNextAbsFile(char* fname)
{
	if (!hasFiles)
		return 0;
	//skips .. when called for the first time
	if (first)
	{
		sprintf(fname, "%s\\%s", folder, found.cFileName);
		first = 0;
		return 1;
	}
	else
	{
		chk = FindNextFileA(hfind, &found);
		if (chk)
			sprintf(fname, "%s\\%s", folder, found.cFileName);
		return chk;
	}
}

char* FileGetter::getFoundFileName()
{
	if (!hasFiles)
		return 0;
	return found.cFileName;
}


int openFileDlg(char* fname)
{
	char* filter = "All Files (*.*)\0*.*\0";
	HWND owner = NULL;
	OPENFILENAME ofn;
	char fileName[MAX_PATH];
	strcpy(fileName, "");
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.hwndOwner = owner;
	ofn.lpstrFilter = filter;
	ofn.lpstrFile = fileName;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
	ofn.lpstrDefExt = "";
	GetOpenFileName(&ofn);
	strcpy(fname, ofn.lpstrFile);
	return strcmp(fname, "");
}

int openFolderDlg(char* folderName)
{
	BROWSEINFO bi;
	ZeroMemory(&bi, sizeof(bi));
	SHGetPathFromIDList(SHBrowseForFolder(&bi), folderName);
	return strcmp(folderName, "");
}

void resizeImg(Mat src, Mat& dst, int maxSize, bool interpolate)
{
	double ratio = 1;
	double w = src.cols;
	double h = src.rows;
	if (w > h)
		ratio = w / (double)maxSize;
	else
		ratio = h / (double)maxSize;
	int nw = (int)(w / ratio);
	int nh = (int)(h / ratio);
	Size sz(nw, nh);
	if (interpolate)
		resize(src, dst, sz);
	else
		resize(src, dst, sz, 0, 0, INTER_NEAREST);
}

int getValidColor(int color)
{
	return (color > 255) ? 255 : ((color < 0) ? 0 : color);
}


Mat grayScale(Mat img)
{
	Mat image(img.rows, img.cols, CV_8UC1);

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			Vec3b colors = img.at<Vec3b>(row, col);

			image.at<uchar>(row, col) = (colors[0] + colors[1] + colors[2]) / 3;
		}
	}

	return image;
}

std::vector<Mat> getRGBChannels(Mat img)
{
	Mat g(img.rows, img.cols, CV_8UC1);
	Mat r(img.rows, img.cols, CV_8UC1);
	Mat b(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b color = img.at<Vec3b>(i, j);
			g.at<uchar>(i, j) = color[1];
			b.at<uchar>(i, j) = color[0];
			r.at<uchar>(i, j) = color[2];
		}
	}

	std::vector<Mat> result;
	result.push_back(r);
	result.push_back(g);
	result.push_back(b);

	return result;
}

Mat blackAndWhite(Mat image, int prague)
{
	Mat blackAndWhite(image.rows, image.cols, CV_8UC1);

	for (int row = 0; row < image.rows; row++)
	{
		for (int col = 0; col < image.cols; col++)
		{
			uchar color = image.at<uchar>(row, col);

			blackAndWhite.at<uchar>(row, col) = (color < prague) ? 0 : 255;
		}
	}

	return blackAndWhite;
}

Vec3b rgbColorToHSVColor(Vec3b color)
{
	Vec3f colorNormalized;
	colorNormalized[2] = (float)color[2] / 255.0;
	colorNormalized[1] = (float)color[1] / 255.0;
	colorNormalized[0] = (float)color[0] / 255.0;

	uchar maxColor = 0; // 0 -> blue 1 -> green 2 -> red
	float maxValue = -1;
	float minValue = 2;

	for (int i = 0; i < 3; i++)
	{
		if (colorNormalized[i] > maxValue)
		{
			maxValue = colorNormalized[i];
			maxColor = i;
		}

		if (colorNormalized[i] < minValue)
		{
			minValue = colorNormalized[i];
		}
	}

	float c = maxValue - minValue;
	float s;
	float h;

	s = (maxValue != 0.0) ? c / maxValue : 0;

	if (c != 0.0)
	{
		switch (maxColor)
		{
		case 0: // blue
			h = 240.0 + 60.0 * (colorNormalized[2] - colorNormalized[1]) / c;
			break;
		case 1:
			h = 120.0 + 60.0 * (colorNormalized[0] - colorNormalized[2]) / c;
			break;
		case 2:
			h = 60 * (colorNormalized[1] - colorNormalized[0]) / c;
			break;
		}
	}
	else
	{
		h = 0.0;
	}

	if (h < 0)
	{
		h += 360;
	}

	Vec3b colorHSV(h * 255 / 360, s * 255, maxValue * 255);
	return colorHSV;
}

Vec3b hsvColorToRGBColor(Vec3b hsvColor)
{
	int h = hsvColor[0] * 360 / 255;
	float s = hsvColor[1] / 255.0;
	float v = hsvColor[2] / 255.0;

	float c = v * s;

	float m = v - c;
	float x = c * (1 - std::abs((h / 60) % 2 - 1));
	Vec3f bgr;

	switch (static_cast<int>(h / 60))
	{
	case 0:
		bgr = Vec3f(0, x, c);
		break;
	case 1:
		bgr = Vec3f(0, c, x);
		break;
	case 2:
		bgr = Vec3f(x, c, 0);
		break;
	case 3:
		bgr = Vec3f(c, x, 0);
		break;
	case 4:
		bgr = Vec3f(c, 0, x);
		break;
	case 5:
		bgr = Vec3f(x, 0, c);
		break;
	}

	return Vec3b((bgr[0] + m) * 255, (bgr[1] + m) * 255, (bgr[2] + m) * 255);
}

std::vector<Mat> rgbToHsv(Mat img)
{
	Mat h(img.rows, img.cols, CV_8UC1);
	Mat s(img.rows, img.cols, CV_8UC1);
	Mat v(img.rows, img.cols, CV_8UC1);
	Mat hsv(img.rows, img.cols, CV_8UC3);

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			Vec3b hsvColor = rgbColorToHSVColor(img.at<Vec3b>(row, col));

			h.at<uchar>(row, col) = hsvColor[0];
			s.at<uchar>(row, col) = hsvColor[1];
			v.at<uchar>(row, col) = hsvColor[2];

			hsv.at<Vec3b>(row, col) = hsvColor;
		}
	}


	std::vector<Mat> result;

	result.push_back(h);
	result.push_back(s);
	result.push_back(v);
	result.push_back(hsv);

	return result;
}


bool isInside(Mat img, int row, int col)
{
	return (row >= 0 && row < img.rows) && (col >= 0 && col < img.cols);
}

/**
 * img is gray scaled
 */
int* computeHistogram(Mat img, int accumulator)
{
	int* histogram = (int*)calloc(accumulator, sizeof(int));

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			uchar color = img.at<uchar>(row, col);
			histogram[color * accumulator / 256]++;
		}
	}

	return histogram;
}

Mat removeRedColor(Mat source, int colorMin, int colorMax)
{
	Mat dest = source.clone();
	for (int i = 0; i < dest.rows; i++)
	{
		for (int j = 0; j < dest.cols; j++)
		{

			Vec3b color = dest.at<Vec3b>(i, j);

			if (color[2] > 55 && color[2] < 120)
			{
				dest.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
		}
	}

	return dest;
}

float* computeFDP(Mat img)
{
	float* fdp = (float*)calloc(256, sizeof(float));

	int m = img.rows * img.cols;
	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			uchar color = img.at<uchar>(row, col);
			fdp[color]++;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		fdp[i] = fdp[i] / m;
	}

	return fdp;
}

void showHistogramI(const std::string& name, int* hist, int hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));

	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;
	for (int x = 0; x < hist_cols; x++)
	{
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins
		// colored in magenta
	}
	imshow(name, imgHist);
}

std::vector<int> computeMaxLocals(Mat img, int wh, float th)
{
	float* fdp = computeFDP(img);

	std::vector<int> maxim;
	maxim.push_back(0);
	for (int k = wh; k < 256 - wh; k++)
	{
		float average = 0.0f;
		float maxValue = -0.0f;
		for (int i = k - wh; i <= k + wh; i++)
		{
			average += fdp[i];
			if (fdp[i] > maxValue)
			{
				maxValue = fdp[i];
			}
		}

		average /= (2 * wh + 1);

		if (fdp[k] > th + average && fdp[k] == maxValue)
		{
			maxim.push_back(k);
		}
	}

	maxim.push_back(255);
	return maxim;
}

int getClosserMaxim(int color, std::vector<int> maxLocals)
{
	for (int i = 0; i < maxLocals.size() - 1; i++)
	{
		if (color > maxLocals[i] && color <= maxLocals[i + 1])
		{
			if (std::abs(color - maxLocals[i]) > std::abs(color - maxLocals[i + 1]))
			{
				return maxLocals[i + 1];
			}
			{
				return maxLocals[i];
			}
		}
	}

	return color;
}

Mat reduceGrayLevels(Mat image, REDUCE_GRAY_LEVEL_ALGORITHM algorithm)
{
	std::vector<int> maxLocals = computeMaxLocals(image);

	for (int i = 0; i < maxLocals.size(); i++)
	{
		std::cout << maxLocals[i] << " ";
	}
	if (algorithm == MULTIPLE_PRAGS)
	{
		Mat newImg(image.rows, image.cols, CV_8UC1);

		for (int row = 0; row < newImg.rows; row++)
		{
			for (int col = 0; col < newImg.cols; col++)
			{
				uchar color = image.at<uchar>(row, col);

				newImg.at<uchar>(row, col) = getClosserMaxim(color, maxLocals);
			}
		}

		return newImg;
	}
	else if (algorithm == FLOYD_STEINBERG)
	{
		Mat img = image.clone();

		for (int row = 1; row < img.rows - 1; row++)
		{
			for (int col = 1; col < img.cols - 1; col++)
			{
				uchar oldPixel = img.at<uchar>(row, col);
				uchar newPixel = getClosserMaxim(oldPixel, maxLocals);

				int error = oldPixel - newPixel;
				img.at<uchar>(row, col) = newPixel;

				img.at<uchar>(row + 1, col) = getValidColor(
					img.at<uchar>(row + 1, col) + static_cast<int>(5 * error / 16));
				img.at<uchar>(row + 1, col - 1) = getValidColor(
					img.at<uchar>(row + 1, col - 1) + static_cast<int>(3 * error / 16));
				img.at<uchar>(row, col + 1) = getValidColor(
					img.at<uchar>(row, col + 1) + static_cast<int>(7 * error / 16));
				img.at<uchar>(row + 1, col + 1) = getValidColor(
					img.at<uchar>(row + 1, col + 1) + static_cast<int>(error / 16));
			}
		}

		return img;
	}
}

Mat reduceGrayLevelOnHChannel(Mat img)
{
	std::vector<Mat> hsvChannels = rgbToHsv(img);
	Mat image(img.rows, img.cols, CV_8UC3);

	Mat hsvReduced = reduceGrayLevels(hsvChannels[0], MULTIPLE_PRAGS);

	for (int row = 0; row < image.rows; row++)
	{
		for (int col = 0; col < image.cols; col++)
		{
			Vec3b hsvColor(hsvReduced.at<uchar>(row, col), hsvChannels[1].at<uchar>(row, col),
			               hsvChannels[2].at<uchar>(row, col));

			Vec3b rbgColor = hsvColorToRGBColor(hsvColor);
			image.at<Vec3b>(row, col) = rbgColor;
		}
	}

	return image;
}

bool isOnEdge(Mat img, int row, int col)
{
	int dx[8] = {-1, 0, 1, 0, -1, -1, 1, 1};
	int dy[8] = {0, -1, 0, 1, -1, 1, -1, 1};

	bool is = false;

	for (int i = 0; i < 8; i++)
	{
		if (isInside(img, row + dx[i], col + dy[i]))
		{
			is |= (img.at<Vec3b>(row, col) != img.at<Vec3b>(row + dx[i], col + dy[i]));
		}
	}

	return is;
}

bool isOnEdge(Mat img, int row, int col, int s)
{
	int dx[8] = {-1, 0, 1, 0, -1, -1, 1, 1};
	int dy[8] = {0, -1, 0, 1, -1, 1, -1, 1};

	bool is = false;

	for (int i = 0; i < 8; i++)
	{
		if (isInside(img, row + dx[i], col + dy[i]))
		{
			is |= (img.at<uchar>(row, col) != img.at<uchar>(row + dx[i], col + dy[i]));
		}
	}

	return is;
}

ObjectProperties computeObjectProperties(Mat img, Vec3b color)
{
	ObjectProperties properties(img.rows, img.cols, 1);

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			if (img.at<Vec3b>(row, col) == color)
			{
				properties.add(row, col, isOnEdge(img, row, col));
			}
		}
	}

	properties.computeFinalProperties();

	return properties;
}

ObjectProperties computeObjectProperties(Mat img, uchar color)
{
	ObjectProperties properties(img.rows, img.cols, 1);

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			if (img.at<uchar>(row, col) == color)
			{
				properties.add(row, col, isOnEdge(img, row, col, 1));
			}
		}
	}

	return properties;
}

std::vector<ObjectProperties> computeObjectsProperties(Mat img)
{
	std::vector<ObjectProperties> result;

	Vec3b color;
	std::map<int, ObjectProperties> objects;
	std::map<int, Vec3b> idToColor;

	int count = 0;
	int id;
	bool exists;

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			color = img.at<Vec3b>(row, col);

			exists = false;
			for (auto kv : idToColor)
			{
				if (kv.second == color)
				{
					id = kv.first;
					exists = true;
				}
			}


			if (!exists)
			{
				objects[count] = ObjectProperties(img.rows, img.cols, count);
				idToColor[count] = color;
				id = count;
				count++;
			}

			objects[id].add(row, col, isOnEdge(img, row, col));
		}
	}

	for (auto kv : objects)
	{
		kv.second.computeFinalProperties();
		result.push_back(kv.second);
	}

	return result;
}

Point getPointAfterAngle(Point p, int length, float angle)
{
	Point point;

	point.x = (int)round(p.x + length * cos(angle));
	point.y = (int)round(p.y + length * sin(angle));

	return point;
}


Mat imageLabeling(Mat img)
{
	srand(time(NULL));

	int label = 0;
	Mat labels(img.rows, img.cols, CV_8UC1, Scalar(0));
	//Mat imgColored(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

	int dx[8] = {-1, 0, 1, 0, -1, -1, 1, 1};
	int dy[8] = {0, -1, 0, 1, -1, 1, -1, 1};


	//std::vector<Vec3b> colors;

	std::queue<cv::Point> queue;
	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			if (img.at<uchar>(row, col) == 0 && labels.at<uchar>(row, col) == 0)
			{
				label++;
				// Vec3b color(rand() % 255, rand() % 255, rand() % 255);
				// colors.push_back(color);

				queue.push(Point(row, col));

				while (!queue.empty())
				{
					Point p = queue.front();
					queue.pop();

					for (int i = 0; i < 8; i++)
					{
						int r = p.x + dx[i];
						int c = p.y + dy[i];

						if (isInside(labels, r, c))
						{
							if (labels.at<uchar>(r, c) == 0 && img.at<uchar>(r, c) == 0)
							{
								queue.push(Point(r, c));
								labels.at<uchar>(r, c) = label;
								//imgColored.at<Vec3b>(r, c) = colors[label - 1];
							}
						}
					}
				}
			}
		}
	}

	return labels;
}

std::vector<Mat> imageLabelingAg2(Mat img)
{
	int label = 0;
	Mat labels(img.rows, img.cols, CV_8UC1, Scalar(0));
	std::vector<std::vector<int>> edges(img.rows, std::vector<int>(img.cols));

	int dx[4] = {-1, -1, -1, 0};
	int dy[4] = {-1, 0, 1, -1};

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			if (img.at<uchar>(row, col) == 0 && labels.at<uchar>(row, col) == 0)
			{
				std::vector<int> l;
				for (int i = 0; i < 4; i++)
				{
					int r = row + dx[i];
					int c = col + dy[i];

					if (labels.at<uchar>(r, c) > 0)
					{
						l.push_back(labels.at<uchar>(r, c));
					}
				}

				if (l.size() == 0)
				{
					label++;
					labels.at<uchar>(row, col) = label;
				}
				else
				{
					int minValue = label + 2;
					for (auto x : l)
					{
						if (x < minValue)
						{
							minValue = x;
						}
					}

					labels.at<uchar>(row, col) = minValue;
					for (auto x : l)
					{
						if (x != minValue)
						{
							edges[minValue].push_back(x);
							edges[x].push_back(minValue);
						}
					}
				}
			}
		}
	}


	Mat labelsInt = labels.clone();

	int newLabel = 0;
	int* newLabels = static_cast<int*>(calloc(label + 1, sizeof(int)));

	for (int i = 1; i <= label; i++)
	{
		if (newLabels[i] == 0)
		{
			newLabel++;
			std::queue<int> queue;

			newLabels[i] = newLabel;
			queue.push(i);

			while (!queue.empty())
			{
				int x = queue.front();
				queue.pop();

				for (auto y : edges[x])
				{
					if (newLabels[y] == 0)
					{
						newLabels[y] = newLabel;
						queue.push(y);
					}
				}
			}
		}
	}

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			if (img.at<uchar>(row, col) == 0)
			{
				labels.at<uchar>(row, col) = newLabels[labels.at<uchar>(row, col)];
			}
		}
	}

	std::vector<Mat> res;
	res.push_back(labels);
	res.push_back(labelsInt);

	return res;
}


Mat colorImageLabeled(Mat labeledImage)
{
	std::map<int, Vec3b> colors;

	Mat img(labeledImage.rows, labeledImage.cols, CV_8UC3, Scalar(255, 255, 255));

	for (int row = 0; row < labeledImage.rows; row++)
	{
		for (int col = 0; col < labeledImage.cols; col++)
		{
			uchar label = labeledImage.at<uchar>(row, col);
			if (label > 0)
			{
				if (colors.find(label) == colors.end())
				{
					Vec3b color(rand() % 255, rand() % 255, rand() % 255);
					colors[label] = color;
				}

				img.at<Vec3b>(row, col) = colors[label];
			}
		}
	}

	return img;
}

/**
 * Img is black-white
* Dir = 7 || Dir = 4
*/

Mat traceContour(Mat img, int n)
{
	Mat newImg(img.rows, img.cols, CV_8UC1, Scalar(255));

	if (n != 8 && n != 4)
	{
		return newImg;
	}

	std::vector<Point> contour;

	std::vector<int> chainedCode;
	std::vector<int> derivationCode;

	Point first;
	// find first element
	bool finded = false;
	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			if (img.at<uchar>(row, col) == 0)
			{
				first.x = row;
				first.y = col;
				finded = true;
				break;
			}
		}

		if (finded)
		{
			break;
		}
	}


	int dir = 0;
	int dx[8] = {0, -1, -1, -1, 0, 1, 1, 1};
	int dy[8] = {1, 1, 0, -1, -1, -1, 0, 1};

	dir = (n == 8) ? 7 : 0;


	if (finded)
	{
		bool stop = false;
		contour.push_back(first);
		Point current(contour[0].x, contour[0].y);
		int dirAux = 0;
		int r;
		int c;

		while (!stop)
		{
			if (n == 8)
			{
				dirAux = ((dir) % 2 == 0) ? (dir + 7) % 8 : (dir + 6) % 8;;
			}
			else
			{
				dirAux = (dir + 3) % 4;
			}

			for (int repeat = 0; repeat < n; repeat++)
			{
				int d = (dirAux + repeat) % n;

				r = current.x + dx[(n == 4) ? d * 2 : d];
				c = current.y + dy[(n == 4) ? d * 2 : d];

				if (isInside(img, r, c) && img.at<uchar>(r, c) == 0)
				{
					if (d >= dir)
					{
						derivationCode.push_back(d - dir);
					}
					else
					{
						derivationCode.push_back(n - dir + d);
					}


					chainedCode.push_back(d);
					dir = d;
					current.x = r;
					current.y = c;
					contour.push_back(current);
					break;
				}
			}

			if (current == first)
			{
				break;
			}
		}
	}

	for (int i = 0; i < contour.size(); i++)
	{
		newImg.at<uchar>(contour[i].x, contour[i].y) = 0;
	}

	std::cout << "Initial point" << first << std::endl << "Chained code: ";
	for (int i = 0; i < chainedCode.size(); i++)
	{
		std::cout << chainedCode[i] << " ";
	}

	std::cout << std::endl << "Derivation code: ";
	for (int i = 0; i < derivationCode.size(); i++)
	{
		std::cout << derivationCode[i] << " ";
	}
	return newImg;
}

Mat reconstructContour(Point start, std::vector<int> code)
{
	int dx[8] = {0, -1, -1, -1, 0, 1, 1, 1};
	int dy[8] = {1, 1, 0, -1, -1, -1, 0, 1};


	Mat img(255, 650, CV_8UC1, Scalar(255));

	Point current = start;
	img.at<uchar>(current.x, current.y) = 0;
	for (int i = 0; i < code.size(); i++)
	{
		current.x += dx[code[i]];
		current.y += dy[code[i]];

		img.at<uchar>(current.x, current.y) = 0;
	}

	return img;
}


Mat dilatation(Mat img, int dilationSize)
{
	Mat destination(img.rows, img.cols, CV_8UC1, Scalar(255));

	int dx[8] = {0, -1, -1, -1, 0, 1, 1, 1};
	int dy[8] = {1, 1, 0, -1, -1, -1, 0, 1};

	for (int row = 1; row < img.rows - 1; row++)
	{
		for (int col = 1; col < img.cols - 1; col++)
		{
			if (img.at<uchar>(row, col) == 0)
			{
				bool valid = false;
				for (int i = 0; i < 8; i++)
				{
					valid |= (img.at<uchar>(row + dx[i], col + dy[i]) == 0);
				}

				if (valid)
				{
					for (int j = 1; j <= dilationSize; j++)
					{
						for (int i = 0; i < 8; i++)
						{
							destination.at<uchar>(row + dx[i] * j, col + dy[i] * j) = 0;
						}
					}
				}
			}
		}
	}

	return destination;
}


Mat erosion(Mat img, int erosionSize)
{
	Mat destination(img.rows, img.cols, CV_8UC1, Scalar(255));

	int dx[8] = {0, -1, -1, -1, 0, 1, 1, 1};
	int dy[8] = {1, 1, 0, -1, -1, -1, 0, 1};

	for (int row = 1; row < img.rows - 1; row++)
	{
		for (int col = 1; col < img.cols - 1; col++)
		{
			if (img.at<uchar>(row, col) == 0)
			{
				bool valid = true;

				for (int j = 1; j <= erosionSize; j++)
				{
					for (int i = 0; i < 8; i++)
					{
						valid &= (img.at<uchar>(row + dx[i] * j, col + dy[i] * j) == 0);
					}
				}

				if (valid)
				{
					destination.at<uchar>(row, col) = 0;
				}
			}
		}
	}

	return destination;
}

Mat opening(Mat img, int size)
{
	return dilatation(erosion(img, size), size);
}

Mat closure(Mat img, int size)
{
	return erosion(dilatation(img, size), size);
}

std::vector<Point> extractContour(Mat img)
{
	Mat imgEroded = erosion(img, 1);

	std::vector<Point> result;

	for (int row = 0; row < img.rows; ++row)
	{
		for (int col = 0; col < img.cols; ++col)
		{
			if (img.at<uchar>(row, col) != imgEroded.at<uchar>(row, col))
			{
				Point p(row, col);
				result.push_back(p);
			}
		}
	}

	return result;
}

Mat regionFill(Mat img, Point startPoint)
{
	// std::vector<Point> contour = extractContour(img);
	//
	// std::map<int, std::vector<int>> inside;
	// int x = 0;
	// for (Point p: contour)
	// {
	// 	inside[p.x].push_back(p.y);
	// 	if (inside[p.x].size() == 2 && (std::abs(p.y - inside[p.x].at(0)) > 50))
	// 	{
	// 		x = p.x;
	// 		break;
	// 	}
	// }
	//
	// startPoint.x = x;
	//
	// if (inside[x].at(1) < inside[x].at(0))
	// {
	// 	int aux = inside[x].at(1);
	// 	inside[x].at(1) = inside[x].at(0);
	// 	inside[x].at(0) = aux;
	// }
	// startPoint.y = rand() % inside[x].at(1) + inside[x].at(0);
	//
	// std::cout << startPoint;

	Mat newImg = img.clone();
	int dx[4] = {0, 0, -1, 1};
	int dy[4] = {1, -1, 0, 0};

	std::vector<Point> points;
	points.push_back(startPoint);

	newImg.at<uchar>(startPoint.x, startPoint.y) = 0;
	int s = 0;
	while (!points.empty())
	{
		std::vector<Point> newPoints;
		for (int i = 0; i < points.size(); i++)
		{
			for (int j = 0; j < 4; j++)
			{
				int x = points[i].x + dx[j];
				int y = points[i].y + dy[j];

				if (isInside(img, x, y) && img.at<uchar>(x, y) == 255 && newImg.at<uchar>(x, y) != 0)
				{
					Point p(x, y);
					newPoints.push_back(p);
					newImg.at<uchar>(x, y) = 0;
				}
			}
		}

		points.clear();
		for (Point p : newPoints)
		{
			points.push_back(p);
		}
	}

	return newImg;
}


double getAverage(Mat img)
{
	int sum = 0;

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			sum += img.at<uchar>(row, col);
		}
	}

	return static_cast<double>(sum) / (img.rows * img.cols);
}


double computeDeviation(Mat img)
{
	int average = static_cast<int>(getAverage(img));
	int sum = 0;

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			sum += std::pow(img.at<uchar>(row, col) - average, 2);
		}
	}

	return std::sqrt(sum / (img.rows * img.cols));
}

int* computeFDPC(Mat img)
{
	float* fdpc = (float*)calloc(256, sizeof(float));
	int* f = (int*)calloc(256, sizeof(int));

	float* fdp = computeFDP(img);

	fdpc[0] = fdp[0];
	f[0] = fdpc[0] * 255;
	for (int i = 1; i < 256; i++)
	{
		fdpc[i] = fdpc[i - 1] + fdp[i];
		f[i] = 255 * fdpc[i];
	}


	return f;
}

Mat imgBinarization(Mat img, float error)
{
	int* histo = computeHistogram(img);

	int maxIntensity = -1;
	int maxValue = INT_MIN;

	int minIntensity = 256;
	int minValue = INT_MAX;

	for (int i = 0; i < 256; i++)
	{
		if (histo[i] > maxValue)
		{
			maxValue = histo[i];
			maxIntensity = i;
		}

		if (histo[i] < minValue)
		{
			minValue = histo[i];
			minIntensity = i;
		}
	}

	float t0 = 0.0f;
	float t1 = 100.0f;

	float ng1, ng2;
	int n1 = 0;
	int n2 = 0;

	while (true)
	{
		n1 = 0;
		n2 = 0;
		ng1 = 0;
		ng2 = 0;
		for (int i = minIntensity; i <= (int)t1; i++)
		{
			n1 += histo[i];
			ng1 += i * histo[i];
		}

		for (int i = (int)t1; i < maxIntensity; i++)
		{
			n2 += histo[i];
			ng2 += i * histo[i];
		}

		if (n1 != 0 && n2 != 0)
		{
			ng1 /= n1;
			ng2 /= n2;
		}

		t0 = t1;
		t1 = (ng1 + ng2) / 2;

		if (std::abs(t0 - t1) < error)
		{
			break;
		}
	}

	return blackAndWhite(img, static_cast<int>(t1));
}


Mat negativeImg(Mat img)
{
	Mat newImg(img.rows, img.cols, CV_8UC1);
	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			newImg.at<uchar>(row, col) = 255 - img.at<uchar>(row, col);
		}
	}

	return newImg;
}

Mat changeContrast(Mat img, int goMin, int goMax)
{
	int* histo = computeHistogram(img);

	int giMax = -1;
	int maxValue = INT_MIN;

	int giMin = 256;
	int minValue = INT_MAX;

	for (int i = 0; i < 256; i++)
	{
		if (histo[i] > maxValue)
		{
			maxValue = histo[i];
			giMax = i;
		}

		if (histo[i] < minValue)
		{
			minValue = histo[i];
			giMin = i;
		}
	}

	float rap = (goMax - goMin) / static_cast<float>(giMax - giMin);

	Mat newImg(img.rows, img.cols, CV_8UC1);

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			int value = goMin + static_cast<int>((img.at<uchar>(row, col) - giMin) * rap);
			newImg.at<uchar>(row, col) = getValidColor(value);
		}
	}

	return newImg;
}

Mat gammaCorrection(Mat img, float gamma)
{
	Mat newImg(img.rows, img.cols, CV_8UC1);

	int l = 255;

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			newImg.at<uchar>(row, col) = getValidColor(
				static_cast<int>(l * std::pow(static_cast<float>(img.at<uchar>(row, col)) / l, gamma)));
		}
	}

	return newImg;
}


Mat changeLuminosity(Mat img, int luminosity)
{
	Mat newImg(img.rows, img.cols, CV_8UC1);

	int l = 255;

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			newImg.at<uchar>(row, col) = getValidColor(img.at<uchar>(row, col) + luminosity);
		}
	}

	return newImg;
}

Mat equalizeHistogram(Mat img)
{
	int* fdpc = computeFDPC(img);

	Mat newImg(img.rows, img.cols, CV_8UC1);

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			newImg.at<uchar>(row, col) = getValidColor(fdpc[img.at<uchar>(row, col)]);
		}
	}

	return newImg;
}

Mat convolutionFiltruJos(Mat img, int nucleu[3][3], int k)
{
	Mat d(img.rows, img.cols, CV_8UC1);
	int w = 2 * k + 1;

	int c = 0;

	for (int u = 0; u < w; ++u)
	{
		for (int v = 0; v < w; ++v)
		{
			c += nucleu[u][v];
		}
	}

	if (c == 0)
	{
		c = 1;
	}

	for (int row = 0; row < img.rows; ++row)
	{
		for (int col = 0; col < img.cols; ++col)
		{
			int s = 0;
			for (int u = 0; u < w; ++u)
			{
				for (int v = 0; v < w; ++v)
				{
					int r = row + u - k;
					int c = col + v - k;
					if (isInside(img, r, c))
					{
						s += (nucleu[u][v] * img.at<uchar>(r, c));
					}
				}
			}

			d.at<uchar>(row, col) = s / c;
		}
	}

	return d;
}

Mat convolutionFiltruSus(Mat img, int nucleu[3][3], int k)
{
	int sp = 0;
	int sm = 0;

	int w = 2 * k + 1;
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (nucleu[i][j] > 0)
			{
				sp += nucleu[i][j];
			}
			else
			{
				sm += (-nucleu[i][j]);
			}
		}
	}

	int lmax = -1;

	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			if (img.at<uchar>(row, col) > lmax)
			{
				lmax = img.at<uchar>(row, col);
			}
		}
	}

	int c = 2 * max(sm, sp);
	Mat d(img.rows, img.cols, CV_8UC1);

	for (int row = 0; row < img.rows; ++row)
	{
		for (int col = 0; col < img.cols; ++col)
		{
			int s = 0;
			for (int u = 0; u < w; ++u)
			{
				for (int v = 0; v < w; ++v)
				{
					int r = row + u - k;
					int c = col + v - k;
					if (isInside(img, r, c))
					{
						s += (nucleu[u][v] * img.at<uchar>(r, c));
					}
				}
			}

			d.at<uchar>(row, col) = getValidColor(s / c + lmax / 2);
		}
	}

	return d;
}

Mat convolution(Mat img, int nucleu[3][3], int k, CONVOLUTION_TYPE type)
{
	if (type == FILTRU_TRECE_JOS)
	{
		return convolutionFiltruJos(img, nucleu, k);
	}

	return convolutionFiltruSus(img, nucleu, k);
}

Mat filterConvolutionAverage(Mat img)
{
	int n[3][3] = {
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 1}
	};

	return convolution(img, n, 1, FILTRU_TRECE_JOS);
}

Mat filterGaussian(Mat img)
{
	int n[3][3] = {
		{1, 2, 1},
		{2, 4, 2},
		{1, 2, 1}
	};

	return convolution(img, n, 1, FILTRU_TRECE_JOS);
}

Mat filterLaplace(Mat img)
{
	int n[3][3] = {
		{0, -1, 0},
		{-1, 4, -1},
		{0, -1, 0}
	};

	return convolution(img, n, 1, FILTRU_TRECE_SUS);
}

Mat filterLaplace2(Mat img)
{
	int n[3][3] = {
		{-1, -1, -1},
		{-1, 8, -1},
		{-1, -1, -1}
	};

	return convolution(img, n, 1, FILTRU_TRECE_SUS);
}

Mat filterHighPass(Mat img)
{
	int n[3][3] = {
		{0, -1, 0},
		{-1, 5, -1},
		{0, -1, 0}
	};

	return convolution(img, n, 1, FILTRU_TRECE_SUS);
}

Mat filterHighPass2(Mat img)
{
	int n[3][3] = {
		{-1, -1, -1},
		{-1, 9, -1},
		{-1, -1, -1}
	};

	return convolution(img, n, 1, FILTRU_TRECE_SUS);
}

void centering_transform(Mat img)
{
	// imaginea trebuie să aibă elemente de tip float
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat generic_frequency_domain_filter(Mat src)
{
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	centering_transform(srcf);
	//aplicarea transformatei Fourier, se obține o imagine cu valori numere complexe
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
	//divizare în două canale: partea reală și partea imaginară
	Mat channels[] = {Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F)};
	split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))
	//calcularea magnitudinii și fazei în imaginile mag, respectiv phi, cu elemente de tip float
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);
	//aici afișați imaginile cu fazele și magnitudinile
	// ......

	//aici inserați operații de filtrare aplicate pe coeficienții Fourier
	// ......
	//memorați partea reală în channels[0] și partea imaginară în channels[1]
	// ......
	//aplicarea transformatei Fourier inversă și punerea rezultatului în dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);


	centering_transform(dstf);
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

Mat logMagnitudeFourier(Mat src)
{
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	centering_transform(srcf);
	//aplicarea transformatei Fourier, se obține o imagine cu valori numere complexe
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
	//divizare în două canale: partea reală și partea imaginară
	Mat channels[] = {Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F)};
	split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))
	//calcularea magnitudinii și fazei în imaginile mag, respectiv phi, cu elemente de tip float
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	for (int row = 0; row < mag.rows; row++)
	{
		for (int col = 0; col < mag.cols; col++)
		{
			mag.at<float>(row, col) = log(mag.at<float>(row, col) + 1);
		}
	}

	normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8UC1);

	return mag;
}

Mat filterLowPassIdeal(Mat src, int r)
{
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	//centering transformation
	centering_transform(srcf);

	//perform transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[2] = {Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F)};
	split(fourier, channels); //c[0]=Re(DFT(I)), c[1]=Im(DFT(I))

	int h = channels[0].rows;
	int w = channels[0].cols;

	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
			if (pow(h / 2 - row, 2) + pow(w / 2 - col, 2) > pow(r, 2))
			{
				channels[0].at<float>(row, col) = 0;
				channels[1].at<float>(row, col) = 0;
			}
	}

	//perform inverse transform
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	//inverse centering transf
	centering_transform(dstf);

	//normalize result
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

Mat filterHighPassIdeal(Mat src, int r)
{
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	//centering transformation
	centering_transform(srcf);

	//perform transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[2] = {Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F)};
	split(fourier, channels);

	int h = channels[0].rows;
	int w = channels[0].cols;

	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
			if (pow(h / 2 - row, 2) + pow(w / 2 - col, 2) < pow(r, 2))
			{
				channels[0].at<float>(row, col) = 0;
				channels[1].at<float>(row, col) = 0;
			}
	}

	//perform inverse transform
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	//inverse centering transf
	centering_transform(dstf);

	//normalize result
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

Mat filterGaussLowPassFrequency(Mat src, int a)
{
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	//centering transformation
	centering_transform(srcf);

	//perform transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[2] = {Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F)};
	split(fourier, channels);

	int h = channels[0].rows;
	int w = channels[0].cols;

	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
		{
			double rap = -(pow(h / 2 - row, 2) + pow(w / 2 - col, 2)) / pow(a, 2);
			channels[0].at<float>(row, col) = channels[0].at<float>(row, col) * exp(rap);
			channels[1].at<float>(row, col) = channels[1].at<float>(row, col) * exp(rap);
		}
	}

	//perform inverse transform
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	//inverse centering transf
	centering_transform(dstf);

	//normalize result
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}


Mat filterGaussHighPassFrequency(Mat src, int a)
{
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);

	//centering transformation
	centering_transform(srcf);

	//perform transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[2] = {Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F)};
	split(fourier, channels);

	int h = channels[0].rows;
	int w = channels[0].cols;

	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
		{
			double rap = -(pow(h / 2 - row, 2) + pow(w / 2 - col, 2)) / pow(a, 2);
			channels[0].at<float>(row, col) = channels[0].at<float>(row, col) * (1 - exp(rap));
			channels[1].at<float>(row, col) = channels[1].at<float>(row, col) * (1 - exp(rap));
		}
	}

	//perform inverse transform
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	//inverse centering transf
	centering_transform(dstf);

	//normalize result
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}


Mat medianFilter(Mat img, int w)
{
	Mat destination(img.rows, img.cols, CV_8UC1, Scalar(0));

	double t = (double)getTickCount();

	int margin = w / 2;
	int adv = (w * w) / 2;

	for (int row = margin; row < img.rows - margin - 1; ++row)
	{
		for (int col = margin; col < img.cols - margin - 1; ++col)
		{
			// std::multiset<int> elements;
			std::vector<int> elements;
			for (int r = row - margin; r < row + margin + 1; ++r)
			{
				for (int c = col - margin; c < col + margin + 1; ++c)
				{
					elements.push_back(img.at<uchar>(r, c));
				}
			}

			std::sort(elements.begin(), elements.end());

			destination.at<uchar>(row, col) = elements[adv];
		}
	}

	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("Method vector: Time = %.3f [ms] W: %d\n", t * 1000, w);

	return destination;
}

Mat medianFilter2(Mat img, int w)
{
	Mat destination(img.rows, img.cols, CV_8UC1, Scalar(0));

	double t = (double)getTickCount();

	int margin = w / 2;
	int adv = (w * w) / 2;
	for (int row = margin; row < img.rows - margin - 1; ++row)
	{
		for (int col = margin; col < img.cols - margin - 1; ++col)
		{
			std::multiset<int> elements;
			for (int r = row - margin; r < row + margin + 1; ++r)
			{
				for (int c = col - margin; c < col + margin + 1; ++c)
				{
					elements.insert(img.at<uchar>(r, c));
				}
			}

			std::multiset<int>::iterator it = elements.begin();

			advance(it, adv);

			destination.at<uchar>(row, col) = *it;
		}
	}

	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("Method sets: Time = %.3f [ms] W:\n", t * 1000, w);

	return destination;
}

Mat filterGaussianNoises(Mat img, int w)
{
	Mat destination = img.clone();

	double t = (double)getTickCount();


	float sigma = w / 6.0;
	float rap = 1.0 / (2.0 * PI * pow(sigma, 2));

	int k = w / 2;

	Mat values(w, w, CV_32F);

	// std::cout << "Param1M1" << std::endl;
	for (int row = 0; row < w; row++)
	{
		for (int col = 0; col < w; col++)
		{
			values.at<float>(row, col) = rap * exp(-(pow(row - w / 2, 2) + pow(col - w / 2, 2)) / (2 * pow(sigma, 2)));
		}
	}

	for (int row = k; row < img.rows - k; ++row)
	{
		for (int col = k; col < img.cols - k; ++col)
		{
			float s = 0.0;

			for (int u = 0; u < w; ++u)
			{
				for (int v = 0; v < w; ++v)
				{
					int r = row + u - k;
					int c = col + v - k;
					if (isInside(img, r, c))
					{
						s += values.at<float>(u, v) * img.at<uchar>(r, c);
					}
				}
			}

			destination.at<uchar>(row, col) = getValidColor(static_cast<int>(s));
		}
	}

	t = ((double)getTickCount() - t) / getTickFrequency();
	// printf("Time = %.3f [ms] W:\n", t * 1000, w);
	return destination;
}

Mat filterGaussianNoises2(Mat img, int w)
{
	Mat destination = img.clone();

	double t = (double)getTickCount();

	int margin = w / 2;

	float sigma = w / 6.0;
	float rap = 1.0 / (sqrt(2.0 * PI) * sigma);

	int k = w / 2;


	float* g = new float[w];

	for (int i = 0; i < w; i++)
	{
		g[i] = rap * exp(-(pow(i - w / 2, 2)) / (2 * pow(sigma, 2)));
	}

	for (int row = k; row < img.rows - k; ++row)
	{
		for (int col = k; col < img.cols - k; ++col)
		{
			float s = 0.0;

			for (int u = 0; u < w; ++u)
			{
				// int r = row + u - k;
				int c = col + u - k;
				// if (isInside(img, r, col)) {
				// 	s += g[u] * img.at<uchar>(r, col);
				// }

				if (isInside(img, row, c))
				{
					s += g[u] * img.at<uchar>(row, c);
				}
			}

			destination.at<uchar>(row, col) = getValidColor(static_cast<int>(s));
		}
	}

	Mat dest2 = destination.clone();
	for (int row = k; row < img.rows - k; ++row)
	{
		for (int col = k; col < img.cols - k; ++col)
		{
			float s = 0.0;

			for (int u = 0; u < w; ++u)
			{
				int r = row + u - k;
				// int c = col + u - k;
				if (isInside(img, r, col))
				{
					s += g[u] * img.at<uchar>(r, col);
				}

				// if (isInside(img, row, c)) {
				// 	s += g[u] * img.at<uchar>(row, c);
				// }
			}

			dest2.at<uchar>(row, col) = getValidColor(static_cast<int>(s));
		}
	}


	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("Time = %.3f [ms] W:\n", t * 1000, w);
	return dest2;
}

Mat convolution(Mat img, int nucleu[3][3])
{
	Mat d(img.rows, img.cols, CV_8UC1, Scalar(0));

	int k = 1, w = 3;
	for (int row = 0; row < img.rows; ++row)
	{
		for (int col = 0; col < img.cols; ++col)
		{
			int s = 0;
			for (int u = 0; u < w; ++u)
			{
				for (int v = 0; v < w; ++v)
				{
					int r = row + u - k;
					int c = col + v - k;
					if (isInside(img, r, c))
					{
						s += (nucleu[u][v] * img.at<uchar>(r, c));
					}
				}
			}

			d.at<uchar>(row, col) = getValidColor(s);
		}
	}

	return d;
}

Mat convolution(Mat img, int nucleu[2][2])
{
	Mat d(img.rows, img.cols, CV_8UC1, Scalar(0));

	int k = 0, w = 2;
	for (int row = 0; row < img.rows; ++row)
	{
		for (int col = 0; col < img.cols; ++col)
		{
			int s = 0;
			for (int u = 0; u < w; ++u)
			{
				for (int v = 0; v < w; ++v)
				{
					int r = row + u - k;
					int c = col + v - k;
					if (isInside(img, r, c))
					{
						s += (nucleu[u][v] * img.at<uchar>(r, c));
					}
				}
			}

			d.at<uchar>(row, col) = getValidColor(s);
		}
	}

	return d;
}

std::vector<Mat> computeGradient(Mat img, int nucleuX[3][3], int nucleuy[3][3])
{
	std::vector<Mat> res;

	res.push_back(convolution(img, nucleuX));
	res.push_back(convolution(img, nucleuy));

	res.push_back(convolutionFiltruSus(img, nucleuX, 1));
	res.push_back(convolutionFiltruSus(img, nucleuy, 1));
	return res;
}


std::vector<Mat> computeGradientWithPrewitt(Mat img)
{
	int nucleuX[3][3] = {
		{-1, 0, 1},
		{-1, 0, 1},
		{-1, 0, 1}
	};

	int nucleuY[3][3] = {
		{1, 1, 1},
		{0, 0, 0},
		{-1, -1, -1}
	};

	return computeGradient(img, nucleuX, nucleuY);
}


std::vector<Mat> computeGradientWithSobel(Mat img)
{
	int nucleuX[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};

	int nucleuY[3][3] = {
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1}
	};

	return computeGradient(img, nucleuX, nucleuY);
}

std::vector<Mat> computeGradientWithRobertCross(Mat img)
{
	int nucleuX[2][2] = {
		{1, 0},
		{0, -1}
	};

	int nucleuY[2][2] = {
		{0, -1},
		{1, 0}
	};

	std::vector<Mat> res;
	res.push_back(convolution(img, nucleuX));
	res.push_back(convolution(img, nucleuY));

	return res;
}

/**
 * type 0 -> prewiit
 * 1 -> robert cross
 * 2 -> sobel
 */
std::vector<Mat> computeModuleAndDirection(Mat img, int type)
{
	std::vector<Mat> fxfy;
	std::vector<Mat> res;

	switch (type)
	{
		case 0:
			fxfy = computeGradientWithPrewitt(img);
			break;
		case 1:
			fxfy = computeGradientWithRobertCross(img);
			break;
		case 2:
			fxfy = computeGradientWithSobel(img);
			break;
		default:
			fxfy = computeGradientWithSobel(img);
	}

	Mat moduleGradient(img.rows, img.cols, CV_8UC1, Scalar(0));
	Mat directionGradient(img.rows, img.cols, CV_8UC1, Scalar(0));


	for (int r = 0; r < img.rows; r++)
	{
		for (int c = 0; c < img.cols; c++)
		{
			moduleGradient.at<uchar>(r, c) = getValidColor((int)std::sqrt(fxfy[0].at<uchar>(r, c) * fxfy[0].at<uchar>(r, c) + fxfy[1].at<uchar>(r, c) * fxfy[1].at<uchar>(r, c)));
			float teta = std::atan2(fxfy[1].at <uchar> (r, c), fxfy[0].at<uchar>(r, c));
			int dir = 0;
			if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8)) dir = 0;
			if ((teta > PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) dir = 1;
			if ((teta > -PI / 8 && teta < PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8) dir = 2;
			if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta < -PI / 8)) dir = 3;

			directionGradient.at<uchar>(r,c) = dir;
		}
	}

	res.push_back(moduleGradient);
	res.push_back(directionGradient);

	return res;

}


Mat imageFixedBinarisation(Mat img, float p)
{
	std::vector<Mat> res = computeModuleAndDirection(img, 0);

	int hist[256];
	for (int i = 0 ; i < 256; i++)
	{
		hist[i] = 0;
	}

	for (int r = 0 ; r < img.rows; r++)
	{
		for (int c = 0; c < img.cols; c++)
		{
			hist[res[0].at<uchar>(r, c)]++;
		}
	}

	int nrNonMuchie = (1 - p) * (img.rows * img.cols - hist[0]);

	int s = 0;
	int prag = 0;
	for (int i = 1; i < 256; i++)
	{
		s += hist[i];

		if (s > nrNonMuchie)
		{
			prag = i;
			break;
		}
	}

	return blackAndWhite(res[0], prag);
}


Mat cannyAlgorithm(Mat img)
{
	Mat imgFilter = filterGaussian(img);
	std::vector<Mat> modAndDir = computeModuleAndDirection(imgFilter, 2);

	Mat res(img.rows, img.cols, CV_8UC1, Scalar(0));

	// imshow("Mod", modAndDir[0]);

	

	for (int r = 1; r < img.rows - 1; r++)
	{
		for (int c = 1 ; c < img.cols; c++)
		{
			bool isMaxim = false;

			switch (modAndDir[1].at<uchar>(r,c))
			{
			case 0:
				isMaxim = (modAndDir[0].at<uchar>(r - 1, c) < modAndDir[0].at<uchar>(r, c) && modAndDir[0].at<uchar>(r + 1, c) < modAndDir[0].at<uchar>(r, c));
				break;
			case 1:
				isMaxim = (modAndDir[0].at<uchar>(r - 1, c + 1) < modAndDir[0].at<uchar>(r, c) && modAndDir[0].at<uchar>(r + 1, c - 1) < modAndDir[0].at<uchar>(r, c));
				break;
			case 2:
				isMaxim = (modAndDir[0].at<uchar>(r, c - 1) < modAndDir[0].at<uchar>(r, c) && modAndDir[0].at<uchar>(r, c + 1) < modAndDir[0].at<uchar>(r, c));
				break;
			case 3:
				isMaxim = (modAndDir[0].at<uchar>(r - 1, c - 1) < modAndDir[0].at<uchar>(r, c) && modAndDir[0].at<uchar>(r + 1, c + 1) < modAndDir[0].at<uchar>(r, c));
				break;
			}

			res.at<uchar>(r, c) = (isMaxim) ? modAndDir[0].at<uchar>(r, c) : 0;
		}
	}

	return res;
}


float filter(Mat source, Mat mask, int row, int col, bool uh)
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
				if (uh) {
					out += (mask.at<float>(u, v) * source.at<uchar>(r, c));
				} else {
					out += (mask.at<float>(u, v) * source.at<float>(r, c));
				}
			}
		}
	}

	return out;
}


Mat convolution(Mat source, Mat mask, bool uh)
{
	Mat destination(source.rows, source.cols, CV_32FC1);

	int hs_r = mask.rows / 2;
	int hs_c = mask.cols / 2;
	for (int r = hs_r; r < source.rows - hs_r; ++r)
	{
		for (int c = hs_c; c < source.cols - hs_c; c++)
		{
			destination.at<float>(r, c) = filter(source, mask, r, c, uh);
		}
	}

	return destination;
}

Mat mul(Mat m1, Mat m2)
{
	Mat dest = m1.clone();

	for (int r = 0; r < m1.rows; r++)
	{
		for (int c = 0; c < m1.cols; c++)
		{
			dest.at<float>(r, c) *= m2.at<float>(r, c);
		}
	}

	return dest;
}

Mat sub(Mat m1, Mat m2)
{
	Mat dest = m1.clone();

	for (int r = 0; r < m1.rows; r++)
	{
		for (int c = 0; c < m1.cols; c++)
		{
			dest.at<float>(r, c) -= m2.at<float>(r, c);
		}
	}

	return dest;
}

Mat add(Mat m1, Mat m2)
{
	Mat dest = m1.clone();

	for (int r = 0; r < m1.rows; r++)
	{
		for (int c = 0; c < m1.cols; c++)
		{
			dest.at<float>(r, c) += m2.at<float>(r, c);
		}
	}

	return dest;
}

void harrisCorners(Mat source, float k)
{
	Mat gray;

	cvtColor(source, gray, COLOR_BGR2GRAY);

	imshow("Img gray", gray);
	imshow("Original image", source);

	float sobel_x[9] = { -1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f };
	float sobel_y[9] = { -1.0f, -2.0f, -1.0f, 0.0, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f };
	float sum_mask[9] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

	Mat mask_x(3, 3, CV_32FC1, sobel_x);
	Mat mask_y(3, 3, CV_32FC1, sobel_y);
	Mat mask_sum(3, 3, CV_32FC1, sum_mask);

	Mat pxFilter = convolution(gray, mask_x, true);
	Mat pyFilter = convolution(gray, mask_y, true);

	imshow("PxFilter", pxFilter);
	imshow("PyFilter", pyFilter);

	Mat px2 = mul(pxFilter, pxFilter);
	Mat py2 = mul(pyFilter, pyFilter);
	Mat pxpy = mul(pxFilter, pyFilter);

	Mat sumpx2 = convolution(px2, mask_sum, false);
	Mat sumpy2 = convolution(py2, mask_sum, false);
	Mat sumpxpy = convolution(pxpy, mask_sum, false);

	Mat mul1 = mul(sumpx2, sumpy2);
	Mat mul2 = mul(sumpxpy, sumpxpy);
	Mat det = sub(mul1, mul2);


	Mat trace = add(sumpx2, sumpy2);

	Mat trace2(trace.rows, trace.cols, CV_32FC1, Scalar(k));
	for (int r = 0; r < trace2.rows; r++)
	{
		for (int c = 0; c < trace2.cols; c++)
		{
			trace2.at<float>(r, c) *= trace.at<float>(r, c);
		}
	}

	Mat harris = sub(det, trace2);

	float threshold = 150.0f;
	std::vector<Point> corners;
	for (int r = 0; r < harris.rows; r++)
	{
		for (int c = 0; c < harris.cols; c++)
		{
			if (harris.at<float>(r,c) > threshold)
			{
				Point p(c, r);
				corners.push_back(p);
			}
		}
	}

	for (auto p: corners)
	{

		circle(source, p, 5, Scalar(0), 5, 8, 0);
		// std::cout << p.x << " " << p.y << std::endl;
	}

	imshow("corners", source);
	// Mat harris(trace2.rows, trace2.cols, CV_32FC1, Scalar(k));
}

bool compareSum(Point p1, Point p2)
{
	return (p1.x + p1.y) < (p2.x + p2.y);
}

bool compareDiff(Point p1, Point p2)
{
	return (p1.x - p1.y) < (p2.x - p2.y);
}

void getBorderBoxes(std::vector<Point> points, Point &tl, Point &tr, Point &bl, Point &br)
{
	std::sort(points.begin(), points.end(), compareSum);


	tl.x = points[0].x;
	tl.y = points[0].y;
	br.x = points[points.size() - 1].x;
	br.y = points[points.size() - 1].y;

	std::sort(points.begin(), points.end(), compareDiff);


	bl.x = points[0].x;
	bl.y = points[0].y;
	tr.x = points[points.size() - 1].x;
	tr.y = points[points.size() - 1].y;
}

Mat perspectiveProjection(Mat source, cv::Point2f sourcePoints[4], int s)
{
	Point2f destinationPoints[4];
	destinationPoints[0] = Point2f(0, 0);
	destinationPoints[1] = Point2f(source.rows - 1, 0);
	destinationPoints[2] = Point2f(0, source.cols - 1);
	destinationPoints[3] = Point2f(source.rows - 1, source.cols - 1);

	std::ofstream fout("file.txt");
	
	Mat m = getPerspectiveTransform(sourcePoints, destinationPoints);
	// // std::cout << m.type();
	Mat mapx(source.rows, source.cols, CV_64FC1);
	Mat mapy(source.rows, source.cols, CV_64FC1);

	Mat destination(source.rows, source.cols, source.type());

	for (int r = 0; r <source.rows; r++)
	{
		for (int c = 0; c < source.cols; c++)
		{
			double xp =  (m.at<double>(0, 0) * r + m.at<double>(0, 1) * c + m.at<double>(0, 2)) / (m.at<double>(2, 0) * r + m.at<double>(2, 1) * c + 1);
			double yp =  (m.at< double>(1, 0) * r + m.at<double>(1, 1) * c + m.at<double>(1, 2)) / (m.at<double>(2, 0) * r + m.at<double>(2, 1) * c + 1);
			xp = std::abs(static_cast<int>(xp));
			yp = std::abs(static_cast<int>(yp));


			if (xp > source.rows - 1)
			{
				xp = source.rows - 1;
			}
			if (yp > source.cols - 1)
			{
				yp = source.cols - 1;
			}
			// mapx.at<double>(r, c) = xp;
			// mapy.at<double>(r, c) = yp;



			// fout << "(" << xp << "," << yp << ") ";
			// xp = std::abs(static_cast<int>(xp));
			// yp = std::abs(static_cast<int>(yp));

			// if (isInside(source, xp, yp))
			// {
				destination.at<Vec3b>(r, c) = source.at<Vec3b>(xp, yp);
			// }
		}
		// fout << "\n";
	}


	// remap(source, destination, mapx, mapy, CV_INTER_LINEAR, BORDER_REPLICATE);


	return destination;
}