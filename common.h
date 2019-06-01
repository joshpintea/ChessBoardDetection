#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <windows.h>
#include "ObjectProperties.h"

using namespace cv;

#define PI 3.14159265
#define POS_INFINITY 1e30
#define NEG_INFINITY -1e30
#define max_(x,y) ((x) > (y) ? (x) : (y))
#define min_(x,y) ((x) < (y) ? (x) : (y))
#define isNan(x) ((x) != (x) ? 1 : 0)

class FileGetter{
	WIN32_FIND_DATAA found;	
	HANDLE hfind;
	char folder[MAX_PATH];			
	int chk;
	bool first;
	bool hasFiles;
public:
	FileGetter(char* folderin,char* ext);
	int getNextFile(char* fname);
	int getNextAbsFile(char* fname);
	char* getFoundFileName();
};

enum REDUCE_GRAY_LEVEL_ALGORITHM
{
	MULTIPLE_PRAGS,
	FLOYD_STEINBERG
};


enum CONVOLUTION_TYPE
{
	FILTRU_TRECE_SUS,
	FILTRU_TRECE_JOS
};

int openFileDlg(char* fname);

int openFolderDlg(char* folderName);

void resizeImg(Mat src, Mat &dst, int maxSize, bool interpolate);

int getValidColor(int color);

Mat grayScale(Mat img);

std::vector<Mat> getRGBChannels(Mat img);

Mat blackAndWhite(Mat image, int prague);

Vec3b rgbColorToHSVColor(Vec3b color);

Vec3b hsvColorToRGBColor(Vec3b hsvColor);

std::vector<Mat> rgbToHsv(Mat img);

bool isInside(Mat img, int i, int j);

int * computeHistogram(Mat img, int accumulator = 256);

float* computeFDP(Mat img);

void showHistogramI(const std::string& name, int* hist, int hist_cols, const int hist_height);

std::vector<int> computeMaxLocals(Mat img, int wh = 5, float th = 0.0003f);

Mat reduceGrayLevels(Mat img, REDUCE_GRAY_LEVEL_ALGORITHM algorithm);

int getClosserMaxim(int color, std::vector<int> maxLocals);

Mat reduceGrayLevelOnHChannel(Mat img);


ObjectProperties computeObjectProperties(Mat img, Vec3b color);

ObjectProperties computeObjectProperties(Mat img, uchar color);

std::vector<ObjectProperties> computeObjectsProperties(Mat img);

Point getPointAfterAngle(Point p,int lenght, float angle);

Mat imageLabeling(Mat img);

std::vector<Mat> imageLabelingAg2(Mat img);

Mat colorImageLabeled(Mat img);

Mat traceContour(Mat img, int n);

Mat reconstructContour(Point start, std::vector<int> code);

Mat dilatation(Mat img, int dilationSize);

Mat erosion(Mat img, int erosionSize);

Mat opening(Mat img, int size);

Mat closure(Mat img, int size);

std::vector<Point> extractContour(Mat img);

Mat regionFill(Mat img, Point startPoint);

double getAverage(Mat img);

double computeDeviation(Mat img);

int* computeFDPC(Mat img);

Mat imgBinarization(Mat img, float error = 0.05);

Mat negativeImg(Mat img);

Mat changeContrast(Mat img, int goMin, int goMax);

Mat gammaCorrection(Mat img, float gamma);

Mat changeLuminosity(Mat img, int luminosity);

Mat equalizeHistogram(Mat img);

Mat convolution(Mat img, int **nucleu, int k, CONVOLUTION_TYPE type);

Mat filterGaussian(Mat img);

Mat filterConvolutionAverage(Mat img);

Mat generic_frequency_domain_filter(Mat src);

void centering_transform(Mat img);

Mat convolutionFiltruJos(Mat img, int nucleu[3][3], int k);

Mat convolutionFiltruSus(Mat img, int  nucleu[3][3], int k);

Mat filterLaplace(Mat img);
Mat filterLaplace2(Mat img);
Mat filterHighPass(Mat img);
Mat filterHighPass2(Mat img);

Mat logMagnitudeFourier(Mat src);

Mat filterLowPassIdeal(Mat src, int r);
Mat filterHighPassIdeal(Mat src, int r);
Mat filterGaussLowPassFrequency(Mat src, int a);
Mat filterGaussHighPassFrequency(Mat src, int a);

Mat medianFilter(Mat img, int w);
Mat medianFilter2(Mat img, int w);

Mat filterGaussianNoises(Mat img, int w);
Mat filterGaussianNoises2(Mat img, int w);


std::vector<Mat> computeGradientWithSobel(Mat img);

std::vector<Mat> computeGradientWithPrewitt(Mat img);
std::vector<Mat> computeGradient(Mat img, int nucleuX[3][3], int nucleuy[3][3]);
std::vector<Mat> computeGradientWithRobertCross(Mat img);

std::vector<Mat> computeModuleAndDirection(Mat img, int type);

Mat imageFixedBinarisation(Mat imp, float p);
Mat cannyAlgorithm(Mat img);



float filter(Mat source, Mat mask, int row, int col, bool uh = true);
Mat convolution(Mat source, Mat mask, bool uh);
void harrisCorners(Mat source, float k);
Mat removeRedColor(Mat source, int colorMin, int colorMax);

void getBorderBoxes(std::vector<Point> points, Point &tl, Point &tr, Point &bl, Point &br);

Mat perspectiveProjection(Mat source, cv::Point2f sourcePoints[4], int s);
