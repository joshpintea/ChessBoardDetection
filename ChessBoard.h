#pragma once
#include <opencv2/core/mat.hpp>
#include "Piece.h"

using namespace cv;
using namespace std;

enum CHESSBOARD_RECONSTRUCT_SOLVER
{
	METHOD1,
	METHOD2
};

enum CLASSIFICATION
{
	HORIZONTAL,
	HOG
};


class ChessBoard
{
private:
	string basePath = "D:/MyWorkSpace/Image Processing/ChessBoardDetection/ChessBoardDetection/Images/training_images/";
	vector<Point> points;
	vector<vector<int>> horizontalHistPawn;
	vector<vector<int>> horizontalHistRook;
	vector<vector<int>> horizontalHistKing;
	vector<vector<int>> horizontalHistKnight;
	vector<vector<int>> horizontalHistEmpty;
	vector<vector<int>> horizontalHistBishop;
	vector<vector<int>> horizontalHistQueen;
	vector<vector<Piece>> chessBoardPieces;
	Mat imgProjectedGray;

	void trainDescriptorHorizontal(vector<vector<int>> &desc, string path, int count);
	void reconstructChessBoardMethod1(Mat img, CLASSIFICATION cls);
	string classificationHorizontal(Mat img);
	string classificationHog(Mat img);

public:
	void setPoints(vector<Point> points);
	void train();
	void reconstructChessBoard(Mat sourceImg, CHESSBOARD_RECONSTRUCT_SOLVER solver, CLASSIFICATION cls);
};
