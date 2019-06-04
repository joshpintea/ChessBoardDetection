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
	string basePath = "C:/Users/Bobossuno/Desktop/PICLONE99/ChessBoardDetection/Images/training_images/";
	string piecesPath = "C:/Users/Bobossuno/Desktop/PICLONE99/ChessBoardDetection/Images/pieces/";
	string darkDir = piecesPath + "dark_background/";
	string whiteDir = piecesPath + "white_background/";

	vector<Point> points;
	vector<vector<int>> horizontalHistPawn;
	vector<vector<int>> horizontalHistRook;
	vector<vector<int>> horizontalHistKing;
	vector<vector<int>> horizontalHistKnight;
	vector<vector<int>> horizontalHistEmpty;
	vector<vector<int>> horizontalHistBishop;
	vector<vector<int>> horizontalHistQueen;
	vector<vector<Piece>> chessBoardPieces;
	//hog
	vector<vector<float>> bishopDescriptors;
	vector<vector<float>> kingDescriptors;
	vector<vector<float>> pawnDescriptors;
	vector<vector<float>> emptyDescriptors;
	vector<vector<float>> knightDescriptors;
	vector<vector<float>> queenDescriptors;
	vector<vector<float>> rookDescriptors;


	Mat imgProjectedGray;

	void trainDescriptorHorizontal(vector<vector<int>> &desc, string path, int count);
	void reconstructChessBoardMethod1(Mat img, CLASSIFICATION cls);
	void reconstructChessBoardMethod2(Mat img, CLASSIFICATION cls);

	string classificationHorizontal(Mat img);
	string classificationHog(Mat img);
	float vectors_distance(vector<float> a, vector<float> b);
	bool myCompare(pair<float, int> a, pair<float, int> b);
	string decide(float b, float p, float kn, float k, float q, float e, float r);

	void colorClassification(Mat img);
public:
	void setPoints(vector<Point> points);
	void train();
	void reconstructChessBoard(Mat sourceImg, CHESSBOARD_RECONSTRUCT_SOLVER solver, CLASSIFICATION cls);
	void trainBishopDescriptor();
	void trainPawnDescriptor();
	void trainKingDescriptor();
	void trainQueenDescriptor();
	void trainRookDescriptor();
	void trainKnightDescriptor();
	void trainEmptyDescriptor();

};
