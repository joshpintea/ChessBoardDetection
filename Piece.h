#pragma once
#include <string>

class Piece
{
public:
	std::string type;
	float sumColor;
	int color; // white and blue
	int backgroundColor;

	int widthPositive;
	int widthNegative;
	int positive;
	int negative;
	std::string toString();
	Piece()
	{
		type = "None";
		color = -1;
		backgroundColor = -1;
	}
};
