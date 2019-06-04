#include "stdafx.h"
#include "Piece.h"
#include <iostream>


std::string Piece::toString()
{
	std::string res;
	res = "(" + std::to_string(backgroundColor) + " " + this->type + " " + std::to_string(color) + ") ";
	return res;
}
