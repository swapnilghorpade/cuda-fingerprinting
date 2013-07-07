#include <stdio.h>
#include <stdlib.h>
#pragma once
class Point {
public:
	int X;
	int Y;
	Point(int x, int y) {
		X = x;
		Y = y;
	}

	bool Equals(Point P) {
		return (X == P.X) && (Y == P.Y);
	}

	int Distance(Point P) {
		return (X-P.X)*(X-P.X) + (Y-P.Y)*(Y-P.Y);
	}

	Point Subtract(Point P) {
		Point *temp = new Point(X-P.X,Y-P.Y);
		return *temp;
	}

	int VectorProduct(Point P) {
		return X*P.Y - Y* P.X;
	}
	void print() {
		printf("(%d,%d)\n",X,Y);
	}
};
