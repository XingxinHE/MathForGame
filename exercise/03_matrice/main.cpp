#include <iostream>
#include <Eigen/Dense>
#include <math.h> 
#include <Eigen/LU>
#include <vector>

using namespace Eigen;
using namespace std;

void ExerciseWithEigenFunction();
void ExerciseWithEigen();

int main()
{
	ExerciseWithEigenFunction();
	ExerciseWithEigen();
}

void ExerciseWithEigenFunction()
{
	cout << "\n\n\n***************************************************" << endl;
	cout << "***************************************************" << endl;
	cout << "The following exercise is solved by Eigen function." << endl;
	cout << "***************************************************" << endl;
	cout << "***************************************************" << endl;
	//Exercise 1
	cout << "\nExercise 1 \n" << endl;
	Matrix2d M1a;
	M1a << 2, 7, -3, 1 / 2;
	Matrix3d M1b;
	M1b << 0, 0, 1, 0, 1, 0, 1, 0, 0;
	Matrix3d M1c;
	M1c << 1 / 2, sqrt(3) / 2, 0, -sqrt(3) / 2, 1 / 2, 0, 0, 0, 1;
	Matrix3d M1d;
	M1d << 5, 7, 1, 17, 2, 64, 10, 14, 2;

	cout << "M1a: \n" << M1a << endl;
	cout << "M1a determinant: " << M1a.determinant() << endl;
	cout << "M1b: \n" << M1b << endl;
	cout << "M1b determinant: " << M1b.determinant() << endl;
	cout << "M1c: \n" << M1c << endl;
	cout << "M1c determinant: " << M1c.determinant() << endl;
	cout << "M1d: \n" << M1d << endl;
	cout << "M1d determinant: " << M1d.determinant() << endl;

	//Exercise 2
	cout << "\n\n\nExercise 2 \n" << endl;
	Matrix3d M2a;
	Matrix3d M2b;
	Matrix4d M2d;
	M2a << 2, 0, 0, 0, 3, 0, 0, 0, 4;
	M2b << 1, 0, 0, 0, 2, 2, 3, 0, 8;
	M2d << 1, 0, 0, 4,
		0, 1, 0, 3,
		0, 0, 1, 7,
		0, 0, 0, 1;
	cout << "M2a: \n" << M2a << endl;
	cout << "M2a inverse: \n" << M2a.inverse() << endl;
	cout << "M2b: \n" << M2b << endl;
	cout << "M2b inverse: \n" << M2b.inverse() << endl;
	cout << "M2d: \n" << M2d << endl;
	cout << "M2d inverse: \n" << M2d.inverse() << endl;


	//Exercise 4
	cout << "\n\n\nExercise 4 \n" << endl;
	Matrix3d M4;
	M4 << 2, 0, 0,
		5, 2, 3,
		-4, 3, 2;
	cout << "M4: \n" << M4 << endl;
	cout << "M4 eigenvalues: " << M4.eigenvalues() << endl;
}

void ExerciseWithEigen()
{
	cout << "\n\n\n******************************************************************************" << endl;
	cout << "******************************************************************************" << endl;
	cout << "The following exercise is solved by function written by myself with Eigen api." << endl;
	cout << "******************************************************************************" << endl;
	cout << "******************************************************************************" << endl;
}