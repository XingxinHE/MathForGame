#include <iostream>
#include <Eigen/Dense>
#include <string>

int main()
{
	//Exercise 1
	std::cout << "Exercise 1. \n" << std::endl;

	Eigen::Vector3d P(2, 2, 1);
	Eigen::Vector3d Q(1, -2, 0);

	std::cout << "Vector P = <2, 2, 1>" << std::endl;
	std::cout << "Vector Q = <1, -2, 0>" << std::endl;
	std::cout << "The dot product is: " << P.dot(Q) << std::endl;
	std::cout << "The cross product is a vector: \n" << P.cross(Q) << std::endl;
	

	//Exercise 2
	//TODO

	//Exercise 3
	std::cout << "\n\n\nExercise 3. \n" << std::endl;
	std::cout << "Point P1 = (1, 2, 3)" << std::endl;
	std::cout << "Point P2 = (-2, 2, 4)" << std::endl;
	std::cout << "Point P3 = (7, -8, 6)" << std::endl;
	Eigen::Vector3d P1(1, 2, 3);
	Eigen::Vector3d P2(-2, 2, 4);
	Eigen::Vector3d P3(7, -8, 6);
	std::cout << "\n The vector is: " << std::endl;
	Eigen::Vector3d P1P2 = P2 - P1;
	Eigen::Vector3d P1P3 = P3 - P1;
	std::cout << "Vector P1->P2: \n" << P1P2 << std::endl;
	std::cout << "Vector P1->P3: \n" << P1P3 << std::endl;
	std::cout << "\nThe area of parallelogram formed by 3 points is: "
		<< P1P2.cross(P1P3).norm() << std::endl;

	return 0;
}
