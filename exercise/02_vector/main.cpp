#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <cmath>

std::vector<Eigen::Vector3d> GramSchimitOrtho(std::vector<Eigen::Vector3d> input_vectors);
std::vector<Eigen::Vector4d> GramSchimitOrtho(std::vector<Eigen::Vector4d> input_vectors);



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
	std::cout << "\n\n\nExercise 2. \n" << std::endl;
	Eigen::Vector3d e1((sqrt(2) / 2), (sqrt(2) / 2), 0);
	Eigen::Vector3d e2(-1, 1, -1);
	Eigen::Vector3d e3(0, -2, -2);

	std::vector<Eigen::Vector3d> input_vectors{ e1,e2,e3 };
	std::cout << "Input vectors are: \n" << std::endl;
	for (auto v : input_vectors)
	{
		std::cout << "Vector--" << std::endl;
		std::cout << v << std::endl;
	}

	std::vector<Eigen::Vector3d> ortho_vectors = GramSchimitOrtho(input_vectors);

	std::cout << "After Gram-Schmidt Orthogonalization: \n" << std::endl;
	for (auto v : ortho_vectors)
	{
		std::cout << v << std::endl;
	}


	std::cout << "\nEvaluation: " << std::endl;
	std::cout << "q1 * q2" << std::endl;
	std::cout << round(ortho_vectors[0].dot(ortho_vectors[1])) << std::endl;
	std::cout << "q1 * q3" << std::endl;
	std::cout << round(ortho_vectors[0].dot(ortho_vectors[2])) << std::endl;
	std::cout << "q2 * q3" << std::endl;
	std::cout << round(ortho_vectors[1].dot(ortho_vectors[2])) << std::endl;



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


	//Exercise Extra - Testing R^n for GramSchmidt algorithm
	std::cout << "\n\n\nExercise extra. \n" << std::endl;
	Eigen::Vector4d W(8, 1, 3, 4);
	Eigen::Vector4d E(2, 4, 2, 6);
	Eigen::Vector4d R(3, 9, 5, 3);
	Eigen::Vector4d T(7, 8, 6, 2);
	std::vector<Eigen::Vector4d> input_vectors_Xd{ W,E,R,T };
	std::vector<Eigen::Vector4d> output_vectors_Xd = GramSchimitOrtho(input_vectors_Xd);

	std::cout << "Input R^n vectors are: \n" << std::endl;
	for (auto v : input_vectors_Xd)
	{
		std::cout << "Vector--" << std::endl;
		std::cout << v << std::endl;
	}

	std::cout << "After Gram-Schmidt Orthogonalization: \n" << std::endl;
	for (auto v : output_vectors_Xd)
	{
		std::cout << v << std::endl;
	}


	std::cout << "\nEvaluation: " << std::endl;
	std::cout << "a1 * a2" << std::endl;
	std::cout << round(output_vectors_Xd[0].dot(output_vectors_Xd[1])) << std::endl;
	std::cout << "a1 * a3" << std::endl;
	std::cout << round(output_vectors_Xd[0].dot(output_vectors_Xd[2])) << std::endl;
	std::cout << "a1 * a4" << std::endl;
	std::cout << round(output_vectors_Xd[0].dot(output_vectors_Xd[3])) << std::endl;
	std::cout << "a2 * a3" << std::endl;
	std::cout << round(output_vectors_Xd[1].dot(output_vectors_Xd[2])) << std::endl;
	std::cout << "a2 * a4" << std::endl;
	std::cout << round(output_vectors_Xd[1].dot(output_vectors_Xd[3])) << std::endl;
	std::cout << "a3 * a4" << std::endl;
	std::cout << round(output_vectors_Xd[2].dot(output_vectors_Xd[3])) << std::endl;

	std::string input = "";
	std::cin >> input;
	return 0;
}



std::vector<Eigen::Vector3d> GramSchimitOrtho(std::vector<Eigen::Vector3d> input_vectors)
{
	std::vector<Eigen::Vector3d> ortho_vectors;


	int amount = input_vectors.size();
	Eigen::Vector3d q1 = input_vectors[0];
	ortho_vectors.push_back(q1);
	for (int i = 1; i < amount; i++)
	{
		Eigen::Vector3d rhs;
		rhs.setZero();
		for (int j = 0; j < i; j++)
		{
			rhs = rhs + (input_vectors[i].dot(ortho_vectors[j].normalized()) * (ortho_vectors[j].normalized()));
		}

		ortho_vectors.push_back(input_vectors[i] - rhs);
	}

	return ortho_vectors;
}

std::vector<Eigen::Vector4d> GramSchimitOrtho(std::vector<Eigen::Vector4d> input_vectors)
{

	std::vector<Eigen::Vector4d> ortho_vectors;



	int amount = input_vectors.size();
	Eigen::Vector4d q1 = input_vectors[0];
	ortho_vectors.push_back(q1);
	for (int i = 1; i < amount; i++)
	{
		Eigen::Vector4d rhs;
		rhs.setZero();
		for (int j = 0; j < i; j++)
		{
			rhs = rhs + (input_vectors[i].dot(ortho_vectors[j].normalized()) * (ortho_vectors[j].normalized()));
		}

		ortho_vectors.push_back(input_vectors[i] - rhs);
	}

	return ortho_vectors;
}