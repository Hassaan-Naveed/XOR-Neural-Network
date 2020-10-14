// MNISTLoader.cpp

#include <iostream>;
#include <fstream>;
#include <sstream>;
#include <vector>;
#include <tuple>;
using namespace std;

class Loader 
{
public:

	Loader() 
	{
	}

	tuple<vector<int>> loadData()
	{
		
		tuple<vector<int>> testingData;
		tuple<vector<int>> trainingData;

		vector<int> testingLabels;
		vector<int> testingPics;

		vector<int> trainingLabels;
		vector<int> trainingPics;

		ifstream file;
		string line;

		file.open("./MNIST Data Set/mnist_test.csv", ios::in);

		while (getline(file, line)) 
		{
			string lineVal;

		}

		testingData = testingLabels, testingPics;
		trainingData = trainingLabels, trainingPics;

		return testingData, trainingData;
	}

	vector<int> vectorisedResult(int i) 
	{
		vector<int> vec = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		vec[i] = 1;
		return vec;
	}
	
};