// Project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
// Included Libraries
#include <iostream>;
#include <fstream>;
#include <sstream>;
#include <string>;
#include <time.h>;
#include <vector>;
#include <random>;
#include <Eigen/Dense>;

using namespace std;
using namespace Eigen;

// Data Loader Class
class Loader
{
public:

	void loadData(vector<vector<vector<vector<int>>>>& testingData, vector<vector<vector<vector<int>>>>& trainingData)
	{
		// A Guide to these Vectors:
		// The bottom layer vectors contain the data itself. There are 2 of these
		// One contains the data, and the other contains the labels
		// These vectors are wrapped in another vector, called "wrapper", so they can be randomly sampled easily for SGD
		// There is one wrapper vector for each image, therefore there are 70,000 wrapper vectors in total
		// All wrapper vectors are contained in a vector called "dataVec"
		// The "dataVec" vector is finaly stored in testingData or trainingData depending on which dataset it contains.

		testingData.push_back(readData("Data Set/xor_test.csv"));
		trainingData.push_back(readData("Data Set/xor_train.csv"));
	};

	vector<vector<vector<int>>> readData(string filePath)
	{
		// Open the filepath and create a variable for storing the line.
		ifstream file(filePath);
		string line;

		// Stores the wrapper vector
		vector<vector<vector<int>>> dataVec;

		// Remove the first line in the file as it consists of headings.
		getline(file, line);

		// For each row in the file
		while (getline(file, line))
		{
			// Used for keeping track of the first number in a row, which is the label.
			int count = 0;

			// Convert the row to a stringstream object
			stringstream ss(line);

			// String for storing each number in the row
			string lineVal;

			// Store data for each row
			vector<int> labels;
			vector<int> pics;

			// Store the labels and pics vectors together
			vector<vector<int>> wrapper;

			// For each number in the row, using a comma as the delimiter
			while (getline(ss, lineVal, ','))
			{
				// If the number is the first one in the row, store it in the labels vector
				// Labels is stored as a 10 unit vector filled with zeros, with a 1 at the position of the label.
				// This format makes it easier for checking the outputs of the network
				if (count == 0)
				{
					labels.push_back(stoi(lineVal));
				}
				// Otherwise store it in the pics vector
				else
				{
					pics.push_back(stoi(lineVal));
				}
				// Increment the counter so it does not store pics in the label vector
				count++;
			}
			// Store labels and pics in wrapper, and then wrapper in dataVec
			wrapper.push_back(labels);
			wrapper.push_back(pics);
			dataVec.push_back(wrapper);
		}
		// Close the file to save memory
		file.close();
		return dataVec;
	}
};


// Network Class
class Network {

public:

	int netStruct[3];
	vector<MatrixXf> biases;
	vector<MatrixXf> weights;
	vector<vector<vector<vector<int>>>> testingData;
	vector<vector<vector<vector<int>>>> trainingData;

	Network(int structure[])
	{
		// layers is how many layers will be in the network - hardcoded to 3 (includinhg i/o layers)
		// Therefore the network will have one hidden layer
		// netSTruct is an array holding the amount of neurons in each layer
		// netStruct populated to the amount of contents of given parameter
		const int layers = 3;

		for (int i = 0; i < layers; i++)
		{
			netStruct[i] = structure[i];
		}

		// Create instance of loader class
		Loader loader;
		cout << "Loading Data..." << endl;
		loader.loadData(testingData, trainingData);
		cout << "Data Loaded! \n" << endl;
		

		// Create matrices for weights and biases
		MatrixXf b1 = initMatrix(netStruct[1], 1); //Biases of hidden layer neurons
		MatrixXf b2 = initMatrix(netStruct[2], 1); //Biases of output neurons
		MatrixXf w1 = initMatrix(netStruct[1], netStruct[0]); //Weights between input and hidden layer
		MatrixXf w2 = initMatrix(netStruct[2], netStruct[1]); //Weights between hidden and output layer

		// Create array of matrices to hold all weights and biases
		biases = { b1, b2 };
		weights = { w1, w2 };

		// Begin training the network
		cout << "Network has been created! Press ENTER to begin training. \n" << endl;
		cin.get();
		trainNetwork();
	}

	void trainNetwork()
	{
		int epochs;
		float learningRate;

		// Ensure user inputs are of correct datatype
		cout << "Epochs to train for : " << endl;
		while (!(cin >> epochs))
		{
			cout << "Invalid input, please retry.\n";
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			cout << "Epochs to train for : " << endl;;
		}

		cout << "Learning Rate : " << endl;;
		while (!(cin >> learningRate))
		{
			cout << "Invalid input, please retry.\n";
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			cout << "Learning Rate : " << endl;;
		}
		

		// Begin Training
		SDG(epochs, learningRate);

		// End Training
		cout << "\n\n------Training Complete!------\n\n" << endl;
		cout << "Press ENTER to begin testing" << endl;
		cin.get();

		// Begin Testing
		cout << "\n\n------Beginning Testing!------\n\n" << endl;
		testNetwork();
	}

	void testNetwork()
	{
		for (int i = 0; i < testingData[0].size(); i++)
		{
			// Convert the test data to a matrix for the feedforward algorithm
			MatrixXf testData = vecToMat(testingData[0][i][1]);
			MatrixXf expectedResults = vecToMat(testingData[0][i][0]);

			// feed forward the inputs to the output neurons
			MatrixXf results = feedforward(feedforward(testData, biases[0], weights[0], netStruct[1]), biases[1], weights[1], netStruct[2]);

			cout << "\nExpected Results : \n" << expectedResults << endl;
			cout << "Actual Results : \n" << results << "\n" << endl;
		}

		cout << "\n\n------Testing Complete!------\n\n" << endl;
	}
	
	MatrixXf initMatrix(int rows, int cols)
	{
		// Method for initilising matrices with random numbers between 0 and 1
		
		// Create temporary matrix
		MatrixXf mat(rows, cols);

		for (int i = 0; i < rows; i++)
		{
			for (int y = 0; y < cols; y++)
			{
				// Fill each area of matrix with a random number
				mat(i, y) = ((double)rand() / (RAND_MAX));
			}
		}
		return mat;
	}

	MatrixXf vecToMat(vector<int> vec)
	{
		// Map std vector to eigen matrix
		MatrixXf mat(vec.size(), 1);

		for (int i = 0; i < vec.size(); i++)
		{
			mat(i, 0) = vec[i];
		}
		return mat;
	}
	
	MatrixXf feedforward(MatrixXf a, MatrixXf b, MatrixXf w, int neurons) 
	{
		// Feed forward algorithm calculates output of network
		// Matrix a = inputs
		// Matrix b = biases
		// Matrix w = weights
		// int neurons = how many neurons in the given layer

		// Create new matrix of 1 row and 'neuron' number of columns
		MatrixXf result(neurons, 1);
		// Create new matrix holding the dot product of inputs and weights
		MatrixXf dotProd = w * a;
		
		// For each neuron in the current layer
		for (int i = 0; i < neurons; i++)
		{
			// Calculate the sigmoid of (dot(w * a) + b) for each neuron
			result(i, 0) = sigmoid(dotProd(i, 0) + b(i, 0));
		}
		return result;
	}

	void SDG(int epochs, float learningRate)
	{
		// Apply Schotastic Gradient Descent (SDG) to the networks weights and biases
		// This algorithm trains the network incrementally

		// Shuffle the data in the vector for random sampling
		shuffle(begin(trainingData[0]), end(trainingData[0]), default_random_engine());

		for (int i = 0; i < epochs; i++)
		{
			for (int y = 0; y < trainingData[0].size(); y++)
			{
				// Call backpropogation on the training data
				backpropogation(trainingData[0][y][1], trainingData[0][y][0], learningRate);
			}
			cout << "Epoch " << i << " complete." << endl;
		}
	}

	void backpropogation(vector<int> inputs, vector<int> labels, float learningRate)
	{
		// Convert vectors to matrices for matrix math
		MatrixXf inps = vecToMat(inputs);
		MatrixXf labs = vecToMat(labels);

		// Feed Forward
		MatrixXf hiddenResults = feedforward(inps, biases[0], weights[0], netStruct[1]);
		MatrixXf outputResults = feedforward(hiddenResults, biases[1], weights[1], netStruct[2]);

		// Calculate output layer errors
		MatrixXf outputError = labs - outputResults;

		// Calculate hidden layer errors 
		MatrixXf tWeights = weights[1].transpose();
		MatrixXf hiddenError = tWeights * outputError;

		// Calculate hidden -> output delta weights
		MatrixXf outputGradients = calculateGradient(outputResults);
		MatrixXf deltaOutput = calculateDeltaW(hiddenResults, outputGradients, outputError, learningRate);

		// Calculate input -> hidden deltas
		MatrixXf hiddenGradients = calculateGradient(hiddenResults);
		MatrixXf deltaHidden = calculateDeltaW(inps, hiddenGradients, hiddenError, learningRate);

		// Adjust weights by their deltas
		weights[0] = weights[0] + deltaHidden;
		weights[1] = weights[1] + deltaOutput;

		// Adjust biases by their deltas (which is just the gradient)
		biases[0] = biases[0] + hiddenGradients;
		biases[1] = biases[1] + outputGradients;
	}

	MatrixXf calculateGradient(MatrixXf output) 
	{
		// Perform the sigmoid prime function on each element of the given matrix
		int rows = output.rows();
		int cols = output.cols();
		MatrixXf gradient(rows, cols);

		for (int i = 0; i < rows; i++)
		{
			for (int y = 0; y < cols; y++)
			{
				double temp = output(i, y);
				gradient(i, y) = sigmoidPrime(temp);
			}
		}
		return gradient;
	}

	MatrixXf calculateDeltaW(MatrixXf inputs, MatrixXf grads, MatrixXf errors, float learningRate) 
	{
		// Calculate gradients
		MatrixXf hadamard = grads.array() * errors.array(); // elementwise multiplication (hadamard product)
		MatrixXf gradients = hadamard * learningRate;

		// Calculate deltas
		MatrixXf tInputs = inputs.transpose();
		MatrixXf delta = gradients * tInputs;

		return delta;
	}
	
	double sigmoid(double x)
	{
		// Sigmoid function returns sigmoid of given parameter
		return 1 / (1 + exp(-x));
	}

	double sigmoidPrime(double y)
	{
		// Derivative of the sigmoid function
		// Derivative of sigmoid should be:
		// (sigmoid(z) * (1 - sigmoid(z)))
		// However sigmoid has already been applied in the feedforward function

		return y * (1 - y);
	}
};


// Menu Class
class Menu 
{
public:
	void menu() 
	{
		int option;

		cout << "----XOR Neural Network----\n" << endl;
		cout << "[1] Create Network" << endl;
		cout << "[2] Exit Program" << endl;
		cout << "Please select your option: ";
		cin >> option;

		// Error handling if incorrect inputs are given
		while (true)
		{
			if (option == 1)
			{
				cout << "\n\n------Creating Network!------\n\n" << endl;
				int net[3] = { 2, 2, 1 };
				Network n(net);
			}
			else if (option == 2) 
			{
				exit(0);
			}
			else
			{
				cout << "Invalid input, please try again." << endl;
				cout << "Please select your option: ";
				cin >> option;
			}
		}
	}
};


// Main Function
int main()
{
	srand(time(0));

	Menu m;
	m.menu();

	return 0;
}