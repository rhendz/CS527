//neural net with back propigation

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> 
#include <fstream>

using namespace std;

struct Connection{
	double weight;
	double deltaWeight;
};
class Neuron;
typedef vector<Neuron> Layer;

/*****************NEURON DEFINITION********************/
class Neuron{
	public:
		Neuron(unsigned numOutputs, unsigned index);
		void setOutputVal(double val) { outputVal = val; }
		double getOutputVal(void) const { return outputVal; }
		void feedForward(const Layer &prevLayer);
		void calcOutputGradients(double targetVal);
		void calcHiddenGradients(const Layer &nextLayer);
		void updateInputWeights(Layer &prevLayer);

	private:
		static double eta; // overall learning rate [0.0 ... 1.0]
		static double alpha; // momentum, or the multiplier of the last weight change [0.0 ... n]
		static double transferFunction(double x);
		static double transferFunctionDerivative(double x);
		static double randomWeight(void){return rand()/ double(RAND_MAX);}//generates random number 0 < x < 1
		double sumDOW(const Layer &nextLayer);
		double outputVal;//neuron value
		vector<Connection> outputWeights;//contains weight and change in weight
		unsigned myIndex;
		double gradient;
};

double Neuron::eta = .15;
double Neuron::alpha = .5;

void Neuron::updateInputWeights(Layer &prevLayer){
	//update neuron weights for proceeding layer
	for(unsigned n = 0; n < prevLayer.size(); ++n){
		Neuron &neuron = prevLayer[n]; //shortcut to previous layer's neuron
		double oldDeltaWeight = neuron.outputWeights[myIndex].deltaWeight;

		//individual input, magnified by the gradient and training rate
		double newDeltaWeight = 
			eta //overall learning rate
			* neuron.getOutputVal()
			* gradient
			* alpha //momentum = a fraction of the previous delta weight
			* oldDeltaWeight;
		neuron.outputWeights[myIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[myIndex].weight += newDeltaWeight;
	}

}

double Neuron::sumDOW(const Layer &nextLayer){
	double sum = 0.0;
	
	//sum of contributions to errors of nodes we feed
	for(unsigned n = 0; n < nextLayer.size() - 1; ++n){
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
	double dow = sumDOW(nextLayer);
	gradient = dow * Neuron::transferFunctionDerivative(outputVal);

}
void Neuron::calcOutputGradients(double targetVal){
	double delta = targetVal - outputVal;
	gradient = delta * Neuron::transferFunctionDerivative(outputVal);
}

//calculate new value of neuron based on sum of previous layer
double Neuron::transferFunction(double x){
	//tanh (output range [-1.0, ... , 1.0]
	return tanh(x);
}

//necessary for backpropigation learning
double Neuron::transferFunctionDerivative(double x){
	return 1.0 - x * x;
}

//initialize neuron with random weights for each of its outputs
Neuron::Neuron(unsigned numOutputs, unsigned index){
	//for all connections
	for(unsigned c = 0; c < numOutputs; ++c){
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();//initialize each neuron with random weight
	}

	myIndex = index;
}

void Neuron::feedForward(const Layer &prevLayer){
	double sum = 0.0;

	//sums the previous layer's outputs and bias value
	for(unsigned n = 0; n < prevLayer.size(); ++n){
		sum += prevLayer[n].getOutputVal() * outputWeights[myIndex].weight;
	}

	//calculate neuron's value
	outputVal = Neuron::transferFunction(sum);

}

/*****************NEURAL NET DEFINITION****************/
class Net{
	public:
		Net(const vector<unsigned> &topology);

		void feedForward(const vector<double> &inputVals);
		void backProp(const vector<double> &targetVals);
		void getResults(vector<double> &resultsVals) const;	

	private:
		vector<Layer> layers; //layers[layerNum][neuronNum]
		double error;
		double recentAverageError;
		double recentAverageSmoothingFactor;
};

void Net::getResults(vector<double> &resultVals) const{	
	//clear prior data
	resultVals.clear();

	//for all neurons in output layer
	for(unsigned n = 0; n < layers.back().size() - 1; ++n){
		resultVals.push_back(layers.back()[n].getOutputVal());
	}
	
}

//fills neural net with necessary number of layers with neurons with random weights
Net::Net(const vector<unsigned> &topology){
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		//create layer
		layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		//fill with neurons and biases
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
			layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Made a Neuron!\n";
		}
		
		//update bias neuron's value
		layers.back().back().setOutputVal(1.0);
	}

}

void Net::feedForward(const vector<double> &inputVals){
	assert(inputVals.size() == layers[0].size() - 1);

	//assign inputVals to input neurons
	for(unsigned i = 0; i < inputVals.size(); ++i){
		layers[0][i].setOutputVal(inputVals[i]);
	}

	//forward propigation//
	//for each layer
	for(unsigned layerNum = 1; layerNum < layers.size(); ++layerNum){
		Layer &prevLayer = layers[layerNum - 1];//creates reference to previous layer
		//for each neuron in the layer (except bias neuron)
		for(unsigned n = 0; n < layers[layerNum].size() - 1; ++n){
			//each Neuron feeds forward using preceeding layer's outputs
			layers[layerNum][n].feedForward(prevLayer);
		}
	}

}

void Net::backProp(const vector<double> &targetVals){

	//calculate overall net error (Root Mean Square Error)
	Layer &outputLayer = layers.back();
	error = 0.0;
	//for each neuron in the output layer (except bias)
	for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		error += delta * delta;
	}
	error /= outputLayer.size() - 1;//average error
	error = sqrt(error);//root

	//implement a recent average measurement
	recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

	//calculate output layer gradients
	//for all neurons except bias
	for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//calculate gradients on hidden layers
	//starting with last hidden layer until first hidden layer
	for(unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum){
		Layer &hiddenLayer = layers[layerNum];
		Layer &nextLayer = layers[layerNum + 1];

		//for all neurons in hidden layer
		for(unsigned n = 0; n < hiddenLayer.size(); ++n){
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	//for all layers from outputs to first hidden layer, update connection weights
	for(unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum){
		Layer &layer = layers[layerNum];
		Layer &prevLayer = layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1; ++n){
			layer[n].updateInputWeights(prevLayer);
		}

	}

}


/*****************MAIN FUNCTION****************/

int main(int argc, char** argv){
	
	int numLayers;
	double learningRate;
	vector<int> numNeurLayer;
	int numEpochs;
	int numInputs;
	
	numLayers = atoi(argv[1]);
	if(argc != (8 + numLayers)){
		cerr << "Incorrect input, expected:\n";
		cerr << "Number_of_Hidden_Layers Number_of_Neurons_In_Layer0...N LearningRate TrainingFile TestingFile ValidationFile";
		cerr << " Number_of_Epochs Number_of_Input\n";
		cerr << "Process Terminating\n";
		exit(1);
	}
	int i, j = 2;
	for(i = 0; i < numLayers; i++){
		numNeurLayer.push_back(atoi(argv[j]));
		j++;
	}
	learningRate = atof(argv[j]); 
	j++;
	ifstream trainingFile(argv[j]);
	if(!trainingFile.is_open()){
		cerr << "Error opening training file; program terminating\n";
		exit(1);
	}
	j++;
	ifstream testingFile(argv[j]);
	if(!testingFile.is_open()){
		cerr << "Error opening testing file; program terminating\n";
		exit(1);
	}
	j++;
	ifstream validationFile(argv[j]);
	if(!validationFile.is_open()){
		cerr << "Error opening validation file; program terminating\n";
		exit(1);
	}
	j++;
	numEpochs = atoi(argv[j]);
	j++;
	numInputs = atoi(argv[j]);

/*
	vector<unsigned> topology;
	Net myNet(topology);

	vector<double> inputVals;
	myNet.feedForward(inputVals);

	vector<double> targetVals;
	myNet.backProp(targetVals);

	vector<double> resultVals;
	myNet.getResults(resultVals);
*/
	trainingFile.close();
	testingFile.close();
	validationFile.close();

	return 0;
}
