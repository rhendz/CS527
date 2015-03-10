//Jonathan Lamont
//3/9/2015

#include <time.h>
#include <stdlib.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;


//globals
const int numPatterns = 20;
vector<vector<int> > pattern;
vector<vector<double> > weight;
int permutation[100];
vector<vector<double> > basinSize;
vector<int> aveStable(50, 0);
vector<double> aveUnstable(50, 0);

//functions
int randPlusMinus1();																	//randomly returns + or - 1
void initPatterns(int p);															//initializes p patterns onto pattern vector
void calcWeight(int p);																//finds the weights for p patterns onto weight vector
int testPatterns(int p);															//tests p patterns and returns number of stable patterns
double calcUnstablePercent(int numStable, int numP);	//calculates unstable percent based on stable and pattern counters 

void randPermutation();																//randomly shuffles permutation global
int calcBasinSize(int p);


/************************
    MAIN FUNCTION
************************/

int main(){
	srand(time(NULL));

	//initialize basinSize
	basinSize.resize(50);
	for(int i = 0; i < basinSize.size(); i++){
		basinSize[i].resize(51);
		for(int j = 0; j < basinSize[i].size(); j++){
			basinSize[i][j] = 0;
		}
	}

	//file output init
	ofstream csvFile;
	csvFile.open("results.csv");

	csvFile << "Jonathan Lamont\n" << "Hopfield Network\n" << "CS527\n\n";
	csvFile << "# Patterns,Unstable Fraction,Stable Imprints\n";

	//calculations
	double unstablePercent;
	int numStable;

	//majority of the driver code
	for(int runs = 0; runs < 10; runs++){
		csvFile << "Run," << runs + 1 << "\n";
		for(int p = 0; p < 50; p++){
			initPatterns(p);
			calcWeight(p);	
			numStable = testPatterns(p);
			aveStable[p] += numStable;
			unstablePercent = calcUnstablePercent(numStable, p + 1);
			aveUnstable[p] += unstablePercent;
			printf("p: %2d, numStable: %2d, unstablePercent: %.2f\n", p + 1, numStable, unstablePercent);
			csvFile << p+1 << "," << unstablePercent <<	"," << numStable << "\n";
		}
		csvFile << "\n\n";
	}

	//outputs averages
	csvFile << "Averages\n";
	for(int p = 0; p < 50; p++){
		csvFile << p+1 << "," << aveUnstable[p]/10. << "," << aveStable[p]/10 << "\n";
	}

	//output basinSize
	csvFile << "\n\n# Patterns,vs,Basin Size\n";
	csvFile << " ,";
	for(int i = 0; i < 50; i++){
		csvFile << i + 1 << ",";
	}
	csvFile << "\n";

	//normalize basinSize
	for(int i = 0; i < basinSize.size(); i++){
		for(int j = 0; j < basinSize[i].size(); j++){
			if(*max_element(basinSize[i].begin(), basinSize[i].end()) != 0){
			basinSize[i][j] = (basinSize[i][j] - *min_element(basinSize[i].begin(), basinSize[i].end()))/(*max_element(basinSize[i].begin(), basinSize[i].end()) - *min_element(basinSize[i].begin(), basinSize[i].end()));
			}
		}
	}

	for(int i = 0; i < basinSize.size(); i++){
		csvFile << i + 1 << ",";
		for(int j = 0; j < basinSize[i].size(); j++){
			csvFile << basinSize[i][j] << ",";
		}
		csvFile << "\n";
	}

	csvFile.close();
	return 0;
}

/************************
	FUNCTION IMPLEMENTATION
 ************************/

//randomly returns + or - 1
int randPlusMinus1(){
	if(rand()%2 == 1) return 1;
	else return -1;
}

//initializes p patterns onto pattern vector
void initPatterns(int p){
	pattern.resize(p + 1);
	for(int i = 0; i < pattern.size(); i++){
		pattern[i].resize(100);
		for(int j = 0; j < pattern[i].size(); j++){
			pattern[i][j] = randPlusMinus1();
		}
	}
}

//finds the weights for p patterns onto weight vector
void calcWeight(int p){
	weight.resize(100);
	double neurSum;
	for(int i = 0; i < weight.size(); i++){
		weight[i].resize(100);
		for(int j = 0; j < weight[i].size(); j++){
			if(i != j){
				for(int k = 0; k <= p; k++){
					weight[i][j] += pattern[k][i] * pattern[k][j];
				}
				weight[i][j] = weight[i][j]/100.;
			}
			else weight[i][j] = 0;
		}
	}
}

//tests p patterns and returns number of stable patterns
int testPatterns(int p){
	double locField;
	int newState, isStable;
	int numStable = 0;
	int numUnstable = 0;

	//for all patterns
	for(int k = 0; k <= p; k++){
		isStable = 1;
		//for each neuron
		for(int i = 0; i < pattern[k].size(); i++){
			//for each corresponding neuron in the network
			locField = 0;
			for(int j = 0; j < pattern[k].size(); j++){
				locField += (weight[i][j] * pattern[k][j]);
			}

			//find the new state based on sigmoid function
			if(locField < 0){
				newState =  -1;
			}
			else newState = 1;

			if(newState != pattern[k][i]){
				isStable = 0;
				continue;
			}
		}
		if(isStable == 1){
			numStable++;
			//				cout << "basin size: " << calcBasinSize(k) << "\n";
			basinSize[p][calcBasinSize(k)]++;
		}	
		else{
			numUnstable++;
		}	
	}

	return numStable;
}

double calcUnstablePercent(int numStable, int numP){
	int numUnstable = numP - numStable;
	double percentUnstable;
	percentUnstable = (double) numUnstable / numP;
	return percentUnstable;
}

void randPermutation(){																//randomly shuffles permutation global
	for(int i = 0; i < 100; i++){
		permutation[i] = i;
	}
	random_shuffle(begin(permutation), end(permutation));
}

int calcBasinSize(int p){
	randPermutation();
	vector<int> neuralCopy;
	neuralCopy.resize(100);
	double locField;	

	//copy pattern into neuralCopy
	for(int i = 0; i < pattern[p].size(); i++){
		neuralCopy[i] = pattern[p][i];
	}

	for(int i = 0; i < 50; i++){
		if(neuralCopy[permutation[i]] == 1){
			neuralCopy[permutation[i]] = -1;
		}
		else neuralCopy[permutation[i]] = 1;

		//ten iterations of updates
		for(int j = 0; j < 10; j++){
			for(int k = 0; k < 100; k++){
				locField = 0;
				for(int l = 0; l < 100; l++){
					locField += weight[k][l] * neuralCopy[l];
				}
				if(locField < 0) neuralCopy[k] = -1;
				else neuralCopy[k] = 1;				
			}
		}

		//compare neuralCopy and original pattern
		for(int j = 0; j < pattern[p].size(); j++){
			if(pattern[p][j] != neuralCopy[j]){
				return i + 1;//if basin can result in 51, remove +1
			}

		}

	}
	return 50;	
}
