//  Created by Travis Young and Jonathan Lamont on 2/15/15.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <string>
#include <fstream>
#include <time.h>
#include <sys/stat.h>
#include <cstring>

using namespace std;

//globals
int board[30][30];
int check[30][30] = {0}; //associative array to check if a state has been updated
double correlation[15];
double corrAvg[15];
double mutual_info[15];
double mutAvg[15];
double joint_entropy[15];
double jointAvg[15];
double entAvg;
int initNum;
int correlationLength;
int corrLengAvg;

//functions
int is_updated();
double get_entropy(); 
void initBoard(int f);
void get_correlation();
void getCorrelationLength();
int get_lambda();
void get_joint_entropy();
void get_mutual_info();
void printPGM(int f, int stepNum);
double calcDist(int x1, int y1, int x2, int y2);

int main(int argc, const char * argv[])
{
	//declare variables
	double J1, J2, R1, R2, E, L, h, d;//E = entropy, L = correlation length, h = bias
	int f, i, j, x, y;
	double temp, near_cells = 0, far_cells = 0, s, count;
	initNum = 0;

	//initialize averages
	for(i = 0; i < 15; i++){
		corrAvg[i] = 0;
		mutAvg[i] = 0;
		jointAvg[i] = 0;
	}
	entAvg = 0;
	corrLengAvg = 0;

	//seed random time
	srand((int)time(NULL));

	//get input arguments
	if(argc != 7){
		cout << "INPUT ERROR: Expected Trial #, Activation Strength, Inhibition Strength, Activation Range, Inhibition Range, Bias\n";
		exit(1);
	}
	
	//get commandline arguments
	f = atoi(argv[1]); //experiment #
	J1 = atof(argv[2]); //activation interaction strength
	J2 = atof(argv[3]); //inhibition interaction strength
	R1 = atof(argv[4]); //activation interaction range
	R2 = atof(argv[5]); //inhibition interaction range
	h = atof(argv[6]);

	//make directory for experiment
	string expNum = "./Trial" + to_string(f);
	if(mkdir(expNum.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0){
		cerr << "error creating directory\n";
		exit(1);
	}
	
	//open csv file
	string csvName = expNum + "/experiment" + to_string(f) + ".csv"; 
	ofstream csvFile(expNum);
	csvFile.open(csvName);

	csvFile << "Jonathan Lamont\n";
	csvFile << "Activation/Inhibition Cellular Automaton\n\n";
	csvFile << "Trial Number " << f << "\n";
	csvFile << "J1: " << J1 << ",J2: " << J2 << ",R1: " << R1 << ",R2: " << R2 << ",h = " << h;
	csvFile << "\n\n";

	initBoard(f);
	
	//allow the board to converge
	int converge = 1;//this variable tells when the program is done updating
	int prevVal;
	double distance;
	double newTotal;
	int counter = 0;
	int c2 = 0;
	int numExperiments; 	
	
	for(numExperiments = 0; numExperiments < 4; numExperiments++){
		initBoard(f);
		converge = 1;
		while(converge == 1){
			converge = 0;
			//set/reset check array
			for(i = 0; i < 30; i++){
				for(j = 0; j < 30; j++){
					check[i][j] = 0;
				}
			}
			while(is_updated() != 1){//while not all cells have been updated
				x = rand() % 30;
				y = rand() % 30;	
				if(check[x][y] != 1){//if cell has not been updated
					check[x][y] = 1;

					prevVal = board[x][y];

					near_cells = 0;
					far_cells = 0;	
					//calculates distance and adds to near and far cells arrays
					for(i = 0; i < 30; i++){
						for(j = 0; j < 30; j++){
							distance = calcDist(x,y,i,j);
							if(distance <= R1){
								near_cells += board[i][j];
							}
							else if((distance >= R1) && (distance < R2)){
								far_cells += board[i][j];
							}
						}
					}

					//calculate new state of cell
					newTotal = h + (near_cells * J1) + (far_cells * J2);
					if(newTotal >= 0){ 
						board[x][y] = 1;
					}
					else{
						board[x][y] = -1;
					}
					if(board [x][y] != prevVal){
						converge = 1; 
					}
				}
			}
			printPGM(f, counter);
			counter++;
		}//after convergence	

		//get values
		E = get_entropy();
		get_correlation();
		getCorrelationLength();
		get_joint_entropy();
		get_mutual_info();

		//update averages	
		for(i = 0; i < 15; i++){
			corrAvg[i] += correlation[i];
			mutAvg[i] += mutual_info[i];
			jointAvg[i] += joint_entropy[i];
		}
		entAvg += E;
		corrLengAvg += correlationLength;

		//print variables
		csvFile << "Iteration:," << numExperiments << "\n";
		csvFile << "Entropy:," << E << "\n";
		csvFile << "Corr Length:," << correlationLength << "\n";
		csvFile << "Distance,Correlation,Joint Entropy,Mutual Info\n";
		for(i = 0; i < 15; i++){
			csvFile << i << "," << correlation[i] << ","
				<< joint_entropy[i] << "," << mutual_info[i]<< "\n";
		}
		
		csvFile << "\n\n";

	}

	//update averages	
	for(i = 0; i < 15; i++){
		corrAvg[i] = corrAvg[i] / (numExperiments);
		mutAvg[i] = mutAvg[i] / (numExperiments);
		jointAvg[i] = jointAvg[i] / (numExperiments);
	}
	entAvg = entAvg / numExperiments;
	corrLengAvg = corrLengAvg / numExperiments;

	//print averagess
	csvFile << "Averages\n";
	csvFile << "Entropy:," << entAvg << "\n";
	csvFile << "Corr Length:," << corrLengAvg << "\n";
	csvFile << "Distance,Correlation,Joint Entropy,Mutual Info\n";
	for(i = 0; i < 15; i++){
		csvFile << i << "," << corrAvg[i] << ","
			<< jointAvg[i] << "," << mutAvg[i]<< "\n";
	}
	csvFile << "\n\n";

	csvFile.close();
	return 0;
}

//creates a random initial board
void initBoard(int f){
	int i, j, temp;
	for(i = 0; i < 30; i++){
		for(j = 0; j < 30; j++){
			temp = rand() % 2;
			if(temp == 1) board[i][j] = temp;
			else board[i][j] = -1;
		}
	}
	printPGM(f, -1);//prints initial board
}

//returns 1 if the entire board has been updated
int is_updated(){
	for(int i = 0; i < 30; i++){
		for(int j = 0; j < 30; j++){
			if(check[i][j] == 0) return 0;
		}
	}
	return 1;
}

double calcDist(int x1, int y1, int x2, int y2){
	double absoluteX;
	double absoluteY;

	absoluteX = abs(x1 - x2);
	absoluteY = abs(y1 - y2);

	if(absoluteX > 15){ absoluteX = 30 - absoluteX; }
	if(absoluteY > 15){ absoluteY = 30 - absoluteY; }

	return (absoluteX + absoluteY);
}

double get_entropy(){
	int i, j;
	double pr_pos, pr_neg, B = 0;

	for(i = 0; i < 30; i++){
		for(j = 0; j < 30; j++){
			B += (1 + board[i][j])/2;
		}
	}
	pr_pos = (1/pow(30,2))*B;
	pr_neg = 1 - pr_pos;
	if(pr_pos == 0) B = -(pr_neg*log(pr_neg));
	else if(pr_neg == 0) B = -(pr_pos*log(pr_pos));
	else B = -(pr_pos*log(pr_pos)+pr_neg*log(pr_neg));
	return B;
}

void get_correlation(){
	int l, i, j, i2, j2;
	double iSum, ijSum, firTerm, secTerm;
	for(l = 0; l <= 14; l++){//all distances
		iSum = 0; ijSum = 0;
		if(l == 0){//simplified formula for distance = 0
			for(i = 0; i < 30; i++){
				for(j = 0; j < 30; j++){
					iSum += board[i][j];
				}
			}
			correlation[l] = abs(1 - pow(((1/(30*30))*iSum),2));
		}
		else{//all distances other than 
			for(i = 0; i < 30; i++){
				for(j = 0; j < 30; j++){
					iSum += board[i][j];
					for(i2 = i; i2 < 30; i2++){
						for(j2 = j; j2 < 30; j2++){
							if(calcDist(i,j,i2,j2) == l){
								ijSum += (board[i][j] * board[i2][j2]);
							}
						}
					}
				}
			}
			firTerm = (2. / (30*30*4*l)) * ijSum;
			secTerm = pow(((1. / (30*30)) * iSum), 2);
			correlation[l] = abs(firTerm - secTerm);
		}
	}
}

void getCorrelationLength(){
	double val, possI, possF = 100000;
	int index;
	val = correlation[0]/2.71828;
	
	for(int i = 1; i < 14; i++){
		possI = abs(val - correlation[i]);
		if(possI < possF){
			possF = possI;
			index = i;
		}
	}
	correlationLength = index;
}

void get_joint_entropy(){
	double firstSum, secondSum , firstTerm, secondTerm, thirdTerm, forthTerm;
	int ix,iy, jx,jy;
	double p11, p1_1, p_11,p_1_1;

	for( int l =1; l<15; l++)
	{
		firstSum =0; secondSum =0;
		for( int i=0; i <900 ;++i)
		{  ix = i /30;
			iy= i%30;
			for(int j=i; j<900; ++j)
			{   jx = j/30;
				jy = j%30;
				if(l == calcDist(ix , iy , jx,jy))
				{
					firstSum += (((board[ix][iy] + 1)/2.0) * ((board[jx][jy] + 1)/2.0));
					secondSum += (((-1* board[ix][iy] + 1)/2.0) * ((-1*board[jx][jy] + 1)/2.0));
				}
			}
		}


		p11 = (2.0/(30*30*4*l)) * firstSum;
		p_1_1 = (2.0/(30*30*4*l)) * secondSum;
		p_11 = p1_1 = 1- p11 - p_1_1;


		if(p_1_1 == 0) secondTerm =0; else secondTerm = (p_1_1 * (log(p_1_1)/log(2.0)));
		if(p1_1 == 0) thirdTerm = 0; else thirdTerm = (p1_1 * (log(p1_1)/log(2.0)));
		if(p11 == 0) forthTerm =0; else forthTerm = (p11 * (log(p11)/log(2.0)));

		joint_entropy[l] = -1*( secondTerm  + thirdTerm  + forthTerm );
	}
}

void get_mutual_info(){
	int k = 0;
	for(k = 0; k < 15; k++)
		mutual_info[k] = abs(2 * get_entropy() - joint_entropy[k]);
}

void printPGM(int f, int stepNum){

	string expNum = "./Trial" + to_string(f);
	string expName = expNum + "/experiment" + to_string(f) + "Step" + to_string(stepNum) + ".pgm";
	if(stepNum == -1) expName = expNum + "/experiment" + to_string(f) + "Initial" + to_string(initNum) + ".pgm";
	initNum++;
	ofstream pgmFile;
	pgmFile.open(expName);

	//pgm header
	pgmFile << "P2\n";
	pgmFile << "30 30\n";
	pgmFile << "255\n";

	//writing contents to pgm file
	for(int i = 0; i < 30; i++){
		for(int j = 0; j < 30; j++){
			if(board[i][j] == 1){ 
				pgmFile << "0 ";
			}
			else if(board[i][j] == -1){ 
				pgmFile << "255 ";
			}
		}
	}

	pgmFile.close();	
}
