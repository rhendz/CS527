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
int correlation[15];
int mutual_info[15];
int joint_entropy[15];

//functions
int is_updated(); //done
double get_entropy(); //done
void get_correlation();
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

	//seed random time
	srand((int)time(NULL));

	//initialize the board
	for(i = 0; i < 30; i++){
		for(j = 0; j < 30; j++){
			temp = rand() % 2;
			if(temp == 1) board[i][j] = temp;
			else board[i][j] = -1;
		}
	}

	//get input arguments
	if(argc != 7){
		cout << "INPUT ERROR: Expected Experiment #, Activation Strength, Inhibition Strength, Activation Range, Inhibition Range, Bias\n";
		exit(1);
	}
	f = atoi(argv[1]); //experiment #
	J1 = atof(argv[2]); //activation interaction strength
	J2 = atof(argv[3]); //inhibition interaction strength
	R1 = atof(argv[4]); //activation interaction range
	R2 = atof(argv[5]); //inhibition interaction range
	h = atof(argv[6]);

	string expNum = "./Experiment" + to_string(f);
	if(mkdir(expNum.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0){
		cerr << "error creating directory\n";
		exit(1);
	}
	
	string csvName = expNum + "/experiment" + to_string(f) + ".csv"; 
	ofstream csvFile(expNum);
	csvFile.open(csvName);

	csvFile << "Jonathan Lamont\n\n\n";
	csvFile << "Activation/Inhibition Cellular Automaton\n";
	csvFile << "Experiment Number " << f << "\n";
	csvFile << "J1: " << J1 << ",J2: " << J2 << ",R1: " << R1 << ",R2: " << R2 << ",h = " << h;

	printPGM(f, -1);//prints initial board
	
	//allow the board to converge
	int converge = 1;//this variable tells when the program is done updating
	int prevVal;
	double distance;
	double newTotal;
	int counter = 0;
	int c2 = 0;
	
	
	for(int numExperiments = 0; numExperiments < 4; numExperiments++){
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

	}

	get_correlation();	
	for(i = 0; i < 14; i++){
		cout << correlation[i] << " ";
	}	

	csvFile.close();
	return 0;
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

void get_correlation(){
	int i = 0, j = 0, i2 = 0, j2 = 0, k = 0;
	int sumij = 0, sumi = 0;
	double d = 0;


	for(k = 0; k < 14; k++){
		for(i = 0; i < 30; i++){
			for(j = 0; j < 30; j++){
				sumi += board[i][j]; //calculate second summation in Correlation formula
				for(i2 = 0; i2 < 30; i2++){
					for(j2 = 0; j2 < 30; j2++){
						//avoid double counting cells
						if(i == i2){
							if(j2 > j){
								d = abs(i - i2) + abs(j - j2);
							}
						}
						else d = abs(i - i2) + abs(j - j2);
						if(d == k) sumij += board[i][j] * board[i2][j2];
					}
				}
			}
		}
		if(k == 0) correlation[k] = 1 - (pow((1/(30*30)) * sumi, 2)); 
		else correlation[k] = 2/(30 * 30 * 4 * k) * sumij - (pow((1/(30*30)) * sumi, 2));
	}
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

void get_joint_entropy(){
	int i = 0, j = 0, i2 = 0, j2 = 0, k = 0;
	int ePos = 0, eNeg = 0, pPos = 0, pNeg = 0, P = 0;

	for(k = 0; k < 14; k++){
		for(i = 0; i < 30; i++){
			for(j = 0; j < 30; j++){
				for(i2 = 0; i2 < 30; i2++){
					for(j2 = 0; j2 < 30; j2++){
						//avoid double counting cells
						if(i == i2){
							if(j2 > j){
								ePos = ((board[i][j] + 1) / 2) * ((board[i2][j2] + 1) / 2);
								eNeg = -((board[i][j] + 1) / 2) * -((board[i2][j2] + 1) / 2);

							}
						}
						else{
							ePos = ((board[i][j] + 1) / 2) * ((board[i2][j2] + 1) / 2);
							eNeg = -((board[i][j] + 1) / 2) * -((board[i2][j2] + 1) / 2);

						}
					}
				}
			}
		}
		pPos = (2/(pow(30, 2) * 4 * k)) * ePos;
		pNeg = (2/(pow(30, 2) * 4 * k)) * eNeg;
		P = 1 - pPos - pNeg;
		//prevet log of 0
		if(pPos == 0) joint_entropy[k] = -(pNeg*log(pNeg) + P*log(P));
		else if(pNeg == 0) joint_entropy[k] = -(pPos*log(pPos) + P*log(P));
		else joint_entropy[k] = -(pPos*log(pPos) + pNeg*log(pNeg) + P*log(P));
	}

}

void get_mutual_info(){
	int k = 0;
	for(k = 0; k < 14; k++)
		mutual_info[k] = 2 * get_entropy() - joint_entropy[k];
}


void printPGM(int f, int stepNum){
	
	string expNum = "./Experiment" + to_string(f);
	string expName = expNum + "/experiment" + to_string(f) + "Step" + to_string(stepNum) + ".pgm";
	if(stepNum == -1) expName = expNum + "/experiment" + to_string(f) + "Initial.pgm";
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
