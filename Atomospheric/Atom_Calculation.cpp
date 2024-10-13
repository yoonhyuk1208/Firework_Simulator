#include<iostream>
#include<algorithm>
#include<vector>
#define in(a,b,c) (a)>=(b)&&(a)<=(c)

#define F first
#define S second

using namespace std;

extern "C" {

    const double R = 287.05;
    const double Mol_mass = 28.97;
    const double Den = 1.0;
    const double Tem = 286.15;

    vector<vector<pair<double,double>>> Atom; // density, T
    vector<vector<pair<double,double>>> Atom_Temp;

    void Creat_Atom(int x, int y) {

        vector<pair<double,double>> Temp(y, {Den, Tem});
        Atom.clear();
        Atom_Temp.clear();
        for(int i=0; i<x; i++) {
            Atom.push_back(Temp);
            Atom_Temp.push_back(Temp);
        }
    }

    void Explosion(double powder, double comH, int x, int y){

        double Q = comH*powder; // 열, 밀도 계산
        double delta_T = Q/(1005.0*Atom[x][y].F*0.25);

        Atom[x][y].F *= Atom[x][y].S/(Atom[x][y].S+delta_T);
        Atom[x][y].S += delta_T;

        // 빛 계산
    }

    void Copy_AtomTemp(int x, int y){
        for(int i=0; i<x; i++)for(int j=0; j<y; j++){
            Atom_Temp[i][j] = Atom[i][j];
        }
    }

    void Update_Atom(int x, int y, double FPS){
        Copy_AtomTemp(x, y);

        int dx[] = {1,-1,0,0}, dy[] = {0,0,1,-1};
        double D = 0.1;
        double Alp = 0.1;

        for(int i=0; i<x; i++)for(int j=0; j<y; j++){
            double SumOfDen = 0;
            double SumOfTem = 0;

            for(int k=0; k<4; k++){
                int ii = i + dx[k], jj = j + dy[k];
                if(in(ii, 0, x-1) && in(jj, 0, y-1)){
                    SumOfDen += Atom_Temp[ii][jj].F;
                    SumOfTem += Atom_Temp[ii][jj].S;
                }
                else{
                    SumOfDen += Den;
                    SumOfTem += Tem;
                }
            }

            SumOfDen -= 4 * Atom[i][j].F * FPS;
            SumOfTem -= 4 * Atom[i][j].S * FPS;
            Atom[i][j].F += D * SumOfDen;
            Atom[i][j].S += Alp * SumOfTem;
        }
    }
}