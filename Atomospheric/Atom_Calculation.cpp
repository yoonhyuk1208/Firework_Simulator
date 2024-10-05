#include<iostream>
#include<algorithm>
#include<vector>

#define F first
#define S second

using namespace std;

extern "C" {

    const double R = 287.05;
    const double Mol_mass = 28.97;

    vector<vector<pair<double,double>>> Atom; // density, T

    void Creat_Atom(int x, int y) {
        vector<pair<double,double>> Temp(y, {1.0, 286.15});
        Atom.clear();
        for(int i=0; i<x; i++) {
            Atom.push_back(Temp);
        }
    }

    void Explosion(double powder, double comH, int x, int y){
        double Q = comH*powder; // 열, 밀도 계산
        double delta_T = Q/(1005.0*Atom[x][y].F);

        Atom[x][y].F *= Atom[x][y].S/(Atom[x][y].S+delta_T);
        Atom[x][y].S += delta_T;
    }
}
