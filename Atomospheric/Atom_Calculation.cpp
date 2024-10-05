#include<iostream>
#include<algorithm>
#include<vector>

#define F first
#define S second

using namespace std;

extern "C" {

    const double R = 8.314;
    const double Mol_mass = 28.97;

    vector<vector<pair<double,double>>> Atom; // density, T

    void Creat_Atom(int x, int y) {
        vector<pair<double,double>> Temp(y, {1.0, 286.15});
        Atom.clear();
        for(int i=0; i<x; i++) {
            Atom.push_back(Temp);
        }
    }

}
