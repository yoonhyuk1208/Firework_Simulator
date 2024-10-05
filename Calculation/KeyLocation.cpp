#include<iostream>
#include<map>
using namespace std;

extern "C" {
    int return_locate(char A) {
        char Alpha_arr[33] = {'q','a','z','w','s','x','e','d','c','r','f','v','t','g','b','y','h','n','u','j','m','i','k',',','o','l','.','p',';','/','[','\'',']'};
        map<char,int> hash;
        for(int i=0; i<33; i++){
            hash.insert({Alpha_arr[i], i});
        }

        if(hash.find(A) != hash.end()){
            return hash[A];
        }
        else{
            return -1;
        }
    }
}