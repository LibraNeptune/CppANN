//
// Created by zsms1 on 05/11/2019.
//

#ifndef RNN_INITIALIZATION_H
#define RNN_INITIALIZATION_H

#include <bits/stdc++.h> // Only available when the compiler is g++(or MinGW on Microsoft Windows),
//otherwise, please replace it with specific headers
#include "adept_arrays.h" // The main header file here for numeric computation

using namespace adept;
using namespace std;

#define seed chrono::system_clock::now().time_since_epoch().count()

// Random matrix
class r_m{
public:
    r_m(){};
    Matrix Guassian_matrix(double mean, double stddev,  double d1, double d2);
    Matrix zero_matrix(double d1, double d2);
    ~r_m(){};
};

#endif //RNN_INITIALIZATION_H
