//
// Created by zsms1 on 05/11/2019.
//

#include "initialization.h"

// Guassian matrix
default_random_engine generator (seed);
Matrix r_m::Guassian_matrix(double mean, double stddev, double d1, double d2){
    Matrix random_matrix(d1,d2);
    normal_distribution<double> distribution (0.0,0.1);
    distribution.reset();
    for (int i=0; i<d1; ++i)
        for (int j = 0; j < d2; ++j)
            random_matrix(i,j)=distribution(generator);
    return random_matrix;
}

Matrix r_m::zero_matrix(double d1, double d2){
    Matrix zero(d1,d2);
    return zero=0;
}