//
// Created by zsms1 on 05/11/2019.
//

#ifndef RNN_DATA_H
#define RNN_DATA_H

#include "initialization.h"

using namespace adept;
using namespace std;

// General setting
#define grad_seed 1.0          // Do not modify the value if unnecessary
#define pi 3.1415926536

// Dataset
class data{
public:
//    Matrix x_train,y_train,x_test,y_test,x_valid,y_valid;
    Matrix x_train,y_train,x_test,y_test;
//    data(Matrix x_train, Matrix y_train, Matrix x_test, Matrix y_test, Matrix x_valid, Matrix y_valid);
    data(Matrix x_train, Matrix y_train, Matrix x_test, Matrix y_test);
    ~data(){};
};

// Hyper-Parameters
class parameters_basic{
public:
    int batch_n_train,batch_n_test,batch_s,epoch_n,input_s,hidden_s1,hidden_s2,output_s;
    parameters_basic(int batch_n_train, int batch_n_test, int batch_s, int epoch_n, int input_s,int hidden_s1,int hidden_s2, int output_s):batch_n_train(batch_n_train),batch_n_test(batch_n_test),batch_s(batch_s),epoch_n(epoch_n),input_s(input_s),hidden_s1(hidden_s1),hidden_s2(hidden_s2),output_s(output_s){};
    ~parameters_basic(){};
};

//
class parameters_model{
public:
    vector<Matrix> weights,biases,gradients_weight,gradients_bias,layers;
    parameters_model(vector<Matrix> weights,vector<Matrix> biases,vector<Matrix> gradients_weight,vector<Matrix>gradients_bias,vector<Matrix> layers);
    ~parameters_model(){};
};

#endif //RNN_DATA_H
