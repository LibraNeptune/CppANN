//
// Created by Lincong_Zheng on 31/10/2019.
//

#ifndef RNN_MODELS_H
#define RNN_MODELS_H

//#include "Python.h"
#include "optimizers.h"
#include "functions.h"
//#include "Python.h"

// Parameters are defined here
#define batch_num_train 300
#define batch_num_test 5
#define batch_size 128
#define epoch_num 150
#define learning_rate 5e-3
#define input_size 10
#define hidden_size_1 40
#define hidden_size_2 50
#define output_size 10
// In this case is output size the same as input size and it shouldn't be changed.

//#define  layer_num 3
//#define linear_layer_num 2

// Shallow neural network
class SNN{
public :
    parameters_model p_m;
    data Data;
    parameters_basic p_b;
    Matrix label,input;
    Matrix output;
    optimizer Opt;
    parameters_optimizer p_o;
    SNN(parameters_model p_m, data Data, parameters_basic p_b, optimizer Opt,parameters_optimizer p_o):p_m(p_m),Data(Data),p_b(p_b),Opt(Opt),p_o(p_o){};
    Matrix forward_propagation(Matrix input);// Forward propagation
    void backward_propagation();
    void train();// Train procession
    void test(double error);
    ~SNN(){};
};

#endif //RNN_MODELS_H
