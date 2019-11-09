//
// Created by zsms1 on 05/11/2019.
//

#include "data.h"

//
//data::data(Matrix x_train, Matrix y_train, Matrix x_test, Matrix y_test,Matrix x_valid, Matrix y_valid){
//    this->x_train=x_train;
//    this->y_train=y_train;
//    this->x_test=x_test;
//    this->y_test=y_test;
//    this->x_valid=x_valid;
//    this->y_valid=y_valid;
//}

data::data(Matrix x_train, Matrix y_train, Matrix x_test, Matrix y_test){
    this->x_train=x_train;
    this->y_train=y_train;
    this->x_test=x_test;
    this->y_test=y_test;
}

//
parameters_model::parameters_model(vector<Matrix> weights,vector<Matrix> biases,vector<Matrix> gradients_weight,vector<Matrix>gradients_bias,vector<Matrix> layers){
    this->weights.assign(weights.begin(),weights.end());
    this->biases.assign(biases.begin(),biases.end());
    this->gradients_weight.assign(gradients_weight.begin(),gradients_weight.end());
    this->gradients_bias.assign(gradients_bias.begin(),gradients_bias.end());
    this->layers.assign(layers.begin(),layers.end());
}

