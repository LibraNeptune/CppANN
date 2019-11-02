//
// Created by Lincong_Zheng on 31/10/2019.
//

#ifndef RNN_BASIC_CLASS_H
#define RNN_BASIC_CLASS_H

#include <bits/stdc++.h> // Only available when the compiler is g++(or MinGW on Microsoft Windows),
//otherwise, please replace it with specific headers
#include "adept_arrays.h"                    // The main header file here for numeric computation
//#include "Python.h"

using namespace adept;
using namespace std;

// Parameters are defined here
#define batch_num_train 1000
#define batch_num_test 5
#define batch_size 256
#define epoch_num 3
#define input_size 10
#define hidden_size 100
#define output_size 10
// In this case is output size the same as input size and it shouldn't be changed.

// Parameters for Adam optimizer
#define beta_1 0.9
#define beta_2 0.999
#define epsilon 1e-8
#define alpha 5e-5

// General setting
#define grad_seed 1.0          // Do not modify the value if unnecessary
#define pi 3.1415926536

// Functions
#define relu(x) max(0,x)
#define m_Hat(m,t) m/(1-pow(beta_1,t))
#define v_Hat(v,t) v/(1-pow(beta_2,t))
#define adam_update(w,m,v) w-alpha*m/(sqrt(v)+epsilon)
#define linear(x,w,b) matmul(w,x)+b
#define CEL(y,y_hat) sum(-1*(y*log(y_hat)+(1-y)*log(1-y_hat))) // Function of loss: Cross Entropy Loss
#define MSE(y,y_hat) sum(pow(y-y_hat,2)) //Function of loss: Mean Square Error

// Functions  as tools
#define compress(x,n) x+=1,x/=n
#define Suffle(x) shuffle(x.data_pointer(), x.data_pointer() + x.size(),default_random_engine(rand()))

// Dataset
class data{
public:
    Matrix x_train,y_train,x_test,y_test;
    data(Matrix x_train, Matrix y_train, Matrix x_test, Matrix y_test): x_train( x_train),
    y_train(y_train),x_test(x_test),y_test(y_test){};
    ~data(){};
};

// Hyper-Parameters
class parameters{
public:
    int batch_n_train,batch_n_test,batch_s,epoch_n,input_s,hidden_s,output_s;
    double beta_1_,beta_2_,epsilon_,alpha_;// Parameters for Adam optimizer
    Matrix m_w1,v_w1,m_b1,v_b1,m_w2,v_w2,m_b2,v_b2;
    parameters(int batch_n_train, int batch_n_test, int batch_s, int epoch_n, int input_s,
            int hidden_s, int output_s, double beta_1_,double beta_2_,double epsilon_,
            double alpha_,Matrix m_w1,Matrix v_w1,Matrix m_b1,Matrix v_b1,
            Matrix m_w2,Matrix v_w2,Matrix m_b2,Matrix v_b2):batch_n_train(batch_n_train),
            batch_n_test(batch_n_test),batch_s(batch_s),epoch_n(epoch_n),input_s(input_s),
            hidden_s(hidden_s),output_s(output_s),beta_1_(beta_1_),beta_2_(beta_2_),
            epsilon_(epsilon_),alpha_(alpha_),m_w1(m_w1),v_w1(v_w1),m_b1(m_b1),v_b1(v_b1),
            m_w2(m_w2),v_w2(v_w2),m_b2(m_b2),v_b2(v_b2){};
    ~parameters(){};
};

//
class model_param{
public:
    Matrix w1,b1,w2,b2,l_1_v,a_v,l_2_v,g_w1,g_b1,g_w2,g_b2;
    model_param(Matrix w1, Matrix b1, Matrix w2, Matrix b2, Matrix l_v, Matrix a_v,Matrix l_2_v);
    ~model_param(){};
};

//
class optimizer{
public:
    optimizer(){};
    void Adam(model_param mp, parameters Parameters);
    ~optimizer(){};
};

// Shallow neural network
class SNN{
public :
    model_param mp;
    data Data;
    parameters Parameters;
    Matrix label,input;
    Matrix output;
    optimizer Opt;
    SNN(model_param mp, data Data, parameters Parameters, optimizer Opt): mp(mp), Data(Data),
    Parameters(Parameters),Opt(Opt){};
    Matrix forward_propagation(Matrix input);// Forward propagation
    void backward_propagation();
    void train();// Train procession
    void test(double error);
    ~SNN(){};
};

#endif //RNN_BASIC_CLASS_H
