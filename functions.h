//
// Created by zsms1 on 05/11/2019.
//

#ifndef RNN_FUNCTIONS_H
#define RNN_FUNCTIONS_H

//#include "bits/stdc++.h"
//using namespace std;

// Functions
#define relu(x) max(0,x)
//#define tanh_(x) tanh(x)

#define linear(x,w,b) matmul(w,x)+b
#define CEL(y,y_hat) sum(-1*(y*log(y_hat)+(1-y)*log(1-y_hat))) // Function of loss: Cross Entropy Loss
#define MSE(y,y_hat) sum(pow(y-y_hat,2)) //Function of loss: Mean Square Error

// Functions  as tools
#define compress(x,n) x+=1,x/=n
#define Suffle(x) shuffle(x.data_pointer(), x.data_pointer() + x.size(),default_random_engine(rand()))

#endif //RNN_FUNCTIONS_H
