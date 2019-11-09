//
// Created by zsms1 on 05/11/2019.
//

#ifndef RNN_OPTIMIZERS_H
#define RNN_OPTIMIZERS_H

#include "data.h"

#define m_update(m,g) m=beta_1*m+(1-beta_1)*g
#define v_update(v,g) v=beta_2*v+(1-beta_2)*pow(g,2)
#define m_Hat(m,t) m/(1-pow(beta_1,t))
#define v_Hat(v,t) v/(1-pow(beta_2,t))
#define adam_update(w,m,v,lr) w-=lr*m/(sqrt(v)+epsilon)

// Parameters for Adam optimizer
#define beta_1 0.9
#define beta_2 0.999
#define epsilon 1e-8
#define alpha 5e-5

class parameters_optimizer{
public:
    vector<Matrix> m_t_weight,v_t_bias,m_t_bias,v_t_weight;
    parameters_optimizer(vector<Matrix> m_t_weight,vector<Matrix> v_t_weight,vector<Matrix> m_t_bias,vector<Matrix> v_t_bias);
//    parameters_optimizer(){};
    ~parameters_optimizer(){};
};

class optimizer{
public:
    optimizer(){};
    void Adam(parameters_model mp, parameters_optimizer p_o,double step, double learning_rate);
    ~optimizer(){};
};

#endif //RNN_OPTIMIZERS_H
