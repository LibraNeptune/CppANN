//
// Created by zsms1 on 05/11/2019.
//
#include "optimizers.h"

parameters_optimizer::parameters_optimizer(vector<Matrix> m_t_weight,vector<Matrix> v_t_weight,vector<Matrix> m_t_bias,vector<Matrix> v_t_bias){
    this->m_t_weight.assign(m_t_weight.begin(),m_t_weight.end());
    this->v_t_weight.assign(v_t_weight.begin(),v_t_weight.end());
    this->m_t_bias.assign(m_t_bias.begin(),m_t_bias.end());
    this->v_t_bias.assign(v_t_bias.begin(),v_t_bias.end());
};

void optimizer::Adam(parameters_model mp, parameters_optimizer p_o,double step, double learning_rate=alpha){
    for(int i=0;i<p_o.m_t_bias.size();++i){
    // Weights
        m_update(p_o.m_t_weight[i],mp.gradients_weight[i]);
        v_update(p_o.v_t_weight[i],mp.gradients_weight[i]);
        adam_update(mp.weights[i],m_Hat(p_o.m_t_weight[i],step),v_Hat(p_o.v_t_weight[i],step),learning_rate);
    // Biases
        m_update(p_o.m_t_bias[i],mp.gradients_bias[i]);
        v_update(p_o.v_t_bias[i],mp.gradients_bias[i]);
        adam_update(mp.biases[i],m_Hat(p_o.m_t_bias[i],step),v_Hat(p_o.v_t_bias[i],step),learning_rate);
    }
}
