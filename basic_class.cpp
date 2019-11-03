//
// Created by Lincong_Zheng on 31/10/2019.
//
#include "basic_class.h"

Stack gradient_stack; // Object to store differential statements. It has to show above the active data

model_param::model_param(Matrix w1, Matrix b1, Matrix w2, Matrix b2, Matrix l_v, Matrix a_v,Matrix l_2_v){
    this->w1=w1;
    this->b1=b1;
    this->w2=w2;
    this->b2=b2;
    this->l_1_v=l_1_v;
    this->a_v=a_v;
    this->l_2_v=l_2_v;
}

void optimizer::Adam(model_param mp, parameters Parameters,double step){
    m_update(Parameters.m_w1,mp.g_w1);
    v_update(Parameters.v_w1,mp.g_w1);
    auto m_hat_w1=m_Hat(Parameters.m_w1,step);
    auto v_hat_w1=v_Hat(Parameters.v_w1,step);
    adam_update(mp.w1,Parameters.m_w1,Parameters.v_w1);

    m_update(Parameters.m_b1,mp.g_b1);
    v_update(Parameters.v_b1,mp.g_b1);
    auto m_hat_b1=m_Hat(Parameters.m_b1,step);
    auto v_hat_b1=v_Hat(Parameters.v_b1,step);
    adam_update(mp.b1,Parameters.m_b1,Parameters.v_b1);

    m_update(Parameters.m_w2,mp.g_w2);
    v_update(Parameters.v_w2,mp.g_w2);
    auto m_hat_w2=m_Hat(Parameters.m_w2,step);
    auto v_hat_w2=v_Hat(Parameters.v_w2,step);
    adam_update(mp.w2,Parameters.m_w2,Parameters.v_w2);

    m_update(Parameters.m_b2,mp.g_b2);
    v_update(Parameters.v_b2,mp.g_b2);
    auto m_hat_b2=m_Hat(Parameters.m_b2,step);
    auto v_hat_b2=v_Hat(Parameters.v_b2,step);
    adam_update(mp.b2,Parameters.m_b2,Parameters.v_b2);
}

Matrix SNN::forward_propagation(Matrix input){
    mp.l_1_v=linear(input, mp.w1, mp.b1);
    mp.a_v= relu(mp.l_1_v);
    mp.l_2_v= linear(mp.a_v,mp.w2,mp.b2);
    return mp.l_2_v;
}

// Backword propagation
void SNN::backward_propagation(){
    aMatrix w1,b1,w2,b2;
    w1=mp.w1;
    b1=mp.b1;
    w2=mp.w2;
    b2=mp.b2;

    gradient_stack.new_recording();      // Clear any existing differential statements
    adouble s1=MSE(label,linear(mp.a_v,w2,b2));
//    adouble s1=CEL(label,linear(mp.a_v,w2,b2));
    s1.set_gradient(grad_seed);
    gradient_stack.reverse();
    mp.g_w2=w2.get_gradient();
    mp.g_b2=b2.get_gradient();

    gradient_stack.new_recording();
    adouble s2=MSE(label,linear(relu(linear(input,w1,b1)),w2,b2));
//    adouble s2=CEL(label,linear(relu(linear(input,w1,b1)),w2,b2));
    s2.set_gradient(grad_seed);
    gradient_stack.reverse();
    mp.g_w1=w1.get_gradient();
    mp.g_b1=b1.get_gradient();
}

// Training
void SNN::train(){
//    cout<<"point_0"<<endl;
//    Py_Initialize();
//    PyRun_SimpleString("import numpy");
//    PyRun_SimpleString("import visdom");
//    PyRun_SimpleString("vis = visdom.Visdom()");
//    cout<<"point_1"<<endl;

// Modify the path if you hope to save the data of loss in your PC, or you can comment it
    ofstream fout_0("C:/Users/zsms1/Desktop/loss.txt");
    ofstream fout_1("C:/Users/zsms1/Desktop/loss_full.txt");

    auto batch_s=Parameters.batch_s;
    double step=0;

    for (int i=0;i<Parameters.epoch_n;++i){
        for (int j=0;j<Parameters.batch_n_train;++j){
//            cout<<"point_2"<<endl;
            input=Data.x_train(__,range(batch_s*j,batch_s*j+batch_s-1));
            label=Data.y_train(__,range(batch_s*j,batch_s*j+batch_s-1));
            output=forward_propagation(input);
            auto loss=MSE(label,output);
//            auto loss=CEL(label,output);
            if(j%20==0) cout<<"epoch "<<i<<", batch "<<j<<": "<<loss/batch_size<<endl;
//            cout<<"point_3"<<endl;
// Save data for further plotting in Python
            fout_0<<loss/batch_size<<endl;
            fout_1<<"epoch "<<i<<", batch "<<j<<": "<<loss/batch_size<<endl;
//            PyRun_SimpleString("loss=numpy.loadtxt('C:/Users/zsms1/Desktop/loss.txt')");
//            PyRun_SimpleString("vis.line(X=numpy.array([loss[1]]),"
//                               "Y=numpy.column_stack((numpy.array([loss[0]]))),"
//                               "win='window_1',update='append',"
//                               "opts=dict(showlegend=True,legend=['loss']))");
            backward_propagation();
            Opt.Adam(mp,Parameters,++step);
//            cout<<"point_4"<<endl;
        }
    }
    fout_0<<flush;
    fout_0.close();
    fout_1<<flush;
    fout_1.close();
//    Py_Finalize();
}

// Test module
void SNN::test(double error){
    double accurate_num=0,tmp;
    auto batch_s=Parameters.batch_s;
    for(int i=0;i<Parameters.batch_n_test;++i){
        input=Data.x_test(__,range(batch_s*i,batch_s*i+batch_s-1));
        label=Data.y_test(__,range(batch_s*i,batch_s*i+batch_s-1));
        Matrix compare=label-error<=input<=label+error;
        accurate_num+=sum(compare);
        tmp=label.size();
    }
    double total=Parameters.batch_n_test*tmp;
    cout<<"Result of test"<<endl;
    cout<<"accuracy: ("<<accurate_num/total<<") under an error of ("<<error<<")"<<endl;
}



