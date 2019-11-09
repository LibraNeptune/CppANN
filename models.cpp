//
// Created by Lincong_Zheng on 31/10/2019.
//
#include "models.h"

Stack gradient_stack; // Object to store differential statements. It has to show above the active data

//
Matrix SNN::forward_propagation(Matrix input){
    p_m.layers[0]=linear(input, p_m.weights[0], p_m.biases[0]);
    p_m.layers[1]= relu(p_m.layers[0]);
    p_m.layers[2]= linear(p_m.layers[1],p_m.weights[1],p_m.biases[1]);
    p_m.layers[3]=tanh(p_m.layers[2]);
    p_m.layers[4]=linear(p_m.layers[3],p_m.weights[2],p_m.biases[2]);
    return p_m.layers[4];
}

// Backword propagation
void SNN::backward_propagation(){
    vector<aMatrix> weights(p_m.weights.begin(),p_m.weights.end()),biases(p_m.biases.begin(),p_m.biases.end());
    gradient_stack.new_recording();      // Clear any existing differential statements
    adouble s1=MSE(label,linear(tanh(linear(p_m.layers[1],p_m.weights[1],p_m.biases[1])),weights[2],biases[2]));
//    adouble s1=CEL(label,linear(p_m.layers[1],weights[1],biases[1]));
    s1.set_gradient(grad_seed);
    gradient_stack.reverse();
    p_m.gradients_weight[1]=weights[1].get_gradient();
    p_m.gradients_bias[1]=biases[1].get_gradient();
//
    gradient_stack.new_recording();
    adouble s2=MSE(label,linear(tanh(linear(relu(linear(input, p_m.weights[0], p_m.biases[0])),p_m.weights[1],p_m.biases[1])),weights[2],biases[2]));
//    adouble s2=CEL(label,linear(relu(linear(input,w1,b1)),w2,b2));
    s2.set_gradient(grad_seed);
    gradient_stack.reverse();
    p_m.gradients_weight[0]=weights[0].get_gradient();
    p_m.gradients_bias[0]=biases[0].get_gradient();

    gradient_stack.new_recording();      // Clear any existing differential statements
    adouble s3=MSE(label,linear(p_m.layers[3],weights[2],biases[2]));
//    adouble s1=CEL(label,linear(p_m.layers[1],weights[1],biases[1]));
    s3.set_gradient(grad_seed);
    gradient_stack.reverse();
    p_m.gradients_weight[2]=weights[2].get_gradient();
    p_m.gradients_bias[2]=biases[2].get_gradient();
}

// Training
void SNN::train(){
//    Py_Initialize();
//    PyRun_SimpleString("import numpy");
//    PyRun_SimpleString("import visdom");
//    PyRun_SimpleString("vis = visdom.Visdom()");

// Modify the path if you hope to save the data of loss in your PC, or you can comment it
    ofstream fout_0("C:/Users/zsms1/Desktop/loss.txt");
    ofstream fout_1("C:/Users/zsms1/Desktop/loss_full.txt");
//
    auto batch_s=p_b.batch_s;
    double step=0,loss;
//
    int i,j;
//    reduction(+:) private(input,output,label)
//#pragma omp parallel for reduction(+:loss) private(input,output,label)
    for (i=0;i<p_b.epoch_n;++i){
//        #pragma omp parallel for private(j)
        for (j=0;j<p_b.batch_n_train;++j){
            input=Data.x_train(__,range(batch_s*j,batch_s*j+batch_s-1));
            label=Data.y_train(__,range(batch_s*j,batch_s*j+batch_s-1));
            output=forward_propagation(input);
            loss=MSE(label,output);
//            auto loss=CEL(label,output);
            if(j%20==0){
                printf("epoch %5d | batch %5d | loss %.9f\n",i+1,j+1,loss/batch_size);
                for(int k=0;k<44;++k) {
                    if(k==12||k==26) printf("%c",'|');
                    else printf("%c",'-');
                }
                printf("\n");
            }
// Save data for further plotting in Python
            fout_0<<loss/batch_size<<endl;
            fout_1<<"epoch "<<i+1<<", batch "<<j+1<<": "<<loss/batch_size<<endl;
//            PyRun_SimpleString("loss=numpy.loadtxt('C:/Users/zsms1/Desktop/loss.txt')");
//            PyRun_SimpleString("vis.line(X=numpy.array([loss[1]]),"
//                               "Y=numpy.column_stack((numpy.array([loss[0]]))),"
//                               "win='window_1',update='append',"
//                               "opts=dict(showlegend=True,legend=['loss']))");
            backward_propagation();
            Opt.Adam(p_m,p_o,++step,learning_rate);
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
    auto batch_s=p_b.batch_s;
    for(int i=0;i<p_b.batch_n_test;++i){
        input=Data.x_test(__,range(batch_s*i,batch_s*i+batch_s-1));
        output=forward_propagation(input);
        label=Data.y_test(__,range(batch_s*i,batch_s*i+batch_s-1));
        Matrix compare=label-error<=output<=label+error;
        accurate_num+=sum(compare);
        tmp=label.size();
    }
    double total=p_b.batch_n_test*tmp;
    cout<<"Result of test"<<endl;
    cout<<"accuracy: ("<<accurate_num/total<<") under an error of ("<<error<<")"<<endl;
}