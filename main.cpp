//
// Created by Lincong_Zheng on 31/10/2019.
//

#include "adept_source.h"              //Necessary if no linking to the static library
#include "basic_class.h"

// Main function
int main() {
// Other basic setting
    struct timeb startTime , endTime;
    // Set for random number
    srand((unsigned)time(NULL));
    //Initialization of Adam optimizer
    Matrix m_w1(hidden_size ,input_size);
    m_w1=0;
    Matrix m_b1(hidden_size ,batch_size);
    m_b1=0;
    Matrix m_w2(output_size,hidden_size);
    m_w2=0;
    Matrix m_b2(output_size ,batch_size);
    m_b2=0;

    Matrix v_w1(hidden_size ,input_size);
    v_w1=0;
    Matrix v_b1(hidden_size ,batch_size);
    v_b1=0;
    Matrix v_w2(output_size,hidden_size);
    v_w2=0;
    Matrix v_b2(output_size ,batch_size);
    v_b2=0;

// Convert value to parameters
    parameters Parameters(batch_num_train, batch_num_test, batch_size, epoch_num, input_size, hidden_size, output_size,beta_1,beta_2,epsilon,alpha,m_w1,v_w1,m_b1,v_b1,m_w2,v_w2,m_b2,v_b2);

// Preparation of training data
    Matrix x_train(input_size,batch_size*batch_num_train);
    x_train<<range(0,input_size*batch_size*batch_num_train);
    compress(x_train,50*x_train.size());
    Suffle(x_train);                    // Suffle the matrix
    Matrix y_train=abs(sin(2*pi*50*x_train));

// Preparation of testing data
    Matrix x_test(input_size,batch_size*batch_num_test);
    x_test<<range(0,input_size*batch_size*batch_num_test);
    compress(x_test,5*x_test.size());
    Suffle(x_test);                             // Suffle the matrix
    Matrix y_test=abs(sin(2*pi*50*x_test));

// Assign value to dataset
    data Data(x_train,y_train,x_test,y_test);

// Initialization of weight and bias with an extremely small number
    Matrix Weight_1(hidden_size ,input_size);
    Weight_1<<range(0,hidden_size*input_size);
    compress(Weight_1,1e7*Weight_1.size());
    Suffle(Weight_1);

    Matrix Bias_1(hidden_size,batch_size);
    Bias_1<<range(0,hidden_size*batch_size);
    compress(Bias_1,1e6*Bias_1.size());
    Suffle(Bias_1);

    Matrix Weight_2(output_size ,hidden_size);
    Weight_2<<range(0,output_size *hidden_size);
    compress(Weight_2,1e7*Weight_2.size());
    Suffle(Weight_2);

    Matrix Bias_2(output_size,batch_size);
    Bias_2<<range(0,output_size*batch_size);
    compress(Bias_2,1e6*Bias_2.size());
    Suffle(Bias_2);

    Matrix Linear_1_v(hidden_size,batch_size);
    Matrix Activation_v(hidden_size,batch_size);
    Matrix Linear_2_v(output_size,batch_size);

    model_param m_p(Weight_1,Bias_1,Weight_2,Bias_2,Linear_1_v,Activation_v,Linear_2_v);
    optimizer Opt;

    SNN snn(m_p,Data,Parameters,Opt);
    ftime(&startTime);
    snn.train();
    ftime(&endTime);
    auto millisecond=(endTime.time-startTime.time)*1000 + (endTime.millitm - startTime.millitm);
    cout << "Training time: (" << floor(millisecond/60000)<<") minutes  ("<<
    long(floor(millisecond/1000))%60 <<") seconds  ("<<millisecond%1000<< ") milliseconds" << endl;
    snn.test(1e-4);
    return 0;
}