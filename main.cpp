//
// Created by Lincong_Zheng on 31/10/2019.
//

#include "adept_source.h"              //Necessary if no linking to the static library
#include "models.h"

// Main function
int main() {
// Other basic setting
    struct timeb startTime , endTime;
    // Set for random number
    srand((unsigned)time(NULL));
    //Initialization of Adam optimizer
    r_m rm;
    vector<Matrix> m_t_weight{rm.zero_matrix(hidden_size_1 ,input_size),rm.zero_matrix(hidden_size_2,hidden_size_1),rm.zero_matrix(output_size,hidden_size_2)},
            v_t_weight{rm.zero_matrix(hidden_size_1 ,input_size),rm.zero_matrix(hidden_size_2,hidden_size_1),rm.zero_matrix(output_size,hidden_size_2)},
            v_t_bias{rm.zero_matrix(hidden_size_1 ,batch_size),rm.zero_matrix(hidden_size_2,batch_size),rm.zero_matrix(output_size,batch_size)},
            m_t_bias{rm.zero_matrix(hidden_size_1 ,batch_size),rm.zero_matrix(hidden_size_2,batch_size),rm.zero_matrix(output_size,batch_size)};

    parameters_optimizer p_o(m_t_weight,v_t_weight,m_t_bias,v_t_bias);
// Convert values to basic parameters
    parameters_basic p_b(batch_num_train, batch_num_test, batch_size, epoch_num, input_size, hidden_size_1,hidden_size_2, output_size);

// Preparation of training data
    Matrix x_train(input_size,batch_size*batch_num_train);
    x_train<<range(0,input_size*batch_size*batch_num_train);
    compress(x_train,50*x_train.size())+0.5;
    Suffle(x_train);                    // Suffle the matrix
    Matrix y_train=pow(x_train,2)+0.005;

// Preparation of testing data
    Matrix x_test(input_size,batch_size*batch_num_test);
    x_test<<range(0,input_size*batch_size*batch_num_test);
    compress(x_test,5*x_test.size())+0.5;
    Suffle(x_test);                             // Suffle the matrix
    Matrix y_test=pow(x_test,2)+0.005;

// Assign value to dataset
    data Data(x_train,y_train,x_test,y_test);

// Initialization of weight and bias with an extremely small number
    Matrix Weight_1=rm.Guassian_matrix(0,1,hidden_size_1 ,input_size);
    Matrix Bias_1=rm.Guassian_matrix(0,1,hidden_size_1,batch_size);
    Matrix Weight_2=rm.Guassian_matrix(0,1,hidden_size_2,hidden_size_1);
    Matrix Bias_2=rm.Guassian_matrix(0,1,hidden_size_2,batch_size);
    Matrix Weight_3=rm.Guassian_matrix(0,1,output_size,hidden_size_2);
    Matrix Bias_3=rm.Guassian_matrix(0,1,output_size,batch_size);
    vector<Matrix> weights{Weight_1,Weight_2,Weight_3},biases{Bias_1,Bias_2,Bias_3};
    vector<Matrix> layers{Matrix(hidden_size_1,batch_size),Matrix(hidden_size_1,batch_size),Matrix(hidden_size_2,batch_size),Matrix(hidden_size_2,batch_size),Matrix(output_size,batch_size)};

    vector<Matrix> gradients_weight{Matrix(hidden_size_1 ,input_size),Matrix(hidden_size_2,hidden_size_1),Matrix(output_size,hidden_size_2)};
    vector<Matrix> gradients_bias{Matrix(hidden_size_1,batch_size),Matrix(hidden_size_2,batch_size),Matrix(output_size,batch_size)};

    parameters_model p_m(weights,biases,gradients_weight,gradients_bias,layers);
    optimizer Opt;
    SNN snn(p_m,Data,p_b,Opt,p_o);

    ftime(&startTime);
    snn.train();
    ftime(&endTime);
    auto millisecond=(endTime.time-startTime.time)*1000 + (endTime.millitm - startTime.millitm);
    cout << "Training time: (" << floor(millisecond/60000)<<") minutes  ("<<
    long(floor(millisecond/1000))%60 <<") seconds  ("<<millisecond%1000<< ") milliseconds" << endl;
    snn.test(1e-1);

    cout<<"real data: "<<snn.label(__,0)<<endl;
    cout<<"predict data: "<<snn.output(__,0);

    return 0;
}