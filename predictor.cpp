#include "predictor.h"

#include <Eigen/Core>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <numeric>

Predictor::Predictor(const char *traindatapath)
{
    std::ifstream traindata(traindatapath);
    std::string line;
    std::istringstream ss;
    auto nextLine = [&traindata, &line, &ss]() -> void {
        std::getline(traindata, line);
        ss.clear();
        ss.str(line);
    };

    // Read matrix size.
    nextLine();
    ss >> n_inputvec >> n_hid1vec >> n_hid2vec >> n_outputvec;
    assert(n_inputvec == MNISTreader::pixelSize);
    assert(n_outputvec == 10);

    // Read matrix.
    weight1 = Eigen::MatrixXf(n_hid1vec, n_inputvec);
    for(int i=0;i<n_hid1vec;i++){
        nextLine();
        for(int j=0;j<n_inputvec;j++) ss >> weight1(i, j);
    }
    bias1 = Eigen::VectorXf(n_hid1vec);
    nextLine();
    for(int j=0;j<n_hid1vec;j++) ss >> bias1(j);

    // Don't Repeat Yourself!!
    // std::vector<Eigen::MatrixXf> 作るか、
    // layerをクラスにするかした方がよかったかもね。
    weight2 = Eigen::MatrixXf(n_hid2vec, n_hid1vec);
    for(int i=0;i<n_hid2vec;i++){
        nextLine();
        for(int j=0;j<n_hid1vec;j++) ss >> weight2(i, j);
    }
    bias2 = Eigen::VectorXf(n_hid2vec);
    nextLine();
    for(int j=0;j<n_hid2vec;j++) ss >> bias2(j);

    weight3 = Eigen::MatrixXf(n_outputvec, n_hid2vec);
    for(int i=0;i<n_outputvec;i++){
        nextLine();
        for(int j=0;j<n_hid2vec;j++) ss >> weight3(i, j);
    }
    bias3 = Eigen::VectorXf(n_outputvec);
    nextLine();
    for(int j=0;j<n_outputvec;j++) ss >> bias3(j);

    traindata.close();
}

int Predictor::predict(const InputType &input)
{
    assert(input.size() == n_inputvec);
    Eigen::VectorXf inputvec = input2vec(input);

    Eigen::VectorXf vec3 = feedforward(inputvec, 3);

    assert(vec3.size() == 10);
    //std::cerr << vec3.transpose() << std::endl;  // 0〜9の、どれであるかの確率が出力される

    // 確率が最大のものを選んで返す
    int max_idx = 0;
    float max_val = vec3(0);
    for(int i=1;i<=9;i++){
        if(vec3(i) > max_val){
            max_idx = i;
            max_val = vec3(i);
        }
    }
    return max_idx;
}

Eigen::VectorXf Predictor::feedforward(const Eigen::VectorXf &inputvec, int depth)
{
    Eigen::VectorXf vec1(n_hid1vec);
    // Apply rectified linear function
    vec1 = (weight1 * inputvec + bias1).unaryExpr([](float x){ return x > 0.0f ? x : 0.0f; });
    if(depth == 1) return vec1;

    // DON'T REPEAT YOURSELF!!!!!!!!!!!!!
    // 無駄無駄無駄無駄無駄無駄無駄無駄無駄無駄無駄無駄無駄無駄無駄無駄無駄無駄
    Eigen::VectorXf vec2(n_hid2vec);
    vec2 = (weight2 * vec1 + bias2).unaryExpr([](float x){ return x > 0.0f ? x : 0.0f; });
    if(depth == 2) return vec2;

    Eigen::VectorXf vec3(n_outputvec);
    // Apply softmax function
    vec3 = (weight3 * vec2 + bias3).unaryExpr([](float x){ return std::exp(x); });
    float sum = vec3.sum();
    if(sum == 0.0f) sum = 0.0001;
    vec3 /= sum;
    if(depth == 3) return vec3;

    assert("Too much depth!" && false);
    return vec3;
}

Eigen::VectorXf Predictor::input2vec(const InputType &input)
{
    Eigen::VectorXf result(input.size());

    float ave = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    float stddev = 0;
    for(auto v: input){
        float t = v - ave;
        stddev += t * t;
    }
    stddev = std::sqrt(stddev / input.size());

    for(int i=0;i<input.size();i++){
        result(i) = (input[i] - ave) / stddev;
    }

    return result;
}
