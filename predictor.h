#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "mnistreader.h"
#include <Eigen/Core>

class Predictor{
public:
    using InputType = MNISTpixels;
    Predictor(const char *traindatapath);
    int predict(const InputType& input);
protected:
    int n_inputvec;
    int n_hid1vec;
    int n_hid2vec;
    int n_outputvec;
    Eigen::MatrixXf weight1;
    Eigen::VectorXf bias1;
    Eigen::MatrixXf weight2;
    Eigen::VectorXf bias2;
    Eigen::MatrixXf weight3;
    Eigen::VectorXf bias3;
    Predictor(){}
    Eigen::VectorXf input2vec(const InputType &input);
    Eigen::VectorXf feedforward(const Eigen::VectorXf &inputvec, int depth);
};

#endif // PREDICTOR_H
