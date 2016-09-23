#include "mnistreader.h"
#include "predictor.h"

#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <stdexcept>
#include <cstdlib>

#include <fenv.h>

#include <Eigen/Core>

class Trainee : protected Predictor{
public:
    using AnswerType = int;
    Trainee(int n_hid1, int n_hid2, float init_sigma);
    void train(std::vector<std::pair<InputType, AnswerType>> minibatch, float learning_rate);
    bool dump(const char *traindatapath);
private:
    /* For AdaGrad */
    Eigen::ArrayXXf gsq_w1;
    Eigen::ArrayXf gsq_b1;
    Eigen::ArrayXXf gsq_w2;
    Eigen::ArrayXf gsq_b2;
    Eigen::ArrayXXf gsq_w3;
    Eigen::ArrayXf gsq_b3;
};

Trainee::Trainee(int n_hid1, int n_hid2, float init_sigma)
{
    n_inputvec = MNISTreader::pixelSize;
    n_hid1vec = n_hid1;
    n_hid2vec = n_hid2;
    n_outputvec = 10;

    gsq_w1 = Eigen::ArrayXXf::Zero(n_hid1vec, n_inputvec);
    gsq_b1 = Eigen::ArrayXf::Zero(n_hid1vec);
    gsq_w2 = Eigen::ArrayXXf::Zero(n_hid2vec, n_hid1vec);
    gsq_b2 = Eigen::ArrayXf::Zero(n_hid2vec);
    gsq_w3 = Eigen::ArrayXXf::Zero(n_outputvec, n_hid2vec);
    gsq_b3 = Eigen::ArrayXf::Zero(n_outputvec);

    std::mt19937 mt((std::random_device())());
    std::normal_distribution<float> nd(0.0, init_sigma);

    weight1 = Eigen::MatrixXf(n_hid1vec, n_inputvec);
    for(int i=0;i<n_hid1vec;i++) for(int j=0;j<n_inputvec;j++) weight1(i, j) = nd(mt);
    bias1 = Eigen::VectorXf::Zero(n_hid1vec);

    weight2 = Eigen::MatrixXf(n_hid2vec, n_hid1vec);
    for(int i=0;i<n_hid2vec;i++) for(int j=0;j<n_hid1vec;j++) weight2(i, j) = nd(mt);
    bias2 = Eigen::VectorXf::Zero(n_hid2vec);

    weight3 = Eigen::MatrixXf(n_outputvec, n_hid2vec);
    for(int i=0;i<n_outputvec;i++) for(int j=0;j<n_hid2vec;j++) weight3(i, j) = nd(mt);
    bias3 = Eigen::VectorXf::Zero(n_outputvec);
}

void Trainee::train(std::vector<std::pair<InputType, AnswerType>> minibatch, float learning_rate)
{
    Eigen::MatrixXf dweight3 = Eigen::MatrixXf::Zero(n_outputvec, n_hid2vec);
    Eigen::VectorXf dbias3 = Eigen::VectorXf::Zero(n_outputvec);
    Eigen::MatrixXf dweight2 = Eigen::MatrixXf::Zero(n_hid2vec, n_hid1vec);
    Eigen::VectorXf dbias2 = Eigen::VectorXf::Zero(n_hid2vec);
    Eigen::MatrixXf dweight1 = Eigen::MatrixXf::Zero(n_hid1vec, n_inputvec);
    Eigen::VectorXf dbias1 = Eigen::VectorXf::Zero(n_hid1vec);

    /* For AdaGrad */
    auto fn = [](float lhs, float rhs) -> float { return lhs != 0.0 ? lhs / rhs : 0.0; };
    /*
    Eigen::ArrayXXf gsq_w1 = Eigen::ArrayXXf::Zero(n_hid1vec, n_inputvec);
    Eigen::ArrayXf gsq_b1 = Eigen::ArrayXf::Zero(n_hid1vec);
    Eigen::ArrayXXf gsq_w2 = Eigen::ArrayXXf::Zero(n_hid2vec, n_hid1vec);
    Eigen::ArrayXf gsq_b2 = Eigen::ArrayXf::Zero(n_hid2vec);
    Eigen::ArrayXXf gsq_w3 = Eigen::ArrayXXf::Zero(n_outputvec, n_hid2vec);
    Eigen::ArrayXf gsq_b3 = Eigen::ArrayXf::Zero(n_outputvec);
    auto fn = [](float lhs, float rhs) -> float { return lhs != 0.0 ? lhs / rhs : 0.0; };
    */

    for(auto sample: minibatch){
        Eigen::VectorXf inputvec = input2vec(sample.first);
        Eigen::VectorXf z1 = feedforward(inputvec, 1);
        Eigen::VectorXf z2 = feedforward(inputvec, 2);  // 後付けとはいえ。この計算、あからさまに無駄だな。z1からz2を計算すべき。

        // Calculate delta of output layer.
        Eigen::VectorXf delta3;
        delta3 = feedforward(inputvec, 3);
        delta3(sample.second) -= 1.0f;
        {
            Eigen::ArrayXXf e = delta3 * z2.transpose();
            gsq_w3 += e * e;
            gsq_b3 += delta3.array() * delta3.array();
            dweight3 += e.matrix(); //e.binaryExpr(gsq_w3.sqrt(), fn).matrix();
            dbias3 += delta3; //delta3.array().binaryExpr(gsq_b3.sqrt(), fn).matrix();
        }

        // Calculate delta of 2nd hidden layer.
        Eigen::VectorXf delta2 = Eigen::VectorXf::Zero(n_hid2vec);
        for(int j=0;j<n_hid2vec;j++){
            for(int k=0;k<n_outputvec;k++) delta2(j) += delta3(k) * weight3(k, j) * (z2(j) >= 0.f ? 1.f : 0.f);
        }
        {
            Eigen::ArrayXXf e = delta2 * z1.transpose();
            gsq_w2 += e * e;
            gsq_b2 += delta2.array() * delta2.array();
            dweight2 += e.matrix(); //e.binaryExpr(gsq_w2.sqrt(), fn).matrix();
            dbias2 += delta2; //delta2.array().binaryExpr(gsq_b2.sqrt(), fn).matrix();
        }

        // Calculate delta of 1st hidden layer.
        Eigen::VectorXf delta1 = Eigen::VectorXf::Zero(n_hid1vec);
        for(int j=0;j<n_hid1vec;j++){
            for(int k=0;k<n_hid2vec;k++) delta1(j) += delta2(k) * weight2(k, j) * (z1(j) >= 0.f ? 1.f : 0.f);
        }
        {
            Eigen::ArrayXXf e = delta1 * inputvec.transpose();
            gsq_w1 += e * e;
            gsq_b1 += delta1.array() * delta1.array();
            dweight1 += e.matrix(); //e.binaryExpr(gsq_w1.sqrt(), fn).matrix();
            dbias1 += delta1; //delta1.array().binaryExpr(gsq_b1.sqrt(), fn).matrix();
        }
    }
    weight1 -= dweight1.binaryExpr(gsq_w1.sqrt().matrix(), fn) * learning_rate / minibatch.size();
    bias1 -= dbias1.binaryExpr(gsq_b1.sqrt().matrix(), fn) * learning_rate / minibatch.size();
    weight2 -= dweight2.binaryExpr(gsq_w2.sqrt().matrix(), fn) * learning_rate / minibatch.size();
    bias2 -= dbias2.binaryExpr(gsq_b2.sqrt().matrix(), fn) * learning_rate / minibatch.size();
    weight3 -= dweight3.binaryExpr(gsq_w3.sqrt().matrix(), fn) * learning_rate / minibatch.size();
    bias3 -= dbias3.binaryExpr(gsq_b3.sqrt().matrix(), fn) * learning_rate / minibatch.size();
    /*
    weight1 -= dweight1 * learning_rate / minibatch.size();
    bias1 -= dbias1 * learning_rate / minibatch.size();
    weight2 -= dweight2 * learning_rate / minibatch.size();
    bias2 -= dbias2 * learning_rate / minibatch.size();
    weight3 -= dweight3 * learning_rate / minibatch.size();
    bias3 -= dbias3 * learning_rate / minibatch.size();
    */

}

bool Trainee::dump(const char *traindatapath)
{
    std::ofstream dat(traindatapath);

    if(!dat.good()) return false;
    dat << n_inputvec << ' ' << n_hid1vec << ' ' << n_hid2vec << ' ' << n_outputvec << '\n';

    int i, j;
    for(i=0;i<n_hid1vec;i++){
        for(j=0;j<n_inputvec-1;j++) dat << weight1(i, j) << ' ';
        dat << weight1(i, j) << '\n';
    }
    for(j=0;j<n_hid1vec-1;j++) dat << bias1(j) << ' ';
    dat << bias1(j) << '\n';

    for(i=0;i<n_hid2vec;i++){
        for(j=0;j<n_hid1vec-1;j++) dat << weight2(i, j) << ' ';
        dat << weight2(i, j) << '\n';
    }
    for(j=0;j<n_hid2vec-1;j++) dat << bias2(j) << ' ';
    dat << bias2(j) << '\n';

    for(i=0;i<n_outputvec;i++){
        for(j=0;j<n_hid2vec-1;j++) dat << weight3(i, j) << ' ';
        dat << weight3(i, j) << '\n';
    }
    for(j=0;j<n_outputvec-1;j++) dat << bias3(j) << ' ';
    dat << bias3(j) << '\n';

    if(!dat.good()) return false;
    return true;
}

void train(const char *imagespath, const char *labelspath, int n_hid1, int n_hid2, float sigma, float epsilon)
{
    MNISTreader reader(imagespath, labelspath);
    Trainee trainee(n_hid1, n_hid2, sigma);

    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    if(reader.isError()){
        std::string errstring;

        switch(reader.getError()){
        case MNISTreaderError::imagesFileIOFailure:
            errstring = "Failed to open ";
            errstring += imagespath;
            break;
        case MNISTreaderError::labelsFileIOFailure:
            errstring = "Failed to open ";
            errstring += labelspath;
            break;
        case MNISTreaderError::imagesFileMalformed:
        case MNISTreaderError::unexpectedPixelsSize:
            errstring = "Broken or invalid file ";
            errstring += imagespath;
            break;
        case MNISTreaderError::labelsFileMalformed:
            errstring = "Broken or invalid file ";
            errstring += labelspath;
            break;
        case MNISTreaderError::differentNumberOfImage:
            errstring = imagespath;
            errstring += " and ";
            errstring += labelspath;
            errstring += " are not corresponding";
            break;
        }
        throw std::invalid_argument(errstring);
    }
    int n_trained = 0;
    while(1){
        auto minibatch = reader.minibatch(50);
        if(minibatch.size() == 0) break;
        trainee.train(minibatch, epsilon);
        n_trained += 50;
        //if(n_trained > 2000) break;
        std::cout << '\r' << n_trained << "/" << reader.length() << std::flush;
    }
    std::cout << std::endl;
    trainee.dump("traindata");
}

int main(int argc, char **argv)
{
    float sigma = 0.2;
    float epsilon = 0.01;
    int n_hid1 = 10;
    int n_hid2 = 10;

    if(argc == 2) epsilon = std::atof(argv[1]);
    else if(argc >= 3){
        n_hid1 = std::atoi(argv[1]);
        n_hid2 = std::atoi(argv[2]);
        if(argc >= 4) epsilon = std::atof(argv[3]);
        if(argc >= 5) sigma = std::atof(argv[4]);
    }
    train("train-images-idx3-ubyte", "train-labels-idx1-ubyte", n_hid1, n_hid2, sigma, epsilon);
    return 0;
}
