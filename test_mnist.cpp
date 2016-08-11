#include "mnistreader.h"
#include "predictor.h"

#include <iostream>
#include <fstream>
#include <cstdint>

void test_mnist(const char *imagespath, const char *labelspath, const char *traindatapath)
{
    MNISTreader reader(imagespath, labelspath);

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

    Predictor predictor(traindatapath);
    std::ofstream dump_result("t10ktest-labels-idx1-ubyte", std::ios::binary);
    int t;
    t = 0x01080000;
    dump_result.write(reinterpret_cast<char*>(&t), 4);
    t = ((10000 & 0xFF) << 24) | ((10000 & 0xFF00) << 8);
    dump_result.write(reinterpret_cast<char*>(&t), 4);
    int n_correct = 0;
    while(reader.readNext()){
        int result = predictor.predict(reader.pixels());
        std::cout << "Prediction: " << result << " Answer: " << reader.label()
                << (result == reader.label() ? " good" : " bad") << std::endl;
        dump_result.write(reinterpret_cast<char*>(&result), 1);
        if(result == reader.label()) n_correct++;
    }
    std::cout << "Result: " << n_correct << "/" << reader.length() << std::endl;
}

int main()
{
    test_mnist("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "traindata");
    return 0;
}
