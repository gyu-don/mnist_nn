#include "mnistreader.h"
#include <cstdint>

MNISTreader::MNISTreader(const char *images_filepath, const char *labels_filepath)
    : n_images(0), current_pos(-1), current_label(-1), error(MNISTreaderError::noError),
      images(images_filepath, std::ios_base::in | std::ios_base::binary),
      labels(labels_filepath, std::ios_base::in | std::ios_base::binary)
{
    if(images.bad()){
        setError(MNISTreaderError::imagesFileIOFailure);
        return;
    }
    if(labels.bad()){
        setError(MNISTreaderError::labelsFileIOFailure);
        return;
    }
    std::int32_t t, u;
    images.read(reinterpret_cast<char*>(&t), 4);
    if(t != 0x03080000){
        setError(MNISTreaderError::imagesFileMalformed);
    }
    labels.read(reinterpret_cast<char*>(&t), 4);
    if(t != 0x01080000){
        setError(MNISTreaderError::labelsFileMalformed);
    }
    images.read(reinterpret_cast<char*>(&t), 4);
    labels.read(reinterpret_cast<char*>(&u), 4);
    if(t != u){
        setError(MNISTreaderError::differentNumberOfImage);
        return;
    }
    n_images = (t >> 24) & 0xFF | (t >> 8) & 0xFF00 | (t << 8) & 0xFF0000 | (t << 24) & 0xFF000000;
    images.read(reinterpret_cast<char*>(&t), 4);
    images.read(reinterpret_cast<char*>(&u), 4);
    if(t != (28 << 24) || u != (28 << 24)){
        setError(MNISTreaderError::unexpectedPixelsSize);
        return;
    }
}

bool MNISTreader::readNext()
{
    readNext_impl(current_pixels);
}

bool MNISTreader::readNext_impl(MNISTpixels &dest)
{
    if(isError()) return false;
    if(current_pos >= n_images - 1) return false;
    images.read(reinterpret_cast<char*>(dest.data()), 784);
    if(images.bad()){
        setError(MNISTreaderError::imagesFileIOFailure);
        return false;
    }
    if(images.eof()){
        setError(MNISTreaderError::imagesFileUnexpectedEnd);
        return false;
    }

    char c;
    labels.read(&c, 1);
    if(labels.bad()){
        setError(MNISTreaderError::labelsFileIOFailure);
        return false;
    }
    if(labels.eof()){
        setError(MNISTreaderError::labelsFileUnexpectedEnd);
        return false;
    }
    current_label = c;
    current_pos++;
    return true;
}

std::vector<std::pair<MNISTpixels, int>> MNISTreader::minibatch(int num)
{
    using Pair = std::pair<MNISTpixels, int>;
    std::vector<Pair> vec;
    int i;

    for(i=0;i<num;i++){
        MNISTpixels arr;
        if(readNext_impl(arr)) vec.push_back(Pair(std::move(arr), current_label));
        else break;
    }
    if(i > 0) current_pixels = vec.back().first;
    return vec;
}

void MNISTreader::setError(MNISTreaderError e, bool file_close)
{
    error = e;
    if(file_close){
        images.close(); labels.close();
    }
}
