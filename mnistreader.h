#ifndef MNISTREADER_H
#define MNISTREADER_H

#include <vector>
#include <array>
#include <fstream>

using MNISTpixels = std::array<unsigned char, 784>;

enum class MNISTreaderError {
    noError,
    imagesFileIOFailure,
    labelsFileIOFailure,
    differentNumberOfImage,  // If images file's number of images and labels file's number of items are different, this error will be set.
    unexpectedPixelsSize,
    imagesFileMalformed,  // Only check magic number.
    labelsFileMalformed,  // Only check magic number.
    imagesFileUnexpectedEnd,
    labelsFileUnexpectedEnd
};

class MNISTreader
{
public:
    MNISTreader(const char *images_filepath, const char *labels_filepath);
    bool readNext();
    std::vector<std::pair<MNISTpixels, int>> minibatch(int num);
    int label() const noexcept { return current_label; }
    const MNISTpixels& pixels() const noexcept { return current_pixels; }
    int index() const noexcept { return current_pos; }
    int length() const noexcept { return n_images; }
    MNISTreaderError getError() const noexcept { return error; }
    bool isError() const noexcept { return error != MNISTreaderError::noError; }
    static const int pixelHeight = 28;
    static const int pixelWidth = 28;
    static const int pixelSize = 784;
private:
    int n_images;
    int current_pos;
    int current_label;
    MNISTreaderError error;
    std::ifstream images;
    std::ifstream labels;
    MNISTpixels current_pixels;
    bool readNext_impl(MNISTpixels& dest);
    void setError(MNISTreaderError e, bool file_close=true);
};

#endif // MNISTREADER_H
