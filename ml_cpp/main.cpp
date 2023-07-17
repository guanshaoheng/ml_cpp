#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <torch/torch.h>

using namespace cv;

int main(int argc, char** argv)
{
    // torch in C++
    torch::Tensor tensor = torch::rand({ 2, 3 }).to(torch::kCUDA);

    std::cout << "Please rerere" << std::endl;

    torch::Tensor tensor2 = torch::einsum("ij, ij->ij", { tensor, tensor });

    std::cout << tensor << std::endl;

    std::cout << tensor2 << std::endl;

    // OpenCV
    /*Mat image;
    image = imread("D:\\ubuntu_exchange\\c_test_new\\selfie_foto.png");
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);
    waitKey(0);*/

    return 0;
}
