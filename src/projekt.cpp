#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


int show(Mat img)
{
    //Imshow
    imshow("okno", img);
    waitKey(0);

    // Ret
    destroyAllWindows();
    return 0;
}

int main(int argc, char** argv)
{
    string path = string(PROJECT_SOURCE_DIR) + "/data/asl_alphabet_test/A_test.jpg";
    Mat input = imread(path); //,IMREAD_GRAYSCALE)
    if (input.empty()) // Sprawdzenie, czy udalo sie otworzyc obraz z sciezka
    {
        cout << "ERROR: Can't open image" << endl;
        return -1;
    }
    show(input);

    //tło
    /*
    string pathbg = string(PROJECT_SOURCE_DIR) + "/data/asl_alphabet_test/nothing_test.jpg";
    Mat bg = imread(pathbg); //,IMREAD_GRAYSCALE)
    if (bg.empty()) // Sprawdzenie, czy udalo sie otworzyc obraz z sciezka
    {
        cout << "ERROR: Can't open image" << endl;
        return -1;
    }
    show(bg);
    */
    Mat output = input.clone();

    Mat img = input.clone();
    //Mat img;
    //absdiff(input, bg, img);
    img.convertTo(img, CV_8UC1);
    //img = Mat(input.rows, input.cols, CV_8UC1);

    //Skala szarosci
    cvtColor(img, img, CV_RGB2GRAY);
    show(img);

    //rozmycie
    //blur(img, img, Size(3,3) );
    //show(img);

    // Dylataja z wykorzystaniem wlasnego elementu strukturalnego
    int kernel_size=3;
    float sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8; // zobacz @getGaussianKernel w dokumentacji opencv
    GaussianBlur(img, img, Size(kernel_size, kernel_size), sigma);
    show(img);

    // Progowanie adaptacyjne
    adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 101,-35);
    show(img);

    // Binaryzacja
    //threshold(img, img, 20, 255, THRESH_BINARY);

    // Morfologiczne otwarcie
    //morphologyEx(img, img, MORPH_OPEN, Mat());

    // Wyznaczenie konturow
    //vector<vector<Point> > contours;
    //findContours(img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    //show(img);

    return 0;
}
