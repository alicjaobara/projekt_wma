#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;

// Directory containing positive sample images
static string posSamplesDir = "/home/alicja/gnu/projekt_wma/data/asl_alphabet_train/A";
// Set the file to write the features to
static string featuresFile = "/home/alicja/gnu/projekt_wma/data/hog_A.txt";

//parametry do HOG
static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(0, 0);

int show(Mat img, string okno){
    imshow(okno, img);
    waitKey(0);
    cout<<"wielkość "<<okno<<" "<<img.size()<<endl;
    destroyAllWindows();
    return 0;
}

void zapis(string featuresFile, vector< float> featureVector){
    fstream File;
    File.open(featuresFile.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << featuresFile.c_str() << endl; // Remove this line for libsvm which does not support comments
            // Calculate feature vector from current image file
            if (!featureVector.empty()) {
                for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
                    File << " " << (feature + 1) << ":" << featureVector.at(feature);
                }
                File << endl;
            }

        printf("\n");
        File.flush();
        File.close();
    } else {
        printf("Error opening file '%s'!\n", featuresFile.c_str());

    }

}

int main()
{

    //image load
    Mat img1 = imread(string(PROJECT_SOURCE_DIR) + "/data/asl_alphabet_test/B_test.jpg");
    show(img1,"img1");
    Mat img2 = imread(string(PROJECT_SOURCE_DIR) + "/data/asl_alphabet_train/B/B2.jpg");
    show(img2,"img2");

    //rgb 2 gray
    Mat img1_gray;
    cvtColor(img1, img1_gray, CV_RGB2GRAY);

    Mat img2_gray;
    cvtColor(img2, img2_gray, CV_RGB2GRAY);

    //resize smaller
    Mat r_img1_gray;
    resize(img1_gray, r_img1_gray, Size(64, 128));
    Mat r_img2_gray;
    resize(img2_gray, r_img2_gray, Size(64, 128));

    show(r_img1_gray,"img1 gray r");
    show(r_img2_gray,"img2 gray r");

    //extractino hog feature
    HOGDescriptor d1( Size(64,8), Size(8,8), Size(4,4), Size(4,4), 9);
    HOGDescriptor d2( Size(64,8), Size(8,8), Size(4,4), Size(4,4), 9);
    // Size(32,16), //winSize
    // Size(8,8), //blocksize
    // Size(4,4), //blockStride,
    // Size(4,4), //cellSize,
    // 9, //nbins,

    //hog feature compute
    vector< float> descriptorsValues1;
    vector< Point> locations1;
    d1.compute( r_img1_gray, descriptorsValues1, trainingPadding, winStride, locations1);
    vector< float> descriptorsValues2;
    vector< Point> locations2;
    d2.compute( r_img2_gray, descriptorsValues2, trainingPadding, winStride, locations2);

    //hog feature size
    //cout << descriptorsValues1.size() << endl;

    /*
    //copy vector to mat
    //create Mat
    Mat A(descriptorsValues1.size(),1,CV_32FC1);
    //copy vector to mat
    memcpy(A.data,descriptorsValues1.data(),descriptorsValues1.size()*sizeof(float));
    //create Mat
    Mat B(descriptorsValues2.size(),1,CV_32FC1);
    //copy vector to mat
    memcpy(B.data,descriptorsValues2.data(),descriptorsValues2.size()*sizeof(float));
    */

    zapis(featuresFile, descriptorsValues1);


    return 0;
}
