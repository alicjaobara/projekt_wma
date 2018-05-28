#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h> //DIR

using namespace std;
using namespace cv;

// Directory containing sample images
static string SamplesDir= "/home/alicja/gnu/projekt_wma/data/asl_alphabet_train/";

//static string SamplesDirB = "/home/alicja/gnu/projekt_wma/data/asl_alphabet_train/B/";
// Set the file to write the features to
static string featuresFile = "/home/alicja/gnu/projekt_wma/data/hogFeatures/hog.txt";

//parametry do HOG
static const Size trainingPadding = Size(0, 0);
static const Size winStride = Size(8, 8);

static vector<string> validExtensions;


static void storeCursor(void) {
    printf("\033[s");
}

static void resetCursor(void) {
    printf("\033[u");
}

static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

// pokazywanie obrazów
int show(Mat img,string okno)
{
    imshow(okno, img);
    waitKey(0);
    cout<<"wielkość "<<okno<<" "<<img.size()<<endl;
    destroyAllWindows();
    return 0;
}
static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog) {
    Mat imageData = imread(imageFilename, IMREAD_GRAYSCALE);
    resize(imageData, imageData, Size(64, 128));
    if (imageData.empty()) {
        featureVector.clear();
        printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
        return;
    }
    // Check for mismatching dimensions
    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
        featureVector.clear();
        printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
        return;
    }
    vector<Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
    imageData.release(); // Release the image again after features are extracted
}
static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
    printf("Opening directory %s\n", dirName.c_str());
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
            if (ep->d_type & DT_DIR) {
                continue;
            }
            extensionLocation = string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                printf("Found matching data file '%s'\n", ep->d_name);
                fileNames.push_back((string) dirName + ep->d_name);
            } else {
                printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
            }
        }
        (void) closedir(dp);
    } else {
        printf("Error opening directory '%s'!\n", dirName.c_str());
    }
    return;
}

int doPliku(string sign){
    HOGDescriptor hog;
    static vector<string> TrainingImages;
    string SamplesDirCurrent = SamplesDir + sign + "/";
    getFilesInDirectory(SamplesDirCurrent, TrainingImages, validExtensions);
    unsigned long overallSamples = TrainingImages.size();
    if (overallSamples == 0) {
        printf("No training sample files found, nothing to do!\n");
        return EXIT_SUCCESS;
    }

    printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFile.c_str());
    float percent;
    fstream File;
    File.open(featuresFile.c_str(), ios::out|ios::app);
    if (File.good() && File.is_open()) {
        // Iterate over sample images
        for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
            storeCursor();
            vector<float> featureVector;
            const string currentImageFile = TrainingImages.at(currentFile);
            // Output progress
            if ( (currentFile+1) % 10 == 0 || (currentFile+1) == overallSamples ) {
                percent = ((currentFile+1) * 100 / overallSamples);
                printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile+1), percent, currentImageFile.c_str());
                fflush(stdout);
                resetCursor();
            }
            // Calculate feature vector from current image file
            calculateFeaturesFromInput(currentImageFile, featureVector, hog);
            if (!featureVector.empty()) {
                // Save feature vector components
                for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
//                    if(feature == 1)
//                        File << featureVector.at(feature);
//                    else
                        File << "," << featureVector.at(feature);
                }
                File << ", " <<sign << endl;
            }
        }
        printf("\n");
        File.flush();
        File.close();
    }
    else {
        printf("Error opening file '%s'!\n", featuresFile.c_str());
        return EXIT_FAILURE;
    }
}

int main(int argc, char** argv)
{
    validExtensions.push_back("jpg");
    fstream File;
    File.open(featuresFile.c_str(), ios::out);
    if (File.good() && File.is_open()) {
        File << "first line" << endl;
        File.flush();
        File.close();
    }
    doPliku("A");


    return 0;
}
