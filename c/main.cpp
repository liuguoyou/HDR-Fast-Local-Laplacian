//////////////////////////////////////////////////////////////////////////
//
// fast local laplacian filter
//
// by Lincoln Hard ( lincolnhardabc@gmail.com )
//
//////////////////////////////////////////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#include "entry.h"

int get_num_exposures(const char* list_file_path){
    FILE* fp = fopen(list_file_path, "r");
    char buf[128] = { 0 };
    int num_line = 0;
    while (fgets(buf, 128, fp) != NULL){
        ++num_line;
    }
    fclose(fp);
    return num_line;
}

//opencv read image
void parse_list(const char* list_file_path, const int num_exposures, float* exposure_values, unsigned char** exposure_images){
    FILE* fp = fopen(list_file_path, "r");
    char buf[128] = { 0 };

    int exposures_idx = 0;
    int frame_size = 3 * IMG_WIDTH * IMG_HEIGHT * sizeof(unsigned char);
    for (exposures_idx = 0; exposures_idx < num_exposures; ++exposures_idx){
        fscanf(fp, "%s %f\n", buf, &(exposure_values[exposures_idx]));
        Mat im = imread(buf);
        memcpy(exposure_images[exposures_idx], im.data, frame_size);
    }
    fclose(fp);
}

//opencv display image
void display_result(const unsigned char* output){
    Mat im = Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, (void*)output);
    imshow("result", im);
    waitKey(0);
}

int main(int ac, char** av){
    if (ac != 2){
        printf("Usage: llf_enhanced.exe [exposure list file]\n");
        exit(-1);
    }
    int num_exposure = get_num_exposures(av[1]);
    float* exposure_vals = (float*)malloc(num_exposure * sizeof(float));
    unsigned char** exposure_ims = (unsigned char**)malloc(num_exposure * sizeof(unsigned char*));
    unsigned char* output_im = (unsigned char*)malloc(3 * IMG_WIDTH * IMG_HEIGHT * sizeof(unsigned char));
    int exposures_idx = 0;
    for (exposures_idx = 0; exposures_idx < num_exposure; ++exposures_idx){
        exposure_ims[exposures_idx] = (unsigned char*)malloc(3 * IMG_WIDTH * IMG_HEIGHT * sizeof(unsigned char));
    }
    parse_list(av[1], num_exposure, exposure_vals, exposure_ims);

    FLLF_main(num_exposure, exposure_vals, exposure_ims, output_im);
    display_result(output_im);

    free(exposure_vals);
    for (exposures_idx = 0; exposures_idx < num_exposure; ++exposures_idx){
        free(exposure_ims[exposures_idx]);
    }
    free(exposure_ims);
    return 0;
}

