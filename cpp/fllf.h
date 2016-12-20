#if !defined FLLF_H
#define FLLF_H

#include "platform.h"

#define LLF_SIGMA_R 0.916290732 //threshold log(2.5)
#define LLF_ALPHA 0.1 //The £\ parameter controls how details are modified: £\=0.25 amplifies detail the most, while £\=1 keeps it unchanged
#define LLF_BETA 0.25 //The £] parameter controls the balance between local and global contrast. Small values favor local contrast
#define LLF_NUM_REF 200 //number of reference lum, local means, parameter g in paper
#define LLF_GAMMA 0.4545

typedef struct{
    int w;
    int h;
}IM_SIZE_TYPE;

typedef struct{
    int num_exposure;
    float* exposure_values;
    unsigned char** exposure_images;
    float* loglum;
    float loglummax;
    float loglummin;
    unsigned char* ldr;
}FLLF_INFO_TYPE;

typedef float(*basic_algebra_ptr)(float, float);

__inline float fadd(float a, float b){
    return a + b;
};

__inline float fminus(float a, float b){
    return a - b;
};

void tone_mapping_local_laplacian(FLLF_INFO_TYPE*);

void build_hdr_image(FLLF_INFO_TYPE*);

#endif /* FLLF_H */