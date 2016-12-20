#include "platform.h"
#include "fllf.h"

void FLLF_main(const int num_exposure, float* exposure_values, unsigned char** exposure_images, unsigned char* output_im){
    win_tic();

    init_stack();
    FLLF_INFO_TYPE fllf_info;
    fllf_info.num_exposure = num_exposure;
    fllf_info.exposure_values = exposure_values;
    fllf_info.exposure_images = exposure_images;
    fllf_info.ldr = output_im;
    fllf_info.loglum = (float*)alloc_from_stack(IMG_WIDTH * IMG_HEIGHT * sizeof(float));
    build_hdr_image(&fllf_info);
    tone_mapping_local_laplacian(&fllf_info);
    free_stack();

    printf("FLLF_main duration = %f\n", win_toc());
}