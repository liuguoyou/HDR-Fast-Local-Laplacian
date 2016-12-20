#if !defined ENTRY_H
#define ENTRY_H

#ifdef __cplusplus
extern "C"{
#endif

#include "platform.h"
#include "fllf.h"

void FLLF_main(const int num_exposure, float* exposure_values, unsigned char** exposure_images, unsigned char* output_image);

#ifdef __cplusplus
}
#endif

#endif /* ENTRY_H */