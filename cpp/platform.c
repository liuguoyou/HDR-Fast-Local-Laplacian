#include "platform.h"

static STACK_TYPE stack_instance = { NULL, NULL, 0 };
__int64 hdr_start_time = 0;
__int64 hdr_end_time = 0;
__int64 hdr_freq = 0;

void win_tic(void){
    QueryPerformanceFrequency((LARGE_INTEGER*)&hdr_freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&hdr_start_time);
}

double win_toc(void){
    QueryPerformanceCounter((LARGE_INTEGER*)&hdr_end_time);
    double duration = (hdr_end_time - hdr_start_time) * 1.0 / hdr_freq;
    return duration;
}

int fcompfunc(const void* elem1, const void* elem2){
    return (*(float*)elem1 > *(float*)elem2) ? 1 : (*(float*)elem1 < *(float*)elem2) ? -1 : 0;
}

void float_sorting(float* src, const unsigned int num_element){
    qsort(src, num_element, sizeof(float), fcompfunc);
}

void float_max_min(float* src, const unsigned int num_element, float* srcmax, float* srcmin){
    unsigned int idx = 0;
    float maxresult = 0.0f;
    float minresult = FLT_MAX;
    for (idx = 0; idx < num_element; ++idx){
        maxresult = max(maxresult, src[idx]);
        minresult = min(minresult, src[idx]);
    }
    *srcmax = maxresult;
    *srcmin = minresult;
}

void init_stack(void){
    stack_instance.stack_starting_address = _aligned_malloc(STACK_TOTAL_SIZE, 0x20);
    if (stack_instance.stack_starting_address == NULL){
        printf("failed creating memory stack\n");
        exit(-1);
    }
    stack_instance.stack_current_address = (char*)stack_instance.stack_starting_address;
    stack_instance.stack_current_alloc_size = 0;
}

void free_stack(void){
    _aligned_free(stack_instance.stack_starting_address);
}

void* alloc_from_stack(unsigned int len){
    void* ptr = NULL;
    if (len <= 0){
        len = 0x20;
    }
    unsigned int aligned_len = (len + 0xF) & (~0xF);
    stack_instance.stack_current_alloc_size += aligned_len;
    if (stack_instance.stack_current_alloc_size >= STACK_TOTAL_SIZE){
        printf("failed allocating memory from stack anymore\n");
        _aligned_free(stack_instance.stack_starting_address);
        exit(-1);
    }
    ptr = stack_instance.stack_current_address;
    stack_instance.stack_current_address += aligned_len;
    //C99: all zero bits means 0 for fixed points, 0.0 for floating points
    memset(ptr, 0, len);
    return ptr;
}

void partial_free_from_stack(unsigned int len){
    unsigned int aligned_len = (len + 0xF) & (~0xF);
    stack_instance.stack_current_alloc_size -= aligned_len;
    stack_instance.stack_current_address -= aligned_len;
}

unsigned int get_stack_current_alloc_size(void){
    return stack_instance.stack_current_alloc_size;
}

void reset_stack_ptr_to_assigned_position(unsigned int assigned_size){
    stack_instance.stack_current_address = (char*)stack_instance.stack_starting_address + assigned_size;
    stack_instance.stack_current_alloc_size = assigned_size;
}