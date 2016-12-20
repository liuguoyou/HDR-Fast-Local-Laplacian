#if !defined PLATFORM_H
#define PLATFORM_H

#include <stdio.h>  //for printf
#include <stdlib.h> //for aligned malloc
#include <string.h> //for memset  
#include <math.h>   //for log, exp,...
#include <float.h>  //for FLT_MAX
#include <windows.h> //for QueryPerformance

#define STACK_TOTAL_SIZE (0x8000000) //128Mbytes
#define IMG_WIDTH (1000)
#define IMG_HEIGHT (1500)

typedef struct{
    void* stack_starting_address;
    char* stack_current_address;
    unsigned int stack_current_alloc_size;
}STACK_TYPE;

void win_tic(void);

double win_toc(void);

void float_sorting(float*, const unsigned int);

void float_max_min(float*,const unsigned int,float*,float*);

void init_stack(void);

void free_stack(void);

void* alloc_from_stack(unsigned int len);

void partial_free_from_stack(unsigned int len);

unsigned int get_stack_current_alloc_size(void);

void reset_stack_ptr_to_assigned_position(unsigned int assigned_size);

#endif /* PLATFORM_H */