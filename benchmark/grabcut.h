
#include <stdint.h>

typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} pixel_t;

typedef struct {
    uint64_t x;
    uint64_t y;
    uint64_t width;
    uint64_t height;
} rect_t;

typedef struct {
    uint64_t rows;
    uint64_t cols;
    pixel_t *array;
} image_t;

pixel_t *img_at(const image_t *img, uint64_t row, uint64_t col) {
    uint64_t index = row * img->cols + col;
    return &(img->array[index]);
}

double dot_diff(pixel_t *p0, pixel_t *p1) {
    int red_diff = p0->r - p1->r;
    int green_diff = p0->g - p1->g;
    int blue_diff = p0->b - p1->b;
    return (double)(red_diff * red_diff + green_diff * green_diff + blue_diff + blue_diff);
}

// Numbers in theory could be utilized for checkMask
typedef enum __attribute__((__packed__)) {
    GC_BGD = 0, 
    GC_FGD = 1, 
    GC_PR_BGD = 2, 
    GC_PR_FGD = 3
} MaskVal;

typedef MaskVal* mask_t;
typedef double* weight_t;




void grabCut(pixel_t *img);

