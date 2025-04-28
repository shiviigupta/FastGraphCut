
#include <cstdint>


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

//new image_t with contiguous memory
typedef struct {
    int rows;
    int cols;
    uint8_t *r;
    uint8_t *g;
    uint8_t *b;
} image_t;

uint8_t get_r(image_t *img, int row, int col) {
    return img->r[row * img->cols + col];
}

uint8_t get_g(image_t *img, int row, int col) {
    return img->g[row * img->cols + col];
}

uint8_t get_b(image_t *img, int row, int col) {
    return img->b[row * img->cols + col];
}

void set_rgb(image_t *img, int row, int col, uint8_t r, uint8_t g, uint8_t b) {
    img->r[row * img->cols + col] = r;
    img->g[row * img->cols + col] = g;
    img->b[row * img->cols + col] = b;
}

double dot_diff_rgb(uint8_t r1, uint8_t g1, uint8_t b1, uint8_t r2, uint8_t g2, uint8_t b2) {
    int red_diff = r1 - r2;
    int green_diff = g1 - g2;
    int blue_diff = b1 - b2;
    return (double)(red_diff * red_diff + green_diff * green_diff + blue_diff * blue_diff);
}

// Numbers in theory could be utilized for checkMask
typedef enum __attribute__((__packed__)) {
    GC_BGD = 0,
    GC_FGD = 1,
    GC_PR_BGD = 2,
    GC_PR_FGD = 3
} MaskVal;

typedef struct {
    int rows;
    int cols;
    MaskVal* array;
} mask_t;

MaskVal mask_at( mask_t *mask, int row, int col) {
    return mask->array[row * mask->cols + col];
}

void mask_set(mask_t *mask, int row, int col, MaskVal val) {
    mask->array[row * mask->cols + col] = val;
}

typedef double* weight_t;


// For kmeans
typedef struct {
    float r, g, b;
} Centroid;


void grabCut(pixel_t *img);

