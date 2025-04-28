#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iterator>
#include "grabcut.h"
// #include "graph.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
// #include <omp.h>
#include <curand_kernel.h>

#define COMPONENT_COUNT 5

// GPU Tuning
#define NUM_GPU_STREAMS 4
// Max 640
#define NUM_THREAD_BLOCKS 640
// Max like 4*256 = 1024?
#define THREADS_PER_BLOCK 32
#define MAX_SHARED_MEM (64000/16)
#define MAX_COLS (MAX_SHARED_MEM/6)

using namespace std;

typedef struct
{
    double *model;

    double *coefs;
    double *mean;
    double *cov;

    double inverseCovs[COMPONENT_COUNT][3][3];
    double covDeterms[COMPONENT_COUNT];

    double sums[COMPONENT_COUNT][3];
    double prods[COMPONENT_COUNT][3][3];
    int sampleCounts[COMPONENT_COUNT];
    int totalSampleCount;
} GMM_t;

void initLearning(GMM_t *gmm);
void addSample(GMM_t *gmm, int ci, pixel_t color);
void endLearning(GMM_t *gmm);
int whichComponent(GMM_t *gmm, pixel_t color);

void calcInverseCovAndDeterm(GMM_t *gmm, int ci, double singularFix);

void initEmptyGMM(GMM_t *gmm)
{
    int modelSize = 3 /*mean*/ + 9 /*covariance*/ + 1 /*component weight*/;
    if (gmm == NULL)
        return;

    // gmm = (GMM_t *)malloc(sizeof(GMM_t));
    gmm->model = (double *)calloc(modelSize * COMPONENT_COUNT, sizeof(double));
    if (gmm->model == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return; // DO WE NEED THIS?
    }

    gmm->coefs = gmm->model;
    gmm->mean = gmm->coefs + COMPONENT_COUNT;
    gmm->cov = gmm->mean + 3 * COMPONENT_COUNT;

    // Pretty sure this doesn't do anything for a new array
    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
        calcInverseCovAndDeterm(gmm, ci, 0.0);

    gmm->totalSampleCount = 0;
}

double getComponent(GMM_t *gmm, int ci, pixel_t color)
{
    double result = 0;
    if (gmm->coefs[ci] > 0)
    {
        double *m = gmm->mean + 3 * ci;
        double diff[3] = {color.r - m[0], color.g - m[1], color.b - m[2]};
        double mult = diff[0] * (diff[0] * gmm->inverseCovs[ci][0][0] + diff[1] * gmm->inverseCovs[ci][1][0] + diff[2] * gmm->inverseCovs[ci][2][0]) + diff[1] * (diff[0] * gmm->inverseCovs[ci][0][1] + diff[1] * gmm->inverseCovs[ci][1][1] + diff[2] * gmm->inverseCovs[ci][2][1]) + diff[2] * (diff[0] * gmm->inverseCovs[ci][0][2] + diff[1] * gmm->inverseCovs[ci][1][2] + diff[2] * gmm->inverseCovs[ci][2][2]);
        result = 1.0f / sqrt(gmm->covDeterms[ci]) * exp(-0.5f * mult);
    }
    return result;
}

double doSomething(GMM_t *gmm, pixel_t color)
{
    double res = 0;
    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
        res += gmm->coefs[ci] * getComponent(gmm, ci, color);
    return res;
}

int whichComponent(GMM_t *gmm, pixel_t color)
{
    int k = 0;
    double max = 0;

    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    {
        double p = getComponent(gmm, ci, color);
        if (p > max)
        {
            k = ci;
            max = p;
        }
    }

    return k;
}

void initLearning(GMM_t *gmm)
{
    if (gmm == NULL)
        return;

    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    {
        gmm->sums[ci][0] = gmm->sums[ci][1] = gmm->sums[ci][2] = 0;
        gmm->prods[ci][0][0] = gmm->prods[ci][0][1] = gmm->prods[ci][0][2] = 0;
        gmm->prods[ci][1][0] = gmm->prods[ci][1][1] = gmm->prods[ci][1][2] = 0;
        gmm->prods[ci][2][0] = gmm->prods[ci][2][1] = gmm->prods[ci][2][2] = 0;
        gmm->sampleCounts[ci] = 0;
    }
    gmm->totalSampleCount = 0;
}

void addSample(GMM_t *gmm, int ci, pixel_t color)
{
    if (gmm == NULL)
    {
        cout << "gmm is null in addsample\n";
        return;
    }

    if (ci < 0 || ci >= COMPONENT_COUNT)
    {
        std::cerr << "Invalid component index in addSample: " << ci << std::endl;
        return;
    }

    gmm->sums[ci][0] += color.r;
    gmm->sums[ci][1] += color.g;
    gmm->sums[ci][2] += color.b;
    gmm->prods[ci][0][0] += color.r * color.r;
    gmm->prods[ci][0][1] += color.r * color.g;
    gmm->prods[ci][0][2] += color.r * color.b;
    gmm->prods[ci][1][0] += color.g * color.r;
    gmm->prods[ci][1][1] += color.g * color.g;
    gmm->prods[ci][1][2] += color.g * color.b;
    gmm->prods[ci][2][0] += color.b * color.r;
    gmm->prods[ci][2][1] += color.b * color.g;
    gmm->prods[ci][2][2] += color.b * color.b;
    gmm->sampleCounts[ci]++;
    gmm->totalSampleCount++;
}

void endLearning(GMM_t *gmm)
{
    if (gmm == NULL)
        return;
    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    {
        int n = gmm->sampleCounts[ci];
        if (n == 0)
            gmm->coefs[ci] = 0;
        else
        {
            double inv_n = 1.0 / n;
            gmm->coefs[ci] = (double)n / gmm->totalSampleCount;

            double *m = gmm->mean + 3 * ci;
            m[0] = gmm->sums[ci][0] * inv_n;
            m[1] = gmm->sums[ci][1] * inv_n;
            m[2] = gmm->sums[ci][2] * inv_n;

            double *c = gmm->cov + 9 * ci;
            c[0] = gmm->prods[ci][0][0] * inv_n - m[0] * m[0];
            c[1] = gmm->prods[ci][0][1] * inv_n - m[0] * m[1];
            c[2] = gmm->prods[ci][0][2] * inv_n - m[0] * m[2];
            c[3] = gmm->prods[ci][1][0] * inv_n - m[1] * m[0];
            c[4] = gmm->prods[ci][1][1] * inv_n - m[1] * m[1];
            c[5] = gmm->prods[ci][1][2] * inv_n - m[1] * m[2];
            c[6] = gmm->prods[ci][2][0] * inv_n - m[2] * m[0];
            c[7] = gmm->prods[ci][2][1] * inv_n - m[2] * m[1];
            c[8] = gmm->prods[ci][2][2] * inv_n - m[2] * m[2];

            calcInverseCovAndDeterm(gmm, ci, 0.01);
        }
    }
    // Print GMM means
    // std::cout << "GMM Means:" << std::endl;
    // for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    // {
    //     double *m = gmm->mean + 3 * ci;
    //     std::cout << "Component " << ci << ": (" << m[0] << ", " << m[1] << ", " << m[2] << ")" << std::endl;
    // }

    // Print GMM covariance matrices
    // std::cout << "GMM Covariance Matrices:" << std::endl;
    // for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    // {
    //     double *c = gmm->cov + 9 * ci;
    //     std::cout << "Component " << ci << ":" << std::endl;
    //     std::cout << "[" << c[0] << ", " << c[1] << ", " << c[2] << "]" << std::endl;
    //     std::cout << "[" << c[3] << ", " << c[4] << ", " << c[5] << "]" << std::endl;
    //     std::cout << "[" << c[6] << ", " << c[7] << ", " << c[8] << "]" << std::endl;
    // }
}

void calcInverseCovAndDeterm(GMM_t *gmm, int ci, double singularFix)
{
    if (gmm == NULL)
        return;

    if (gmm->coefs[ci] > 0)
    {
        double *c = gmm->cov + 9 * ci;
        double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
        if (dtrm <= 1e-6 && singularFix > 0)
        {
            // Adds the white noise to avoid singular covariance matrix.
            c[0] += singularFix;
            c[4] += singularFix;
            c[8] += singularFix;
            dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
        }
        gmm->covDeterms[ci] = dtrm;

        double inv_dtrm = 1.0 / dtrm;
        gmm->inverseCovs[ci][0][0] = (c[4] * c[8] - c[5] * c[7]) * inv_dtrm;
        gmm->inverseCovs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) * inv_dtrm;
        gmm->inverseCovs[ci][2][0] = (c[3] * c[7] - c[4] * c[6]) * inv_dtrm;
        gmm->inverseCovs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) * inv_dtrm;
        gmm->inverseCovs[ci][1][1] = (c[0] * c[8] - c[2] * c[6]) * inv_dtrm;
        gmm->inverseCovs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) * inv_dtrm;
        gmm->inverseCovs[ci][0][2] = (c[1] * c[5] - c[2] * c[4]) * inv_dtrm;
        gmm->inverseCovs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) * inv_dtrm;
        gmm->inverseCovs[ci][2][2] = (c[0] * c[4] - c[1] * c[3]) * inv_dtrm;
    }
}

int cpu_dot_diff(image_t *img, uint64_t a_index, uint64_t b_index) {
    int r = (int)(img->r[a_index]) - (int)(img->r[b_index]);
    int g = (int)(img->g[a_index]) - (int)(img->g[b_index]);
    int b = (int)(img->b[a_index]) - (int)(img->b[b_index]);
    return (r * r) + (g * g) + (b * b);
}

// Calculate the first row on the CPU because it wouldn't utilize GPU shared
// memory and otherwise the CPU would be sitting idle
float cpuCalcBetaRowZero(image_t *img, double *leftW) {
    float beta = 0;
    float diff;
    for (int i = 1; i < img->cols; i++) {
        diff = cpu_dot_diff(img, i, i-1);
        beta += diff;
        leftW[i] = diff;
    }

    return beta;
}

__device__ float gpu_dot_diff(uint8_t *shared_mem, uint64_t a_index, uint64_t b_index, uint64_t width) {
    int red = (int)(shared_mem[a_index]) - (int)(shared_mem[b_index]);
    int green = (int)(shared_mem[a_index + 2*width]) - (int)(shared_mem[b_index+2*width]);
    int blue = (int)(shared_mem[a_index+4*width]) - (int)(shared_mem[b_index+4*width]);
    return (float)((red * red) + (green * green) + (blue * blue));
}

__global__ void fastCalcBeta(
        uint8_t *pixels, uint64_t rows, uint64_t cols, int tile_width,
        double *leftW, double *upleftW, double *upW, double *uprightW,
        float *globalBeta
) {
    extern __shared__ uint8_t shared_mem[];

    uint8_t *red = pixels;
    uint8_t *green = pixels + (rows * cols);
    uint8_t *blue = pixels + (2 * rows * cols);

    int id = threadIdx.x;
    // Need overlap on both left & right side
    uint64_t horizontal_tiles = (cols + tile_width - 3) / (tile_width - 2);

    float beta = 0.0;
    // Row 0 will be done on CPU
    for (uint64_t y = blockIdx.x + 1; y < rows; y+= gridDim.x) {

        for (uint64_t j = 0; j < horizontal_tiles; j++) {
            // Copy in 2 rows of the image into shared memory
            uint64_t rel_col = j * tile_width - 2 * j;
            for (uint64_t i = id; i < tile_width; i += blockDim.x) {
                if (rel_col + i < cols) {
                    uint64_t upper_offset = (y-1)*cols + rel_col + i;
                    uint64_t lower_offset = y*cols + rel_col + i;


                    shared_mem[i] = red[upper_offset];
                    shared_mem[i+tile_width] = red[lower_offset];

                    shared_mem[i+2*tile_width] = green[upper_offset];
                    shared_mem[i+3*tile_width] = green[lower_offset];

                    shared_mem[i+4*tile_width] = blue[upper_offset];
                    shared_mem[i+5*tile_width] = blue[lower_offset];
                }
            }

            // Process the two rows
            uint64_t row_index = y * cols + rel_col;
            for (uint64_t x = id; x < tile_width-1; x += blockDim.x)
            {
                uint64_t real_x = (j > 0) ? (rel_col + x + 1): (rel_col + x);
                if (real_x < cols) {
                    uint64_t base_index = tile_width+x + ((j > 0) ? 1 : 0);
                    float diff;
                    if (real_x > 0) // left
                    {
                        diff = gpu_dot_diff(shared_mem, base_index, base_index-1, tile_width);
                        beta += diff;
                        leftW[row_index + x] = diff;
                    } else {
                        leftW[row_index + x] = 0;
                    }
                    if (real_x > 0) // upleft
                    {
                        diff = gpu_dot_diff(shared_mem, base_index, base_index-tile_width-1, tile_width);
                        beta += diff;
                        upleftW[row_index + x] = diff;
                    } else {
                        upleftW[row_index + x] = 0;
                    }

                    // Up - Always Happens
                    diff = gpu_dot_diff(shared_mem, base_index, base_index-tile_width, tile_width);
                    beta += diff;
                    upW[row_index + x] = diff;

                    if (real_x < cols - 1) // upright
                    {
                        diff = gpu_dot_diff(shared_mem, base_index, base_index-tile_width+1, tile_width);
                        beta += diff;
                        uprightW[row_index + x] = diff;
                    } else {
                        uprightW[row_index + x] = 0;
                    }
                }
            }
        }
    }

    // __reduce_add_sync is undefined
    for (int i=blockDim.x / 2; i >= 1; i/=2)
        beta += __shfl_down_sync(0xffffffff, beta, i);

    if (id == 0) atomicAdd(globalBeta, beta);
}

__global__ void fastCalcWeights(double *w, double beta, double gamma, uint64_t size)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int t = blockDim.x * gridDim.x;

    uint64_t pixels_per_thread = (size + t - 1) / t;
    uint64_t start = id * pixels_per_thread;
    uint64_t count = min(pixels_per_thread, size-start);

    uint64_t iters = (count + 7) / 8;
    uint64_t i = start;
    switch (count % 8) {
        case 0: do {    w[i] = gamma * exp(-beta * w[i]); i++;
        case 7:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 6:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 5:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 4:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 3:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 2:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 1:         w[i] = gamma * exp(-beta * w[i]); i++;
                } while (--iters > 0);
    }
}

void fastCalcConsts(image_t *img, double *leftW, double *upleftW, double *upW, double *uprightW, double gamma)
{
    double *gpuLeftW, *gpuUpLeftW, *gpuUpW, *gpuUpRightW;
    uint8_t *gpuPixels;
    float *gpuBeta, beta;
    cudaStream_t streams[NUM_GPU_STREAMS];

    uint64_t num_pixels = img->rows * img->cols;

    cudaError_t err;
    err = cudaMalloc(&gpuPixels, num_pixels * 3);
    if (err != cudaSuccess){
      cout<<"Pixel Memory not allocated"<<endl;
      exit(-1);
    }
    err = cudaMalloc(&gpuLeftW, (size_t) num_pixels * sizeof(double));
    if (err != cudaSuccess){
      cout<<"Left Memory not allocated"<<endl;
      exit(-1);
    }
    err = cudaMalloc(&gpuUpLeftW, (size_t) num_pixels * sizeof(double));
    if (err != cudaSuccess){
      cout<<"UpLeft Memory not allocated"<<endl;
      exit(-1);
    }
    err = cudaMalloc(&gpuUpW, (size_t) num_pixels * sizeof(double));
    if (err != cudaSuccess){
      cout<<"Up Memory not allocated"<<endl;
      exit(-1);
    }
    err = cudaMalloc(&gpuUpRightW, (size_t) num_pixels * sizeof(double));
    if (err != cudaSuccess){
      cout<<"UpRight Memory not allocated"<<endl;
      exit(-1);
    }
    err = cudaMalloc(&gpuBeta, sizeof(float));
    if (err != cudaSuccess){
      cout<<"UpRight Memory not allocated"<<endl;
      exit(-1);
    }

    for (int i = 0; i < NUM_GPU_STREAMS; i++)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

    cudaMemcpy(gpuPixels, img->r, num_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPixels + num_pixels, img->g, num_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPixels + 2*num_pixels, img->b, num_pixels, cudaMemcpyHostToDevice);
    cudaMemset(gpuBeta, 0, sizeof(float));


    uint64_t tile_width = min(img->cols, MAX_COLS);
    uint64_t shared_mem_size = 6 * tile_width;

    auto start = std::chrono::high_resolution_clock::now();
    // CalcBeta (potentially slower but more work)
    fastCalcBeta<<<NUM_THREAD_BLOCKS,THREADS_PER_BLOCK, shared_mem_size, streams[0]>>>(
            gpuPixels, img->rows, img->cols, tile_width,
            gpuLeftW, gpuUpLeftW, gpuUpW, gpuUpRightW, gpuBeta);
    cudaMemcpyAsync(&beta, gpuBeta, sizeof(int), cudaMemcpyDeviceToHost, streams[0]);

    double tmpBeta = cpuCalcBetaRowZero(img, leftW);
    cudaDeviceSynchronize();

    beta += tmpBeta;

    if (beta <= 0.0000001f)
        beta = 0;
    else
        beta = 1.f / (2 * beta / (4 * img->cols * img->rows - 3 * img->cols - 3 * img->rows + 2));

    // cout << "beta: " << beta << endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    cout << "calcBeta: " << duration.count() << " us\n";

    cudaMemcpyAsync(gpuLeftW, leftW, img->cols * sizeof(double), cudaMemcpyHostToDevice, streams[0]);
    cudaMemsetAsync(gpuUpLeftW, 0, img->cols * sizeof(float), streams[1]);
    cudaMemsetAsync(gpuUpW, 0, img->cols * sizeof(float), streams[2]);
    cudaMemsetAsync(gpuUpRightW, 0, img->cols * sizeof(float), streams[3]);

    // CalcNWeights (definitely faster)
    start = std::chrono::high_resolution_clock::now();
    double gammaDivSqrt2 = gamma / sqrt(2.0);

    fastCalcWeights<<<NUM_THREAD_BLOCKS/8,THREADS_PER_BLOCK,0,streams[0]>>>(gpuLeftW, beta, gamma, num_pixels);
    fastCalcWeights<<<NUM_THREAD_BLOCKS/8,THREADS_PER_BLOCK,0,streams[1]>>>(gpuUpLeftW, beta, gammaDivSqrt2, num_pixels);
    fastCalcWeights<<<NUM_THREAD_BLOCKS/8,THREADS_PER_BLOCK,0,streams[2]>>>(gpuUpW, beta, gamma, num_pixels);
    fastCalcWeights<<<NUM_THREAD_BLOCKS/8,THREADS_PER_BLOCK,0,streams[3]>>>(gpuUpRightW, beta, gammaDivSqrt2, num_pixels);

    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    cout << "calcNWeights: " << duration.count() << " us\n";

    cudaMemcpyAsync(leftW, gpuLeftW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(upleftW, gpuUpLeftW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[1]);
    cudaMemcpyAsync(upW, gpuUpW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[2]);
    cudaMemcpyAsync(uprightW, gpuUpRightW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[3]);

    cudaDeviceSynchronize();

    cudaFree(gpuLeftW);
    cudaFree(gpuUpLeftW);
    cudaFree(gpuUpW);
    cudaFree(gpuUpRightW);

    for (int i = 0; i < NUM_GPU_STREAMS; i++)
        cudaStreamDestroy(streams[i]);
}

// Technically should have a checkMask fn

static void initMaskWithRect(mask_t *mask, rect_t rect, image_t *img)
{
    mask->rows = img->rows;
    mask->cols = img->cols;
    mask->array = (MaskVal *)calloc(img->rows * img->cols, sizeof(MaskVal));

    int start_x = rect.x;
    int start_y = rect.y;

    int remaining_width = img->cols - start_x;
    int width = (rect.width < remaining_width) ? rect.width : remaining_width;

    int remaining_height = img->rows - start_y;
    int end_y = rect.height < remaining_height ? rect.height : remaining_height;
    end_y += start_y;
    // int margin = 15;
    for (int r = start_y; r < end_y; r++)
    {
        for (int c = start_x; c < start_x + width; c++)
        {
            // if (r > start_y + margin && r < end_y - margin && c > start_x + margin && c < start_x + width - margin)
            //     mask->array[r * img->cols + c] = GC_FGD;
            // else
            //     mask->array[r * img->cols + c] = GC_PR_FGD;

            mask->array[r * img->cols + c] = GC_PR_FGD;
        }
    }
}

__global__ void kmeans_gpu(
    uint8_t *r, uint8_t *g, uint8_t *b, int num_pixels,
    Centroid *centroids, Centroid *new_centroids, int *counts,
    int *labels, int num_clusters, int max_iters)
{
    // int block_id = blockIdx.x;

    // int num_bytes = 256;                     // use for getting parts of global image
    __shared__ Centroid shared_centroids[5]; // = centroids;
    __shared__ float local_sum_r[5];
    __shared__ float local_sum_g[5];
    __shared__ float local_sum_b[5];
    __shared__ int local_count[5];

    int id = blockIdx.x * blockDim.x + threadIdx.x; // and/or y
    int tid = threadIdx.x;                          // thread id within block

    if (id == 0)
        printf("in gpu %d\n", num_pixels);
    if (id >= num_pixels)
        return;

    for (int iter = 0; iter < max_iters; ++iter)
    {
        if (tid == 0)
        {
            for (int i = 0; i < num_clusters; i++)
            {
                shared_centroids[tid] = centroids[tid];
                local_sum_r[tid] = 0.0f;
                local_sum_g[tid] = 0.0f;
                local_sum_b[tid] = 0.0f;
                local_count[tid] = 0;
            }
        }
        __syncthreads();

        float ri = r[id];
        float gi = g[id];
        float bi = b[id];

        float min_dist = INFINITY;
        int label = 0;

        for (int j = 0; j < num_clusters; ++j)
        {
            float dist = (ri - shared_centroids[j].r) * (ri - shared_centroids[j].r) + (gi - shared_centroids[j].g) * (gi - shared_centroids[j].g) + (bi - shared_centroids[j].b) * (bi - shared_centroids[j].b);
            if (dist < min_dist)
            {
                min_dist = dist;
                label = j;
            }
        }
        labels[id] = label;

        atomicAdd(&local_sum_r[label], ri);
        atomicAdd(&local_sum_g[label], gi);
        atomicAdd(&local_sum_b[label], bi);
        atomicAdd(&local_count[label], 1);
        __syncthreads(); // unncessary?

        if (tid == 0)
        {
            for (int i = 0; i < num_clusters; i++)
            {
                if (local_count[tid] == 0)
                    continue; // avoid division by zero
                centroids[i].r = local_sum_r[tid] / local_count[tid];
                centroids[i].g = local_sum_g[tid] / local_count[tid];
                centroids[i].b = local_sum_b[tid] / local_count[tid];
            }
        }

        __syncthreads();
    }
}

// Initialize GMM background and foreground models using kmeans algorithm.
static void initGMMs(image_t *img, mask_t *mask, GMM_t *bgdGMM, GMM_t *fgdGMM)
{

    // More realistically, we should only definitely put the kmean's num_pixels for loop in the kernel, not entire kmeans algorithm
    int kMeansItCount = 10;
    // int k = 5;
    std::vector<uint8_t> bgdR, bgdG, bgdB;
    std::vector<uint8_t> fgdR, fgdG, fgdB;

    for (int r = 0; r < img->rows; r++)
    {
        for (int c = 0; c < img->cols; c++)
        {
            MaskVal m = mask_at(mask, r, c);
            if (m == GC_BGD || m == GC_PR_BGD)
            {
                bgdR.push_back(get_r(img, r, c));
                bgdG.push_back(get_g(img, r, c));
                bgdB.push_back(get_b(img, r, c));
            }

            // GC_FGD | GC_PR_FGD
            else
            {
                fgdR.push_back(get_r(img, r, c));
                fgdG.push_back(get_g(img, r, c));
                fgdB.push_back(get_b(img, r, c));
            }
        }
    }

    int bdg_size = bgdR.size();
    int fgd_size = fgdR.size();

    int *bgdLabels = (int *)malloc(bdg_size * sizeof(int));
    int *fgdLabels = (int *)malloc(fgd_size * sizeof(int));
    int threadsPerBlock = 64;
    {
        int num_clusters = std::min(COMPONENT_COUNT, bdg_size);

        uint8_t *d_bgdR, *d_bgdG, *d_bgdB;
        cudaMalloc((void **)&d_bgdR, bdg_size * sizeof(uint8_t));
        cudaMalloc((void **)&d_bgdG, bdg_size * sizeof(uint8_t));
        cudaMalloc((void **)&d_bgdB, bdg_size * sizeof(uint8_t));

        cudaMemcpy(d_bgdR, bgdR.data(), bdg_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bgdG, bgdG.data(), bdg_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bgdB, bgdB.data(), bdg_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

        Centroid *centroids = (Centroid *)malloc(num_clusters * sizeof(Centroid)); // in kmeans
        Centroid *new_centroids;
        int *counts;
        int *dev_bgdLabels;
        Centroid *dev_centroids;

        cudaMalloc((void **)&dev_centroids, num_clusters * sizeof(Centroid));
        cudaMalloc((void **)&new_centroids, num_clusters * sizeof(Centroid));
        cudaMalloc((void **)&counts, num_clusters * sizeof(int));
        cudaMalloc((void **)&dev_bgdLabels, bdg_size * sizeof(int));

        srand(1);
        for (int i = 0; i < num_clusters; i++)
        {
            int idx = rand() % (bdg_size);
            centroids[i].r = img->r[idx];
            centroids[i].g = img->g[idx];
            centroids[i].b = img->b[idx];
        }
        cudaMemcpy(dev_centroids, centroids, num_clusters * sizeof(Centroid), cudaMemcpyHostToDevice);

        cout << "before bgd num pixels: " << bdg_size << endl;
        int numBlocks = (bdg_size + threadsPerBlock - 1) / (threadsPerBlock);
        auto start = std::chrono::high_resolution_clock::now();
        kmeans_gpu<<<numBlocks, threadsPerBlock>>>(d_bgdR, d_bgdG, d_bgdB, bdg_size,
                                                   dev_centroids, new_centroids, counts, dev_bgdLabels, num_clusters, kMeansItCount);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        cout << "K-means for background took: " << duration.count() << " us\n";

        cudaMemcpy(bgdLabels, dev_bgdLabels, bdg_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_bgdR);
        cudaFree(d_bgdG);
        cudaFree(d_bgdB);
        cudaFree(dev_centroids);
        cudaFree(new_centroids);
        cudaFree(counts);
        cudaFree(dev_bgdLabels);
        free(centroids);
    }

    {
        int num_clusters = std::min(COMPONENT_COUNT, fgd_size);

        uint8_t *d_fgdR, *d_fgdG, *d_fgdB;
        cudaMalloc((void **)&d_fgdR, fgd_size * sizeof(uint8_t));
        cudaMalloc((void **)&d_fgdG, fgd_size * sizeof(uint8_t));
        cudaMalloc((void **)&d_fgdB, fgd_size * sizeof(uint8_t));

        cudaMemcpy(d_fgdR, fgdR.data(), fgd_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fgdG, fgdG.data(), fgd_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fgdB, fgdB.data(), fgd_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

        Centroid *f_centroids = (Centroid *)malloc(num_clusters * sizeof(Centroid)); // in kmeans
        Centroid *new_centroids;
        int *counts;
        int *dev_fgdLabels;
        Centroid *dev_centroids;

        cudaMalloc((void **)&dev_centroids, num_clusters * sizeof(Centroid));
        cudaMalloc((void **)&new_centroids, num_clusters * sizeof(Centroid));
        cudaMalloc((void **)&counts, num_clusters * sizeof(int));
        cudaMalloc((void **)&dev_fgdLabels, fgd_size * sizeof(int));

        srand(1);
        for (int i = 0; i < num_clusters; i++)
        {
            int idx = rand() % (fgd_size);
            f_centroids[i].g = img->g[idx];
            f_centroids[i].r = img->r[idx];
            f_centroids[i].b = img->b[idx];
        }
        cudaMemcpy(dev_centroids, f_centroids, num_clusters * sizeof(Centroid), cudaMemcpyHostToDevice);


        int numBlocks = (fgd_size + threadsPerBlock - 1) / (threadsPerBlock);
        std::cout << "before fgd num pixels " << fgd_size << endl;

        auto start = std::chrono::high_resolution_clock::now();
        kmeans_gpu<<<numBlocks, threadsPerBlock>>>(d_fgdR, d_fgdG, d_fgdB, fgd_size,
                                                   dev_centroids, new_centroids, counts, dev_fgdLabels, num_clusters, kMeansItCount);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        cout << "K-means for foreground took: " << duration.count() << " us\n";

        cudaMemcpy(fgdLabels, dev_fgdLabels, fgd_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_fgdR);
        cudaFree(d_fgdG);
        cudaFree(d_fgdB);
        cudaFree(dev_centroids);
        cudaFree(new_centroids);
        cudaFree(counts);
        cudaFree(dev_fgdLabels);
        free(f_centroids);
    }

    cout << "done with kmeans\n";
    initLearning(bgdGMM);
    for (int i = 0; i < bdg_size; i++)
    {
        pixel_t px = {bgdR[i], bgdG[i], bgdB[i]};
        addSample(bgdGMM, bgdLabels[i], px);
    }
    // std::cout << "BGD GMM means weights after initGMMs" << std::endl;
    endLearning(bgdGMM);

    initLearning(fgdGMM);
    for (int i = 0; i < fgd_size; i++)
    {
        pixel_t px = {fgdR[i], fgdG[i], fgdB[i]};
        addSample(fgdGMM, fgdLabels[i], px);
    }
    // std::cout << "FGD GMM means weights after initGMMs" << std::endl;
    endLearning(fgdGMM);
}

/*
static void assignGMMsComponents(image_t *img, mask_t *mask, GMM_t *bgdGMM, GMM_t *fgdGMM, int *compIdxs)
{
    for (int r = 0; r < img->rows; r++)
    {
        int row_index = r * img->cols;
        for (int c = 0; c < img->cols; c++)
        {
            pixel_t color = {get_r(img, r, c), get_g(img, r, c), get_b(img, r, c)};
            MaskVal m = mask_at(mask, r, c);
            compIdxs[row_index + c] = (m == GC_BGD || m == GC_PR_BGD) ? whichComponent(bgdGMM, color) : whichComponent(fgdGMM, color);
        }
    }
}
*/

/*
// Learn GMMs parameters.
static void learnGMMs(image_t *img, mask_t *mask, int *compIdxs, GMM_t *bgdGMM, GMM_t *fgdGMM, int iter)
{
    initLearning(bgdGMM);
    initLearning(fgdGMM);
    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    {
        for (int r = 0; r < img->rows; r++)
        {
            int row_index = r * img->cols;
            for (int c = 0; c < img->cols; c++)
            {
                if (compIdxs[row_index + c] == ci)
                {
                    MaskVal m = mask_at(mask, r, c);
                    if (iter == 0)
                    {
                        if (m == GC_BGD || m == GC_PR_BGD)
                        {
                            pixel_t color = {get_r(img, r, c), get_g(img, r, c), get_b(img, r, c)};
                            addSample(bgdGMM, ci, color);
                        }
                        else if (m == GC_FGD || m == GC_PR_FGD)
                        {
                            pixel_t color = {get_r(img, r, c), get_g(img, r, c), get_b(img, r, c)};
                            addSample(fgdGMM, ci, color);
                        }
                    }
                    else
                    {
                        if (m == GC_BGD || m == GC_PR_BGD)
                        {
                            pixel_t color = {get_r(img, r, c), get_g(img, r, c), get_b(img, r, c)};
                            addSample(bgdGMM, ci, color);
                        }
                        else
                        {
                            pixel_t color = {get_r(img, r, c), get_g(img, r, c), get_b(img, r, c)};
                            addSample(fgdGMM, ci, color);
                        }
                    }
                }
            }
        }
    }
    // std::cout << "BGD GMM means weights after learning:" << std::endl;
    endLearning(bgdGMM);
    // std::cout << "FGD GMM means weights after learning:" << std::endl;
    endLearning(fgdGMM);
}
*/

/*
static void constructGCGraph(image_t *img, mask_t *mask, GMM_t *bgdGMM, GMM_t *fgdGMM, double lambda,
                             double *leftW, double *upleftW, double *upW, double *uprightW,
                             GCGraph<double> &graph)
{
    if (img == NULL || mask == NULL || bgdGMM == NULL || fgdGMM == NULL)
        return;

    int vtxCount = img->cols * img->rows,
        edgeCount = 2 * (4 * img->cols * img->rows - 3 * (img->cols + img->rows) + 2);

    // cout << "vertex count: " << vtxCount << "\n";
    graph.create(vtxCount, edgeCount);
    // std::cout << "Graph created with " << vtxCount << " vertices and " << edgeCount << " edges." << std::endl;

    // cout << "created graph in construct function\n";
    for (int r = 0; r < img->rows; r++)
    {
        int row_index = r * img->cols;
        for (int c = 0; c < img->cols; c++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            pixel_t color = {get_r(img, r, c), get_g(img, r, c), get_b(img, r, c)};

            // set t-weights
            double fromSource, toSink;
            MaskVal m = mask_at(mask, r, c);
            if (m == GC_PR_BGD || m == GC_PR_FGD)
            {
                fromSource = -log(doSomething(bgdGMM, color) + 1e-6);
                toSink = -log(doSomething(fgdGMM, color) + 1e-6);
            }
            else if (m == GC_BGD)
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.addTermWeights(vtxIdx, fromSource, toSink);

            // set n-weights
            if (c > 0)
            {
                double w = leftW[row_index + c];
                graph.addEdges(vtxIdx, vtxIdx - 1, w, w);
            }
            if (c > 0 && r > 0)
            {
                double w = upleftW[row_index + c];
                graph.addEdges(vtxIdx, vtxIdx - img->cols - 1, w, w);
            }
            if (r > 0)
            {
                double w = upW[row_index + c];
                graph.addEdges(vtxIdx, vtxIdx - img->cols, w, w);
            }
            if (c < img->cols - 1 && r > 0)
            {
                double w = uprightW[row_index + c];
                graph.addEdges(vtxIdx, vtxIdx - img->cols + 1, w, w);
            }
        }
    }
}
*/

/*
static void estimateSegmentation(GCGraph<double> &graph, mask_t *mask)
{
    int flow = graph.maxFlow();
    // cout << "Max flow: " << flow << "\n";
    for (int r = 0; r < mask->rows; r++)
    {
        for (int c = 0; c < mask->cols; c++)
        {
            MaskVal m = mask_at(mask, r, c);
            if (m == GC_PR_BGD || m == GC_PR_FGD)
            {
                if (graph.inSourceSegment(r * mask->cols + c ))
                {
                    // cout << "mask[" << r << "][" << c << "] = " << m;
                    mask_set(mask, r, c, GC_PR_FGD);
                    // cout << " mask[" << r << "][" << c << "] = GC_PR_FGD\n";
                }
                else
                {
                    // cout << "mask[" << r << "][" << c << "] = " << m;
                    mask_set(mask, r, c, GC_PR_BGD);
                    // cout << " mask[" << r << "][" << c << "] = GC_PR_BGD\n";
                }
            }
        }
    }
}
*/

void displayImage(image_t *img)
{
    cv::Mat displayImg(img->rows, img->cols, CV_8UC3);
    for (int r = 0; r < img->rows; r++)
    {
        for (int c = 0; c < img->cols; c++)
        {
            displayImg.at<cv::Vec3b>(r, c) = cv::Vec3b(get_b(img, r, c), get_g(img, r, c), get_r(img, r, c));
        }
    }
    cv::imshow("Image", displayImg);
    cv::waitKey(0);
}

void gettingOutput(image_t *img, mask_t *mask, image_t *foreground, image_t *background)
{
    for (int r = 0; r < img->rows; r++)
    {
        for (int c = 0; c < img->cols; c++)
        {
            uint8_t R = get_r(img, r, c);
            uint8_t G = get_g(img, r, c);
            uint8_t B = get_b(img, r, c);
            if (mask_at(mask, r, c) == GC_FGD || mask_at(mask, r, c) == GC_PR_FGD)
                set_rgb(foreground, r, c, R, G, B), set_rgb(background, r, c, 0, 0, 0);
            else
                set_rgb(background, r, c, R, G, B), set_rgb(foreground, r, c, 0, 0, 0);
        }
    }
    // std::cout << "Segmentation result: " << fg << " foreground, " << bg << " background pixels." << std::endl;
}

void grabCut(image_t *img, rect_t rect, image_t *foreground, image_t *background, int iterCount)
{
    int num_pixels = img->rows * img->cols;

    GMM_t *bgdGMM, *fgdGMM;
    bgdGMM = (GMM_t *)malloc(sizeof(GMM_t));
    fgdGMM = (GMM_t *)malloc(sizeof(GMM_t));
    mask_t *mask = (mask_t *)malloc(sizeof(mask_t));

    initEmptyGMM(bgdGMM);
    initEmptyGMM(fgdGMM);

    int *compIdxs = (int *)malloc(num_pixels * sizeof(int));

    initMaskWithRect(mask, rect, img);
    initGMMs(img, mask, bgdGMM, fgdGMM);

    if (iterCount <= 0)
        return;

    const double gamma = 50;
    // const double lambda = 9 * gamma;

    double *leftW, *upleftW, *upW, *uprightW;
    leftW = (double *)malloc(num_pixels * sizeof(double));
    upleftW = (double *)malloc(num_pixels *  sizeof(double));
    upW = (double *)malloc(num_pixels * sizeof(double));
    uprightW = (double *)malloc(num_pixels * sizeof(double));
    fastCalcConsts(img, leftW, upleftW, upW, uprightW, gamma);

    // for (int i = 0; i < iterCount; i++) //i< iterCount
    // {
    //     GCGraph<double> graph;
    //     assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);
    //     learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM, i);
    //     constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
    //     estimateSegmentation(graph, mask);
    // }
    // gettingOutput(img, mask, foreground, background);

    // displayImage(foreground);
    // displayImage(background);
    // cout << "after lop\n";
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./FastGrabCut <image_path> [x1 y1 x2 y2]" << std::endl;
        return -1;
    }

    string file_path = argv[1];
    cv::Mat image = cv::imread(file_path);

    if (image.empty())
    {
        std::cerr << "Image not loaded!" << std::endl;
        return -1;
    }
    std::cout << "Loaded Image " << file_path << std::endl;

    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->rows = image.rows;
    img->cols = image.cols;
    img->r = (uint8_t *)malloc(img->rows * img->cols * sizeof(uint8_t));
    img->g = (uint8_t *)malloc(img->rows * img->cols * sizeof(uint8_t));
    img->b = (uint8_t *)malloc(img->rows * img->cols * sizeof(uint8_t));

    cout << img->rows << " " << img->cols << endl;

    image_t *foreground = (image_t *)malloc(sizeof(image_t));
    image_t *background = (image_t *)malloc(sizeof(image_t));

    foreground->rows = background->rows = image.rows;
    foreground->cols = background->cols = image.cols;

    foreground->r = (uint8_t *)calloc(img->rows * img->cols, sizeof(uint8_t));
    foreground->g = (uint8_t *)calloc(img->rows * img->cols, sizeof(uint8_t));
    foreground->b = (uint8_t *)calloc(img->rows * img->cols, sizeof(uint8_t));
    background->r = (uint8_t *)calloc(img->rows * img->cols, sizeof(uint8_t));
    background->g = (uint8_t *)calloc(img->rows * img->cols, sizeof(uint8_t));
    background->b = (uint8_t *)calloc(img->rows * img->cols, sizeof(uint8_t));

    for (int r = 0; r < img->rows; r++)
    {
        for (int c = 0; c < img->cols; c++)
        {
            cv::Vec3b color = image.at<cv::Vec3b>(r, c);
            set_rgb(img, r, c, color[2], color[1], color[0]);
        }
    }
    uint64_t x1 = 0, y1 = 0, x2 = img->cols - 1, y2 = img->rows - 1;

    if (argc == 6)
    {
        x1 = std::stoi(argv[2]);
        y1 = std::stoi(argv[3]);
        x2 = std::stoi(argv[4]);
        y2 = std::stoi(argv[5]);
    }
    else
    {
        std::cerr << "Warning: No bounding box provided, using full image" << std::endl;
    }

    grabCut(img, {x1, y1, x2, y2}, foreground, background, 5);

    // grabCut(img, {132, 75, 845, 525}, foreground, background, 5);
    // 132 75 845 525
    free(img->r);
    free(img->g);
    free(img->b);
    free(img);
    free(foreground->r);
    free(foreground->g);
    free(foreground->b);
    free(foreground);
    free(background->r);
    free(background->g);
    free(background->b);
    free(background);
    return 0;
}
