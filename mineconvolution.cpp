#include<stdio.h>>
#include <cuda.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

const int alignment = 32; // 32 byte alignment
const int size = 100;
const int kernel = 3;  // odd
const int batch_size = 128;
const int in_channel = 128;
const int out_channel = 128;

#define InitRandom()                         \
  std::random_device r;                      \
  std::default_random_engine generator(r()); \
  std::uniform_int_distribution<> distribution(0, 255);

#define a(_n, _x, _y, _c) a[(_n) * size * size * in_channel + (_x) * size * in_channel + (_y) * in_channel + (_c)]
#define w(_x, _y, _ci, _co) w[(_x) * kernel * in_channel * out_channel + (_y) * in_channel * out_channel + (_ci) * out_channel + (_co)]
#define b(_n, _x, _y, _c) b[(_n) * size * size * out_channel + (_x) * size * out_channel + (_y) * out_channel + (_c)]
#define CUDA_CALL(func)                                         \
  {                                                             \
    cudaError_t e = (func);                                     \
    if (!(e == cudaSuccess || e == cudaErrorCudartUnloading))   \
    {                                                           \
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));   \
	      abort();                                                \
    }                                                           \
  }

const int block_size = 16;
__global__ void conv2d_cuda_kernel(const uint8_t *__restrict__ a, 
                                   const uint8_t *__restrict__ w, 
                                   uint8_t *__restrict__ b) 
{
  //a输入wkernelb输出
  __shared__ float TiledA[batch_size][size][size][in_channel];
  __shared__ float TiledB[kernel][kernel][in_channel][out_channel];
  int tx = threadIdx.x, ty = threadIdx.y;
  const int i = blockIdx.x * block_size + tx;
  const int j = blockIdx.y * block_size + ty;
  int x = i - kernel / 2, y = j - kernel / 2;
  if (i < size && j < size) {
    // 遍历顺序 
    for (int s = 0; s < batch_size; ++s ) {
      for ( int k = 0; k < kernel; ++k) {
        uint8_t conv = 0;
        // Conv2d for a single pixel, single output channel.
        for (int  int l = 0; l < kernel; ++l) {
          
          

          for ( int CI = 0; CI < in_channel; ++CI) {
            for ( CO = 0; CO < out_channel; ++CO) {

              TiledA[s][x][y][CI] = a(s, x, y, CI);
              TiledB[k][l][CO][CI] = w(k, l, CI, CO);
              syncthreads();

              if (!(x < 0 || x >= size || y < 0 || y >= size)) {
                // conv += a(s, x, y, CI) * w(k, l, CI, CO);
                conv += TiledA[(s) * size * size * in_channel + (x) * size * in_channel + (y) * in_channel + (CI)]
                        *TiledB[(k) * kernel * in_channel * out_channel + (l) * in_channel * out_channel + (CI) * out_channel + (CO)]
              }
              y++;
            }
            x++;
            y -= kernel;
          }
        }
        // Write back to b.
        b(s, i, j, CO) = conv;
      }
    }
  }
}