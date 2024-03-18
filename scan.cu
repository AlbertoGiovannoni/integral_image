#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void scan(int *XY, int InputSize, int *Y) {
    __shared__ int XY_shared[2 * BLOCK_SIZE];

    int idx = threadIdx.x;
    XY_shared[2 * idx] = (2 * idx < InputSize) ? XY[2 * idx] : 0;
    XY_shared[2 * idx + 1] = (2 * idx + 1 < InputSize) ? XY[2 * idx + 1] : 0;

    // Up Sweep
    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        __syncthreads();
        int index = (idx + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE)
            XY_shared[index] += XY_shared[index - stride];
    }

    // Down Sweep
    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (idx + 1) * stride * 2 - 1;
        if (index + stride < 2 * BLOCK_SIZE) {
            XY_shared[index + stride] += XY_shared[index];
        }
    }
    __syncthreads();

    if (idx < InputSize)
        Y[idx] = XY_shared[idx];
}

int main() {
    const int InputSize = 16;
    const int arrayBytes = InputSize * sizeof(int);

    int h_XY[InputSize] = {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16};
    int *d_XY, *d_Y;

    cudaMalloc((void **)&d_XY, arrayBytes);
    cudaMemcpy(d_XY, h_XY, arrayBytes, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_Y, arrayBytes);

    scan<<<1, BLOCK_SIZE>>>(d_XY, InputSize, d_Y);

    int h_Y[InputSize];
    cudaMemcpy(h_Y, d_Y, arrayBytes, cudaMemcpyDeviceToHost);

    printf("Inclusive Scan Result:\n");
    for (int i = 0; i < InputSize; i++) {
        printf("%d ", h_Y[i]);
    }
    printf("\n");

    cudaFree(d_XY);
    cudaFree(d_Y);

    return 0;
}
