#include <stdio.h>
#include <cuda_runtime.h>
#include "define.h"


 __global__ void naiveTranspose(int *input, int *output, int width, int height) {
    __shared__ int temp[BLOCK_SIZE_T][BLOCK_SIZE_T + 1];

    int xIndex = blockIdx.x * BLOCK_SIZE_T + threadIdx.x;
    int yIndex = blockIdx.y * BLOCK_SIZE_T + threadIdx.y;

    if ((xIndex < width) && (yIndex < height)) {
        int id_in = yIndex * width + xIndex;
        temp[threadIdx.y][threadIdx.x] = input[id_in];
    }
    __syncthreads();
    // inversione indici
    xIndex = blockIdx.y * BLOCK_SIZE_T + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_SIZE_T + threadIdx.y;
    if ((xIndex < height) && (yIndex < width)) {
        int id_out = yIndex * height + xIndex;
        output[id_out] = temp[threadIdx.x][threadIdx.y];
    }
}

__global__ void transposeCoalesced(long long int *idata,long long int *odata) {
    __shared__ long long int tile[TILE_SIZE_T][TILE_SIZE_T + 1]; // con TILE_SIZE + 1 si rimuovono i bank conflicts

    int x = blockIdx.x * TILE_SIZE_T + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE_T + threadIdx.y;
    int width = gridDim.x * TILE_SIZE_T;

    for (int j = 0; j < TILE_SIZE_T; j += BLOCK_SIZE_T)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_SIZE_T + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_SIZE_T + threadIdx.y;

    for (int j = 0; j < TILE_SIZE_T; j += BLOCK_SIZE_T) {
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

/*int main3() {
    bool printOutput = false;
    const int inputSize = INPUT_SIZE;
    const int inputBytes = inputSize * inputSize * sizeof(int); // matrici quadrate
    const long long int longBytes = inputSize * inputSize * sizeof(long long int);


    long long int *h_input = (long long int *) malloc(longBytes);
    // Popolo la matrice
    int c = 1;
    for (int j = 0; j < inputSize; j++) {
        for (int i = 0; i < inputSize; i++) {
            h_input[j * inputSize + i] = c;
            c++;
        }
    }

    long long int *d_input, *d_output;

    cudaMalloc((void **) &d_input, longBytes);
    cudaMemcpy(d_input, h_input, longBytes, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_output, longBytes);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    //naiveTranspose<<<inputSize, block>>>(d_input, d_output, inputSize, inputSize);

    dim3 dimGrid(inputSize / TILE_SIZE, inputSize / TILE_SIZE, 1);
    dim3 dimBlock(TILE_SIZE, BLOCK_SIZE, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_input, d_output);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Transpose time: %f milliseconds\n", milliseconds);

    long long int *h_output = (long long int *) malloc(longBytes);
    cudaMemcpy(h_output, d_output, longBytes, cudaMemcpyDeviceToHost);

    if (printOutput) {
        printf("Scan Result:\n");
        int k = 0;
        for (int j = 0; j < inputSize; j++) {
            for (int i = 0; i < inputSize; i++) {
                printf("%lld ", h_output[k]);
                k++;
            }
            printf("\n");
        }
    }

    // Deallocazione della memoria
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}*/
