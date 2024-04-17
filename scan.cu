#include <stdio.h>
#include <cuda_runtime.h>
#include "define.h"


__global__ void scanParallel(long long int *d_input, long long int *d_output, int inputSize, long long int *sum) {
    __shared__ long long int tmp[2 * BLOCK_SIZE_S];
    int row = blockIdx.x;
    int idx = threadIdx.x;
    int offset = row * inputSize;

    tmp[2 * idx] = d_input[offset + idx * 2]; // offset + idx rappresenta l'indice globale
    tmp[2 * idx + 1] = d_input[offset + idx * 2 + 1];

    // Up Sweep
    for (unsigned int stride = 1; stride <= BLOCK_SIZE_S; stride *= 2) {
        __syncthreads();
        int index = (idx + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE_S)
            tmp[index] += tmp[index - stride];
    }

    // Down Sweep
    for (unsigned int stride = BLOCK_SIZE_S / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (idx + 1) * stride * 2 - 1;
        if (index + stride < 2 * BLOCK_SIZE_S) {
            tmp[index + stride] += tmp[index];
        }
    }
    __syncthreads();

    d_output[offset + idx] = tmp[idx];

    if (idx == BLOCK_SIZE_S - 1) {
        sum[row] = tmp[idx];
    }

}

__global__ void add(long long int *output, int length, long long int *n) {
    int blockID = blockIdx.x;

    int blockOffset = blockID * length;
    output[blockOffset] += n[blockID];
}

void scan(long long int *d_input, long long int *d_output, long long int *sum,int inputSize, int blockSize){
    const int numBlocks = (inputSize + blockSize - 1) / blockSize;

    for (int i = 0; i < numBlocks; i++) {
        // Calcola l'offset in base al blocco corrente
        int offset = i * blockSize;

        // Esegui la scan sulla porzione corrente del vettore di input
        scanParallel<<<inputSize, blockSize>>>(d_input + offset, d_output + offset, inputSize, sum);
        // Esegui l'add sulla porzione corrente del vettore di output
        if (i < numBlocks - 1)
            add<<<inputSize, 1>>>(d_input + blockSize * (i + 1), inputSize, sum);
    }
}

/*int main2() {
    const int inputSize = INPUT_SIZE;
    const int blockSize = BLOCK_SIZE;
    bool printOutput = true;
    //const int inputBytes = inputSize * inputSize * sizeof(int); // matrici quadrate
    const long long int longBytes = inputSize * inputSize * sizeof(long long int);
    const int numBlocks = (inputSize + blockSize - 1) / blockSize;


    long long int *h_input = (long long int *) malloc(longBytes);
    // Popolo la matrice
    int c = 1;
    for (int j = 0; j < inputSize; j++) {
        for (int i = 0; i < inputSize; i++) {
            h_input[j * inputSize + i] = c;
            c++;
        }
    }

    long long int *h_output = (long long int *) malloc(longBytes);
    //scan(h_input, h_output, inputSize, blockSize, printOutput);


    free(h_input);
    free(h_output);
    return 0;
}*/
