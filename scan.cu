#include <stdio.h>
#include <cuda_runtime.h>
#include "define.h"


__global__ void scanParallel(long long int *d_input, long long int *d_output, int inputSize, long long int *sum) {
    __shared__ long long int tmp[2 * BLOCK_SIZE_S];
    int row = blockIdx.x;
    int idx = threadIdx.x;
    int offset = row * inputSize;

    // caricamento input nella memoria condivisa
    tmp[2 * idx] = d_input[offset + idx * 2]; // offset + idx rappresenta l'indice globale
    tmp[2 * idx + 1] = d_input[offset + idx * 2 + 1];

    // Up Sweep
    for (unsigned int stride = 1; stride <= BLOCK_SIZE_S; stride *= 2) {
        __syncthreads();
        int index = (idx + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE_S)
            tmp[index] += tmp[index - stride]; // accumulo valore a distanza "stride"
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

    // scrittura risultato in output
    d_output[offset + idx] = tmp[idx];

    // salvataggio dell'ultimo elemento del blocco in sum
    if (idx == BLOCK_SIZE_S - 1) {
        sum[row] = tmp[idx];
    }

}

__global__ void add(long long int *output, int length, long long int *sum) {
    int blockID = blockIdx.x;

    int blockOffset = blockID * length;
    output[blockOffset] += sum[blockID];
}

void scan(long long int *d_input, long long int *d_output, long long int *sum,int inputSize, int blockSize){
    const int numBlocks = (inputSize + blockSize - 1) / blockSize;

    for (int i = 0; i < numBlocks; i++) {
        // Calcola l'offset in base al blocco corrente
        int offset = i * blockSize;

        // Esegui la scan sulla porzione corrente del vettore di input
        scanParallel<<<inputSize, blockSize>>>(d_input + offset, d_output + offset, inputSize, sum);

        // Aggiunge l'ultimo elemento del blocco precedente al primo del successivo
        if (i < numBlocks - 1)
            add<<<inputSize, 1>>>(d_input + blockSize * (i + 1), inputSize, sum);
    }
}