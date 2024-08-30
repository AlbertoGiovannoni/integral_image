#include <stdio.h>
#include <cuda_runtime.h>
#include "define.h"

__global__ void transpose(long long int *idata,long long int *odata) {
    __shared__ long long int tile[TILE_SIZE_T][TILE_SIZE_T + 1]; // con TILE_SIZE + 1 si rimuovono i bank conflicts

    // coordinate globali
    int x = blockIdx.x * TILE_SIZE_T + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE_T + threadIdx.y;
    int width = gridDim.x * TILE_SIZE_T; // larghezza immagine

    // caricamento input nel tile condiviso
    for (int j = 0; j < TILE_SIZE_T; j += BLOCK_SIZE_T)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    // scambio degli indici di blocco per ottenere la trasposizione
    x = blockIdx.y * TILE_SIZE_T + threadIdx.x;
    y = blockIdx.x * TILE_SIZE_T + threadIdx.y;

    // scrittura risultato in output
    for (int j = 0; j < TILE_SIZE_T; j += BLOCK_SIZE_T) {
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}