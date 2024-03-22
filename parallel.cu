#include <stdio.h>
#include <cuda_runtime.h>
#include "transpose.cu"
#include "scan.cu"
#include "define.h"

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
            else printf("Unknown device type\n");
            break;
        case 9: // Hopper
            if (devProp.minor == 0) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

long long int *createImage(long long int *img, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        img[i] = i + 1;
    }
    return img;
}

__host__ void printValues(long long int *image, int width, int height, const char *label, const long long int longBytes) {
    long long int *h_output = (long long int *) malloc(longBytes);
    cudaMemcpy(h_output, image, longBytes, cudaMemcpyDeviceToHost);

    printf("%s", label);
    int k = 0;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
                printf("%lld ", h_output[k]);
            k++;
        }
        printf("\n");
    }
}

int main() {
    bool printOutput = false;
    const long long int longBytes = INPUT_SIZE * INPUT_SIZE * sizeof(long long int);

    // Stampa delle specifiche hardware
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Device: %s\n", deviceProp.name);
    printf("Cores number: %d\n", getSPcores(deviceProp));

    // Popolo la matrice
    long long int *h_input = (long long int *) malloc(longBytes);
    h_input = createImage(h_input, INPUT_SIZE, INPUT_SIZE);


    // tutte le malloc

    // T1
    long long int *d_outputT1;
    cudaMalloc((void **) &d_outputT1, longBytes);
    dim3 dimGrid(INPUT_SIZE / TILE_SIZE_T, INPUT_SIZE / TILE_SIZE_T, 1);
    dim3 dimBlock(TILE_SIZE_T, BLOCK_SIZE_T, 1);

    // T2
    long long int *d_outputT2;
    cudaMalloc((void **) &d_outputT2, longBytes);




    cudaEvent_t startTot, stopTot;
    cudaEventCreate(&startTot);
    cudaEventCreate(&stopTot);

    // Scan 1
    long long int *d_input, *d_outputS;
    cudaMalloc((void **) &d_input, longBytes);
    cudaMemcpy(d_input, h_input, longBytes, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_outputS, longBytes);
    long long int *sum; // vettore somme parziali
    cudaMalloc((void **) &sum, longBytes);

    long long int *h_output = (long long int *) malloc(longBytes);



    cudaEventRecord(startTot);
    scan(d_input, d_outputS, sum, INPUT_SIZE, BLOCK_SIZE_S);

    if(printOutput)
        printValues(d_outputS, INPUT_SIZE, INPUT_SIZE, "Scan 1 results: \n", longBytes);

    // Transpose 1
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_outputS, d_outputT1);

    if(printOutput)
        printValues(d_outputT1, INPUT_SIZE, INPUT_SIZE, "Transpose 1 results: \n", longBytes);

    // Scan 2
    scan(d_outputT1, d_outputS, sum, INPUT_SIZE, BLOCK_SIZE_S);

    if(printOutput)
        printValues(d_outputS, INPUT_SIZE, INPUT_SIZE, "Scan 2 results: \n", longBytes);

    // Trasposta perchè fin qui ho la trasposta dell immagine integrale
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_outputS, d_outputT2);

    cudaEventRecord(stopTot);
    cudaEventSynchronize(stopTot);

    float millisecondsTot = 0;
    cudaEventElapsedTime(&millisecondsTot, startTot, stopTot);

    printf("\nTotal time: %f milliseconds\n", millisecondsTot);


    if(printOutput)
        printValues(d_outputT2, INPUT_SIZE, INPUT_SIZE, "Transpose 2 results: \n", longBytes);


    // Deallocazione della memoria
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_outputS);

    return 0;
}

