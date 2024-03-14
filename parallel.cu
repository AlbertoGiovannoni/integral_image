#include <stdio.h>
#include <cuda_runtime.h>


__global__ void hello() {
    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

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

int* createImage(int* img, int width, int height){
    for (int i = 0; i < width * height; ++i) {
        img[i] = i + 1;
    }
    return img;
}

int main() {
    int height = 100;
    int width = 100;
    bool printOutput = false;
    int imgSize = height * width * sizeof(int);
    int integImgSize = height * width * sizeof(int);

    // Stampa delle specifiche hardware
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Device: %s\n", deviceProp.name);
    printf("Cores number: %d\n", getSPcores(deviceProp));

    // Allocazione memoria CPU
    int* cpu_img = (int*) malloc(imgSize);
    cpu_img = createImage(cpu_img, width, height);

    // Allocazione memoria GPU
    int* gpu_img;
    int* gpu_integImg;
    cudaMalloc((void**)&gpu_img, imgSize);
    cudaMalloc((void**)&gpu_integImg, integImgSize);

    // Copia dei dati da cpu a gpu
    cudaMemcpy(gpu_img, cpu_img, imgSize, cudaMemcpyHostToDevice);


}