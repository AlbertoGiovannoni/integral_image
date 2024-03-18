#include <stdio.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 16


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

int *createImage(int *img, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        img[i] = i + 1;
    }
    return img;
}

__host__ void printImageValues(int *image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", image[i * width + j]);
        }
        printf("\n");
    }
}

__global__ void imageIntegral(int *img, int *integImg, int width, int height) {
    // Indici globali
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        int sum = 0;

        // Calcola l'integrale dell'immagine per la riga corrente
        for (int i = 0; i <= x; i++) {
            for (int j = 0; j <= y; j++) {
                sum += img[j * width + i];
            }
        }

        // Scrive il risultato nell'immagine integrale
        integImg[index] = sum;
    }
}

int main() {
    int height = 3;
    int width = 3;
    int block_dim = 32;
    bool printOutput = true;
    int imgSize = height * width * sizeof(int);
    int integImgSize = height * width * sizeof(int);

    // Stampa delle specifiche hardware
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Device: %s\n", deviceProp.name);
    printf("Cores number: %d\n", getSPcores(deviceProp));

    // Allocazione memoria CPU
    int *cpu_img = (int *) malloc(imgSize);
    cpu_img = createImage(cpu_img, width, height);

    // Allocazione memoria GPU
    int *gpu_img;
    int *gpu_integImg;
    cudaMalloc((void **) &gpu_img, imgSize);
    cudaMalloc((void **) &gpu_integImg, integImgSize);

    // Copia dei dati da cpu a gpu
    cudaMemcpy(gpu_img, cpu_img, imgSize, cudaMemcpyHostToDevice);

    // Grid e Block
    dim3 block(block_dim, block_dim);
    dim3 grid((width + block_dim - 1) / block_dim, (height + block_dim - 1) / block_dim);

    // Tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    imageIntegral<<<grid, block>>>(gpu_img, gpu_integImg, width, height);
    printf("%d", width*height);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    if (printOutput) {
        int *cpu_integImg = (int *) malloc(imgSize);
        cudaMemcpy(cpu_integImg, gpu_integImg, imgSize, cudaMemcpyDeviceToHost);
        printImageValues(cpu_integImg, width, height);
        free(cpu_integImg);
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tempo di esecuzione: %.2f millisecondi\n", milliseconds);


    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);
    cudaEventRecord(stop1);

    cudaEventSynchronize(stop1);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    printf("Tempo di esecuzione dfwefxews: %.2f millisecondi\n", milliseconds1);

    int *cpu_integImg = (int *) malloc(imgSize);
    cudaMemcpy(cpu_integImg, gpu_integImg, imgSize, cudaMemcpyDeviceToHost);
    printImageValues(cpu_integImg, width, height);
    printf("%d ", cpu_integImg[1]);
    free(cpu_integImg);


    free(cpu_img);
    cudaFree(gpu_img);
    cudaFree(gpu_integImg);
    return 0;
}
