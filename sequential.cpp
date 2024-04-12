#include <iostream>
#include <vector>
#include <chrono>

void imagePrint(long long int* image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            std::cout << image[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

long long int* createImage(int height, int width) {
    long long int* image = (long long int*)malloc(width * height * sizeof(long long int));
    for (int i = 0; i < width * height; i++) {
        image[i] = i + 1;
    }
    return image;
}

long long int* integralImage_naive(long long int* image, int width, int height) {
    long long int* integImg = (long long int*)malloc(width * height * sizeof(long long int));

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int sum = 0;
            for (int j = 0; j <= h; j++) {
                for (int i = 0; i <= w; i++) {
                    sum += image[j * width + i];
                }
            }
            integImg[h * width + w] = sum; // Assegna il valore della somma all'immagine integrale
        }
    }

    return integImg;
}

long long int* integralImage_optimized(long long int* image, long long int* integImg, int width, int height) {

    // Calcolo la prima riga
    integImg[0] = image[0];
    for (int i = 1; i < width; i++)
        integImg[i] = integImg[i - 1] + image[i];

    // Calcolo la prima colonna
    for (int i = 1; i < height; i++)
        integImg[i * width] = integImg[(i - 1) * width] + image[i * width];

    // Calcolo il resto
    for (int i = 1; i < height; i++) {
        for (int j = 1; j < width; j++) {
            integImg[i * width + j] = (integImg[(i - 1) * width + j] +
                                       integImg[i * width + j - 1] -
                                       integImg[(i - 1) * width + j - 1] +
                                       image[i * width + j]);
        }
    }

    return integImg;
}

int main() {
    int height = 1024*8;
    int width = 1024*8;
    bool printOutput = false;
    bool naive = false;

    long long int* image = createImage(height, width);
    long long int* integImg = (long long int*)malloc(width * height * sizeof(long long int));
    long long int* integImg_opt = (long long int*)malloc(width * height * sizeof(long long int));

    if (printOutput) {
        std::cout << "Immagine di input" << std::endl;
        imagePrint(image, width, height);
        std::cout << "\n" << std::endl;
    }

    if (naive) {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        integImg = integralImage_naive(image, width, height);
        std::chrono::duration<double> totalTime = std::chrono::steady_clock::now() - start;

        if (printOutput) {
            std::cout << "Output versione naive" << std::endl;
            imagePrint(integImg, width, height);
        }
        std::cout << "Tempo versione naive: " << totalTime.count() << " secondi\n" << std::endl;
    }

    auto start = std::chrono::steady_clock::now();
    integImg_opt = integralImage_optimized(image, integImg_opt, width, height);
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;

    if (printOutput) {
        std::cout << "Output versione ottimizzata" << std::endl;
        imagePrint(integImg_opt, width, height);
    }

    double millis = std::chrono::duration<double, std::milli>(diff).count();

    // Stampa il tempo trascorso in millisecondi
    std::cout << "Tempo di esecuzione: " << millis << " millisecondi" << std::endl;


    free(image);
    free(integImg);
    free(integImg_opt);

    return 0;
}
