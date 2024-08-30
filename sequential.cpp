#include <iostream>
#include <vector>
#include <chrono>

void imagePrint(long long int *image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << image[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

long long int *createImage(long long int *image, int height, int width) {
    for (int i = 0; i < width * height; i++) {
        image[i] = i + 1;
    }
    return image;
}

long long int *integralImage_optimized(long long int *image,  int width, int height) {
    long long int *integImg = (long long int *) malloc(width * height * sizeof(long long int));

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

long long int *scan(long long int *image, int width, int height) {
    long long int tmp;

    for (int j = 0; j < height; j++) {
        tmp = 0;
        for (int i = 0; i < width; i++) {
            tmp += image[j * width + i];
            image[j * width + i] = tmp;
        }
    }

    return image;
}

long long int *transpose(long long int *image, int width, int height) {
    long long int *tmp = (long long int *) malloc(width * height * sizeof(long long int));

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            tmp[i * height + j] = image[j * width + i];
        }
    }
    return tmp;
}


int main() {
    int height = 4;
    int width = 4;
    bool printOutput = true;
    bool optimized = false;

    long long int *image = (long long int *) malloc(width * height * sizeof(long long int));

    image = createImage(image, height, width);

    if (printOutput) {
        std::cout << "Immagine di input" << std::endl;
        imagePrint(image, width, height);
    }

    auto start = std::chrono::steady_clock::now();

    if (optimized) {
        image = scan(image, width, height);

        image = transpose(image, width, height);

        image = scan(image, width, height);

        image = transpose(image, width, height);
    } else{
        image = integralImage_optimized(image, width, height);
    }

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;

    if (printOutput) {
        std::cout << "Immagine integrale: " << std::endl;
        imagePrint(image, width, height);
    }

    double millis = std::chrono::duration<double, std::milli>(diff).count();

    std::cout << "Tempo di esecuzione: " << millis << " millisecondi" << std::endl;

    free(image);

    return 0;
}