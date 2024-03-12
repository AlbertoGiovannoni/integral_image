#include <iostream>
#include <vector>

void imagePrint(const std::vector<std::vector<int>>& image) {
    int height = image.size();
    int width = image[0].size();

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << image[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
std::vector<std::vector<int>> createImage(int height, int width){
    std::vector<std::vector<int>> image;
    int c = 1;

    for (int i = 0; i < height; i++) {
        std::vector<int> row;
        for (int j = 0; j < width; j++) {
            row.push_back(c);
            c++;
        }
        image.push_back(row);
    }

    return image;
}

std::vector<std::vector<int>> integralImage(std::vector<std::vector<int>> image){
    int height = image.size();
    int width = image[0].size();
    std::vector<std::vector<int>> integImg(height, std::vector<int>(width, 0));


    return integImg;
}

int main() {
    int height = 10;
    int width = 10;

    std::vector<std::vector<int>> image = createImage(height, width);
    imagePrint(image);

    std::vector<std::vector<int>> integImg = integralImage(image);
    imagePrint(integImg);

    return 0;
}