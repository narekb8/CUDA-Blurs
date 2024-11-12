#include <iostream>
#include <math.h>
#include <stdint.h>
#include <cuda.h>

// Struct for Pixel data
#pragma pack(1)
typedef struct Pixel32 {
    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t a;
} Pixel;

/*
GaussianKernel
Function for generating a Gaussian kernel.

Params:
float *kernel: float buffer for kernel values
int kernelRadius: Kernel radius
int kernelWidth: Kernel width
float sigma: Sigma value of the kernel
*/
void GaussianKernel(float *kernel, int kernelRadius, int kernelWidth, float sigma)
{ 
    float r, s = 2.0 * sigma * sigma;
    float sum = 0.0;
  
    for (int y = -kernelRadius; y <= kernelRadius; y++) { 
        for (int x = -kernelRadius; x <= kernelRadius; x++) { 
            r = sqrt(x * x + y * y);
            int i = y + kernelRadius;
            int j = x + kernelRadius;
            kernel[i*kernelWidth+j] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += kernel[i*kernelWidth+j]; 
        } 
    }
  
    for (int i = 0; i < kernelWidth; ++i) 
        for (int j = 0; j < kernelWidth; ++j) 
            kernel[i*kernelWidth+j] /= sum; 
}

/*
ApplyBlur
CUDA Kernel for applying a blur to an individual pixel.

Params:
Pixel *img: input buffer of image data
Pixel *imgf: output buffer of image data
float *kernel: float buffer for kernel values
int nx: width of image
int ny: height of image
int kernelWidth: Kernel width
*/
__global__
void ApplyBlur(Pixel *img, Pixel *imgf, float *kernel, int nx, int ny, int kernelWidth)
{
    int kernelRadius = (kernelWidth - 1)/2;
    int tx = threadIdx.x;
    int ty = blockIdx.x;
    int pixelIndex = ty*nx+tx;

    float redVal = 0;
    float greenVal = 0;
    float blueVal = 0;
    float alphaVal = 0;

    for(int j = 0; j < kernelWidth; j++)
    {
        for(int k = 0; k < kernelWidth; k++)
        {
            int ix = k + tx - kernelRadius;
            int iy = j + ty - kernelRadius;
            Pixel pixel = img[iy*nx+ix];
            redVal += pixel.r * kernel[j*kernelWidth+k];
            greenVal += pixel.g * kernel[j*kernelWidth+k];
            blueVal += pixel.b * kernel[j*kernelWidth+k];
            alphaVal += pixel.a * kernel[j*kernelWidth+k];
        }
    }

    Pixel outPixel;
    outPixel.r = redVal;
    outPixel.g = greenVal;
    outPixel.b = blueVal;
    outPixel.a = alphaVal;

    imgf[pixelIndex] = outPixel;
}

/*
isNumber
Helper function to ensure command line argument is an integer.
Source: https://stackoverflow.com/questions/29248585/c-checking-command-line-argument-is-integer-or-not

Params:
const char number[]: char array input
*/
bool isNumber(const char number[])
{
    int i = 0;
    
    if (number[0] == '-')
        i = 1;
    for (; number[i] != 0; i++)
    {
        if (!isdigit(number[i]))
            return false;
    }
    return true;
}



int main(int argc, char *argv[])
{
    uint32_t nx = 0; // Width of image
    uint32_t ny = 0; // Height of image
    uint32_t n = 0; // Total pixel count
    uint32_t offset = 0; // Byte offset to image data

    if(argc < 3)
    {
        std::cout << "Argument count too low!" << std::endl;
        std::cout << "Please run the program with the following options <filepath> <radius>." << std::endl;
        exit(1);
    }

    int kernelRadius; // Radius of kernel
    if(isNumber(argv[2]))
        kernelRadius = atoi(argv[2]);
    else
    {
        std::cout << "Size argument is not an integer!" << std::endl;
        exit(1);
    }
    int kernelWidth = 2 * kernelRadius + 1; // Width of kernel
    float sigma = std::max(kernelRadius / 2.0f, 1.0f); // Sigma value for generating kernel

    // Open input file for reading
    FILE *fptr;
    fptr = fopen(argv[1], "r+");
    if(fptr == NULL)
    {
        std::cout << "File could not be opened!" << std::endl;
        exit(1);
    }

    // Check to ensure that the input file is a valid file, and read in the important header data
    // Current constraints: Width cannot exceed 1024 (max thread count per block on CUDA), 32-bit color is the only supported format
    std::string fileType(2, '\0');
    fread(&fileType[0], sizeof(char), 2, fptr);
    if(fileType.compare("BM") != 0)
    {
        std::cout << "File not a valid bitmap file!" << std::endl;
        exit(1);
    }
    fseek(fptr, 8, SEEK_CUR);
    fread(&offset, 4, 1, fptr);
    fseek(fptr, 4, SEEK_CUR);
    fread(&nx, 4, 1, fptr);
    if(nx > 1024)
    {
        std::cout << "File width exceeds 1024 pixels! Please choose a smaller width file." << std::endl;
    }
    fread(&ny, 4, 1, fptr);
    fseek(fptr, 2, SEEK_CUR);
    uint16_t bitCount;
    fread(&bitCount, 2, 1, fptr);
    if(bitCount != 32)
    {
        std::cout << "File is not 32-bit color!" << std::endl;
        exit(1);
    }
    else
        std::cout << "File opened!" << std::endl;

    n = nx * ny;

    // Allocate memory for the incoming image and kernel data
    Pixel *img, *imgOut;
    img = (Pixel *)malloc(sizeof(Pixel) * n);
    imgOut = (Pixel *)malloc(sizeof(Pixel) * n);
    float *kernel = (float *)malloc(kernelWidth * kernelWidth * sizeof(float));

    // Read in data, chunking it in to prevent fread limit
    fseek(fptr, offset, SEEK_SET);
    for(int i = 0; fread(img + i*1000, sizeof(Pixel), 1000, fptr) == 1000; i++);

    GaussianKernel(kernel, kernelRadius, kernelWidth, sigma);

    // Allocate GPU memory for image and kernel data
    Pixel *cuImg, *cuOut;
    float *cuKernel;
    cudaMalloc(&cuKernel, kernelWidth*kernelWidth*sizeof(float));
    cudaMalloc(&cuImg, n*sizeof(Pixel));
    cudaMalloc(&cuOut, n*sizeof(Pixel));

    // Copy memory from CPU to GPU 
    cudaError_t toGPU = cudaMemcpy(cuImg, img, n*sizeof(Pixel), cudaMemcpyHostToDevice);
    //cudaMemcpy(cuOut, img, n*sizeof(Pixel), cudaMemcpyHostToDevice);
    cudaMemcpy(cuKernel, kernel, kernelWidth*kernelWidth*sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Deployed to GPU" << std::endl;
    ApplyBlur<<<ny, nx>>>(cuImg, cuOut, cuKernel, nx, ny, kernelWidth);

    // Synchronize all threads in order to ensure they are all finished running
    cudaDeviceSynchronize();
    cudaError_t fromGPU = cudaMemcpy(imgOut, cuOut, n*sizeof(Pixel), cudaMemcpyDeviceToHost);
    std::cout << cudaGetErrorName(fromGPU) << " | " << cudaGetErrorString(fromGPU) << std::endl;

    // Read file header in to save in the new file
    uint8_t *fileHeader = (uint8_t *)malloc(offset);
    fseek(fptr, 0, SEEK_SET);
    fread(fileHeader, offset, 1, fptr);

    // Write to new file
    FILE *wptr;
    wptr = fopen("blurred.bmp", "w+");
    fwrite(fileHeader, offset, 1, wptr);
    fwrite(imgOut, sizeof(Pixel), n, wptr);

    // Deallocate all memory and close file streams
    cudaFree(cuImg);
    cudaFree(cuOut);
    cudaFree(cuKernel);

    free(img);
    free(kernel);
    free(imgOut);

    fclose(fptr);
    fclose(wptr);

    return 0;
}