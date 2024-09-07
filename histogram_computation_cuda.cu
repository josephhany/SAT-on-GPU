#include <iostream>
#include <cmath>
#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h" 

using namespace cimg_library; 
using namespace std;


__global__ void histo_privatized_kernel(int* data, int width, int height, int histo_size, int bin_size, unsigned int* histo) {
    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    extern __shared__ unsigned int histo_s[];
    if (threadIdx.x < histo_size && threadIdx.y == 0) histo_s[threadIdx.x] = 0;
        __syncthreads();
    for(unsigned int i=tidx; i<height; i+=blockDim.x * gridDim.x) {
        for(unsigned int j=tidy; j<width; j+=blockDim.y * gridDim.y){
            int pixel = data[i*width + j];
            if(pixel>=0 && pixel<256) {
                atomicAdd(&(histo_s[pixel/bin_size]), 1);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < histo_size && threadIdx.y == 0)
        atomicAdd(&(histo[threadIdx.x]), histo_s[threadIdx.x]);
}

void check_cuda_error(cudaError_t err) {
    if (err!= cudaSuccess) {
      printf("%s",
      cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
}

void gpu_Histogram(int* data, int width, int height, int histo_size, int bin_size, unsigned int* histo) {
    int size_of_data = width * height * sizeof(int);
    int size_of_histo = histo_size * sizeof(unsigned int);

    int *d_data;
    unsigned int *d_histo;

    // 1. Transfer data device memory
 
    cudaError_t err = cudaMalloc((void **) &d_data, size_of_data);
    check_cuda_error(err);
    err = cudaMemcpy(d_data, data, size_of_data, cudaMemcpyHostToDevice);
    check_cuda_error(err);
 
    // 2. Allocate device memory
    err = cudaMalloc((void **) &d_histo, size_of_histo);
    check_cuda_error(err);    

    // 3. Kernel invocation code
    // Assume that block_dim is the same as histo_size to make sure you have enough threads per block to write results
    // an offset can be added here to the block size
    int block_dim = histo_size;
 
    dim3 dimGrid(ceil((float)height/block_dim), ceil((float)width/block_dim), 1);
    dim3 dimBlock(block_dim, block_dim, 1);

    // Add third parameter to determine teh size for the shared memory

    clock_t kernel_start = clock();
    histo_privatized_kernel<<<dimGrid, dimBlock, histo_size*sizeof(unsigned int)>>>(d_data, width, height, histo_size, bin_size, d_histo);
    cudaDeviceSynchronize();
    clock_t kernel_end = clock();
    double kernel_time_spent = (double)(kernel_end - kernel_start) / CLOCKS_PER_SEC;
    cout<< "Time for kernel function: " << kernel_time_spent << "msec" << endl;

    // 4. Transfer histogram from device to host
    err = cudaMemcpy(histo, d_histo, size_of_histo, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    
    // 5. Free device memory
    err = cudaFree(d_data);
    check_cuda_error(err);
    err = cudaFree(d_histo);
    check_cuda_error(err);
}

void sequential_Histogram(int* data, int length, int bin_size, unsigned int* histo) {
    for(int i=0; i<length; i++) {
        if(data[i]>=0 && data[i] <256) {
            histo[data[i]/bin_size]++;
        }
    }
}


int main(int argc, char** argv) {

    // First, prepare the input --------------------------------------------------------------------------
    char file_name[200];
    int histo_size;
    cout<< "Please enter the greyscale image name: ";
    cin>>file_name;
    cout<< "Please enter the number of histogram bins: ";
    cin>>histo_size;

    CImg<int> img(file_name);
    int width = img.width(), height = img.height(), depth = img.depth();

    if(depth != 1){
        cout<< "The image is not greyscale.";
        exit(1);
    }
    cout << "Image data:: width: " << width << ", height: " << height << ", depth: " << depth <<endl;
    cout<<endl<<endl;

    const int bin_size = ceil(256.0 / histo_size);


    // Second, calculate and time the sequential method for the histogram -----------------------------------
    cout<< "Sequntial Method\n\n";
    unsigned int *sequential_histo = new unsigned int[histo_size];
    memset(sequential_histo, 0, histo_size*sizeof(int));

    clock_t seq_start = clock();
    sequential_Histogram(img.data(), width*height*depth, bin_size, sequential_histo);
    clock_t seq_stop = clock();
    double seq_time_spent = (double)(seq_stop - seq_start) / CLOCKS_PER_SEC;
    cout<< "Time for sequential function: " << seq_time_spent << "msec" << endl;
    cout<< "Sequential Output:\n";
    for(int i=0; i<histo_size; i++){
        cout<< sequential_histo[i] << " ";
    }
    cout<<endl;
    
    cout<<endl<<endl;


    // Third, calculate and time the GPU method for the histogram --------------------------------------------
    cout<< "GPU Method\n\n";
    unsigned int *gpu_histo = new unsigned int[histo_size];
    
    clock_t wrapper_start = clock();
    gpu_Histogram(img.data(), width, height, histo_size, bin_size, gpu_histo);
    clock_t wrapper_stop = clock();
    double wrapper_time_spent = (double)(wrapper_stop - wrapper_start) / CLOCKS_PER_SEC;

    cout<< "Time for wrapper function: " << wrapper_time_spent << "msec"<< endl;
    cout<< "Parallel Output:\n";
    for(int i=0; i<histo_size; i++){
        cout<< gpu_histo[i] << " ";
    }
    cout<<endl;

    cout<<endl<<endl;

    // Fourth, comparing both methods ----------------------------------------------------------------------

    double gflop = (2.0*width*height)*1E-9;
    cout << "Gflops: "<< gflop <<endl;
    cout << "Sequential Gflop / msec : " << gflop /  (seq_time_spent) <<endl;
    cout << "Parallel Gflop / msec : " << gflop /  (wrapper_time_spent) <<endl;
    cout << "Speed Up = " << seq_time_spent / wrapper_time_spent << " times speed up\n";

    cout<<endl<<endl;

    // Finally compare both outputs ---------------------------------------------------------------------------
    for(int i=0; i<histo_size; i++){
        if(gpu_histo[i] != sequential_histo[i]) {
            cout<< "ERROR: Parallel and Sequential Ouputs Do Not Match";
            return 0;
        }
    }
    cout<< "Success: Parallel and Sequential Ouputs Match";
}
