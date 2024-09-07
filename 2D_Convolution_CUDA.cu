#include <iostream>
#include <cmath>
#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h" 

#define TILE_WIDTH 8
#define MASK_WIDTH 3
#define BLOCK_WIDTH TILE_WIDTH

using namespace cimg_library; 
using namespace std;

static __constant__ float Mask[MASK_WIDTH][MASK_WIDTH];


__global__ void convolution_2D_tiled_cache_kernel(float *N, int width, int height, float *P) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty; //height
    int col_o = blockIdx.x * TILE_WIDTH + tx; //width
    int row_i = row_o - MASK_WIDTH/2;
    int col_i = col_o - MASK_WIDTH/2;
    __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH];

    if(row_o < height && col_o < width){
        N_ds[ty][tx] = N[row_o*width + col_o];
        __syncthreads();

        int This_tile_start_point_x = blockIdx.x * blockDim.x; // col
        int This_tile_start_point_y = blockIdx.y * blockDim.y; // row
        int Next_tile_start_point_x = (blockIdx.x + 1) * blockDim.x;
        int Next_tile_start_point_y = (blockIdx.y + 1) * blockDim.y;

        float Pvalue = 0;
        for (int i = 0; i < MASK_WIDTH; i++) {
            int N_index_y = row_i + i;
            for (int j = 0; j < MASK_WIDTH; j++) {
                int N_index_x = col_i + j;
                if (N_index_y >= 0 && N_index_y < height && N_index_x >= 0 && N_index_x < width) {
                    if ((N_index_y >= This_tile_start_point_y) && (N_index_y < Next_tile_start_point_y
                    && N_index_x >= This_tile_start_point_x) && (N_index_x < Next_tile_start_point_x)) {
                        Pvalue += N_ds[ty+i-(MASK_WIDTH/2)][tx+j-(MASK_WIDTH/2)]*Mask[i][j];
                    } else {
                        // To replicate the edges:
                        int value_y = max(0, min(height-1, N_index_y));
                        int value_x = max(0, min(width-1, N_index_x));
                        Pvalue += N[value_y*width+value_x] * Mask[i][j];
                    }
                }
            }
        }
        P[row_o*width + col_o] = max(0.0, min(255.0, Pvalue));
    }
}

void check_cuda_error(cudaError_t err) {
    if (err!= cudaSuccess) {
      printf("%s",
      cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
}


void gpu_Convolution(float *N, int width, int height, float mask[MASK_WIDTH][MASK_WIDTH], float *P) {
    int size = width * height * sizeof(float);

    float *d_N, *d_P;

    // 1. Transfer data device memory
 
    cudaError_t err = cudaMalloc((void **) &d_N, size);
    check_cuda_error(err);
    err = cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);
    check_cuda_error(err);
 
    // 2. Allocate device memory
    err = cudaMalloc((void **) &d_P, size);
    check_cuda_error(err); 

    // 3. Copy the mask to the constant memory
    cudaMemcpyToSymbol(Mask, mask, MASK_WIDTH*MASK_WIDTH*sizeof(float));

    // 3. Kernel invocation code
    dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH,1);
    dim3 dimGrid(ceil(width/(float)TILE_WIDTH), ceil(height/(float)TILE_WIDTH), 1);

    clock_t kernel_start = clock();
    convolution_2D_tiled_cache_kernel<<<dimGrid, dimBlock>>>(d_N, width, height, d_P);
    cudaDeviceSynchronize();
    clock_t kernel_end = clock();
    double kernel_time_spent = (double)(kernel_end - kernel_start) / CLOCKS_PER_SEC;
    cout<< "Time for kernel function: " << kernel_time_spent << "msec" << endl;

    // 4. Transfer histogram from device to host
    err = cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    
    // 5. Free device memory
    err = cudaFree(d_N);
    check_cuda_error(err);
    err = cudaFree(d_P);
    check_cuda_error(err);
}



void sequential_Convolution(float *data, int width, int height, float mask[MASK_WIDTH][MASK_WIDTH], float *conv) {
    for(int i=0; i<height; i++) {
        for(int j=0; j<width; j++) {
            int st_i = i - MASK_WIDTH/2, st_j = j - MASK_WIDTH/2;
            float output = 0;
            for(int mi = 0; mi < MASK_WIDTH; mi++){
                for(int mj = 0; mj < MASK_WIDTH; mj++){
                    if(st_i + mi >= 0 && st_i + mi < height && st_j + mj >= 0 && st_j + mj < width)
                        output += mask[mi][mj] * data[(st_i+mi)*width+(st_j+mj)];
                    else {
                        // To replicate the edges:
                        int value_i = max(0, min(st_i+mi, height-1));
                        int value_j = max(0, min(st_j+mj, width-1));
                        output += mask[mi][mj] * data[value_i*width+value_j];
                    }
                }
            }
            conv[i*width+j] = max(0.0, min(255.0, output));
        }
    }
}

int main(int argc, char** argv) {

    // First, prepare the input --------------------------------------------------------------------------
    char file_name[200];
    int mask_num;
    cout<< "Please enter the greyscale image name: ";
    cin>>file_name;
    cout<< "Please enter the number of the mask to be used: \n1. Blur\n2. Emboss\n3. Outline\n4. Sharpen\n5. Left Sobel\n6. Right Sobel\n7. Top Sobel\n8. Bottom Sobel\nYour choice: ";
    cin>>mask_num;

    CImg<float> img(file_name);
    int width = img.width(), height = img.height(), depth = img.depth();

    if(depth != 1){
        cout<< "The image is not greyscale.";
        exit(1);
    }
    cout << "Image data:: width: " << width << ", height: " << height << ", depth: " << depth <<endl;

    float mask[MASK_WIDTH][MASK_WIDTH];
 
    {
    if(mask_num == 1) {
        float temp[MASK_WIDTH][MASK_WIDTH] = {{0.0625, 0.125, 0.0625}, {0.125, 0.25, 0.125}, {0.0625, 0.125, 0.0625}};
            for(int i=0; i<MASK_WIDTH; i++){
                for(int j=0; j<MASK_WIDTH; j++) mask[i][j] = temp[i][j];
            }
    }
    else if(mask_num == 2) {
        float temp[MASK_WIDTH][MASK_WIDTH] = {{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}};
        for(int i=0; i<MASK_WIDTH; i++){
            for(int j=0; j<MASK_WIDTH; j++) mask[i][j] = temp[i][j];
        }             
    }
    else if(mask_num == 3) {
        float temp[MASK_WIDTH][MASK_WIDTH] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
        for(int i=0; i<MASK_WIDTH; i++){
            for(int j=0; j<MASK_WIDTH; j++) mask[i][j] = temp[i][j];
        }      
    }
    else if(mask_num == 4) {
        float temp[MASK_WIDTH][MASK_WIDTH] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
        for(int i=0; i<MASK_WIDTH; i++){
            for(int j=0; j<MASK_WIDTH; j++) mask[i][j] = temp[i][j];
        }      
    }
    else if(mask_num == 5) {
        float temp[MASK_WIDTH][MASK_WIDTH] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
        for(int i=0; i<MASK_WIDTH; i++){
            for(int j=0; j<MASK_WIDTH; j++) mask[i][j] = temp[i][j];
        }     
    }
    else if(mask_num == 6) {
        float temp[MASK_WIDTH][MASK_WIDTH] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        for(int i=0; i<MASK_WIDTH; i++){
            for(int j=0; j<MASK_WIDTH; j++) mask[i][j] = temp[i][j];
        }     
    }
    else if(mask_num == 7) {
        float temp[MASK_WIDTH][MASK_WIDTH] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
        for(int i=0; i<MASK_WIDTH; i++){
            for(int j=0; j<MASK_WIDTH; j++) mask[i][j] = temp[i][j];
        }     
    }
    else if(mask_num == 8) {
        float temp[MASK_WIDTH][MASK_WIDTH] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        for(int i=0; i<MASK_WIDTH; i++){
            for(int j=0; j<MASK_WIDTH; j++) mask[i][j] = temp[i][j];
        }      
    }
    else {
        cout<< "Incorrect choice for mask!";
        exit(1);
    }
    }
    
    cout<< "The chosen mask to apply is:\n";
    for(int i=0; i<MASK_WIDTH; i++){
        for(int j=0; j<MASK_WIDTH; j++) {
            cout<< mask[i][j] << " ";
        }
        cout<<endl;
    }
    cout<<endl<<endl;

    // Second, calculate and time the sequential method for the convolution -----------------------------------
    cout<< "Sequntial Method\n\n";
    float *sequential_conv = new float[width*height];

    clock_t seq_start = clock();
    sequential_Convolution(img.data(), width, height, mask, sequential_conv);
    clock_t seq_stop = clock();
    double seq_time_spent = (double)(seq_stop - seq_start) / CLOCKS_PER_SEC;
    cout<< "Time for sequential function: " << seq_time_spent << "msec" << endl;

    CImg<float> img_conv_seq(sequential_conv, width, height, depth, 1);
    char seq_filename[200] = "seq_";
    strcat(seq_filename, file_name);
    img_conv_seq.save(seq_filename);
    cout<< "Sequential Output is saved in " << seq_filename <<endl;

    cout<<endl;
    
    cout<<endl<<endl;

    // Third, calculate and time the GPU method for the convolution --------------------------------------------
    cout<< "GPU Method\n\n";
    float *gpu_conv = new float[width*height];
    
    clock_t wrapper_start = clock();
    gpu_Convolution(img.data(), width, height, mask, gpu_conv);
    clock_t wrapper_stop = clock();
    double wrapper_time_spent = (double)(wrapper_stop - wrapper_start) / CLOCKS_PER_SEC;

    cout<< "Time for wrapper function: " << wrapper_time_spent << "msec"<< endl;

    CImg<float> img_conv_gpu(gpu_conv, width, height, depth, 1);
    char gpu_filename[200] = "gpu_";
    strcat(gpu_filename, file_name);
    img_conv_gpu.save(gpu_filename);
    cout<< "Parallel Output is saved in " << gpu_filename <<endl;
    cout<<endl;

    cout<<endl<<endl;

    // Fourth, comparing both methods ----------------------------------------------------------------------

    double gflop = (1.0*width*height*MASK_WIDTH*MASK_WIDTH)*1E-9;
    cout << "Gflops: "<< gflop <<endl;
    cout << "Sequential Gflop / msec : " << gflop /  (seq_time_spent) <<endl;
    cout << "Parallel Gflop / msec : " << gflop /  (wrapper_time_spent) <<endl;
    cout << "Speed Up = " << seq_time_spent / wrapper_time_spent << " times speed up\n";

    cout<<endl<<endl;
}
