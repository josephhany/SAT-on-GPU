#include <iostream>
#define cimg_display 0
#define cimg_use_jpeg
#include "CImg.h" 

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

#define MAX_SHARED_WIDTH 100
#define MAX_SHARED_HEIGHT 100

using namespace std;
using namespace cimg_library; 

__global__
void Stage1(float* X, float* Y, float* S, int length, int width) // width = c
{
    // 2D shared memory
    __shared__ float T[BLOCK_HEIGHT][BLOCK_WIDTH]; // shared array of section size
 
    // create some variables to ease things
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
 
    // global index of this thread
    int Row = by * blockDim.y + ty; 
    int Col = bx * blockDim.x + tx; // vert arrow

    // make sure that this thread is loading the correct element from global memory
    if (Row < length && Col < width) {
        T[ty][tx] = X[Col + Row * width]; // each th loads on element from global memory into shared memory
    }
    
    //the code below performs iterative scan on T

    for (int stride = 1; stride < blockDim.x; stride *= 2){ // if we have 16 elements the biggest stride will be 8
 
      if(threadIdx.x >= stride) {
 
          __syncthreads();
          
          float temp = T[ty][tx] + T[ty][tx - stride];
          
          __syncthreads();
          
          T[ty][tx] = temp;
      }
    }
 
    if (Row < length && Col < width) Y[Col + Row * width] = T[ty][tx];
 
    if(tx == BLOCK_WIDTH-1) S[Col/BLOCK_WIDTH + Row * (width/BLOCK_WIDTH)] = T[ty][tx];

}

__global__
void Stage2(float* X, float* Y, int length, int width) // width = c
{
    // 2D shared memory
    __shared__ float T[MAX_SHARED_HEIGHT][MAX_SHARED_WIDTH]; // all the 2d arrya must fit the shared memory unless recurssion will be needed
 
    // create some variables to ease things
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
 
    // global index of this thread
    int Row = by * blockDim.y + ty; 
    int Col = bx * blockDim.x + tx; // vert arrow

    // make sure that this thread is loading the correct element from global memory
    if (Row < length && Col < width) {
        T[ty][tx] = X[Col + Row * width]; // each th loads on element from global memory into shared memory
    }
    
    //the code below performs iterative scan on T

    for (int stride = 1; stride < blockDim.x; stride *= 2){ // if we have 16 elements the biggest stride will be 8
 
      if(threadIdx.x >= stride) {
 
          __syncthreads();
          
          float temp = T[ty][tx] + T[ty][tx - stride];
          
          __syncthreads();
          
          T[ty][tx] = temp;
      }
    }
 
    if (Row < length && Col < width) Y[Col + Row * width] = T[ty][tx];
    //Y[Col + Row * width] = T[ty][tx];
}


__global__
void Stage3(float* X, float* Y, float* S, int length, int width) // width = c
{
    // create some variables to ease things
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
 
    // global index of this thread
    int Row = by * blockDim.y + ty; 
    int Col = bx * blockDim.x + tx; // vert arrow
 
    // every thread will be responsible for one output value

    if(Col >= BLOCK_WIDTH && tx != (BLOCK_WIDTH - 1)) Y[Col + Row * width] = X[Col + Row * width] + S[(Col/BLOCK_WIDTH) - 1 + Row * (width/BLOCK_WIDTH)];
    else Y[Col + Row * width] = X[Col + Row * width];
    
}

void check_cuda_error(cudaError_t err) {
    if (err!= cudaSuccess) {
      printf("%s",
      cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
}

void ScanKernel(float* MatrixA, float* MatrixC, float* MatrixS, float* MatrixS2, float* Final_out, int r, int c)
{
    int sizeA = r * c * sizeof(float);
    int sizeC = r * c * sizeof(float);
    int sizeS = r * (c/BLOCK_WIDTH) * sizeof(float);
    
 
    float *d_MatrixA, *d_MatrixC, *d_Intr_S, *d_Intr_S2, *d_final_out;
 
    // 1. Transfer MatrixA and MatrixB to device memory
 
    cudaError_t err = cudaMalloc((void **) &d_MatrixA, sizeA);
    check_cuda_error(err);
    err = cudaMemcpy(d_MatrixA, MatrixA, sizeA, cudaMemcpyHostToDevice);
    check_cuda_error(err);
 
    // Allocate device memory
    err = cudaMalloc((void **) &d_MatrixC, sizeC);
    check_cuda_error(err);
 
    // FOR DEBUGGING ONLY
    // Allocate device memory
    err = cudaMalloc((void **) &d_Intr_S, sizeS);
    check_cuda_error(err);
    // Allocate device memory
    err = cudaMalloc((void **) &d_Intr_S2, sizeS);
    check_cuda_error(err);
    // Allocate device memory
    err = cudaMalloc((void **) &d_final_out, sizeC);
    check_cuda_error(err);
 
    // 2. Kernel invocation code

    dim3 dimGrid(ceil((float)c/BLOCK_WIDTH), ceil((float)r/BLOCK_HEIGHT), 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    clock_t kernel1_start = clock();
    Stage1<<<dimGrid, dimBlock>>>(d_MatrixA, d_MatrixC, d_Intr_S, r, c);
    cudaDeviceSynchronize();
    clock_t kernel1_end = clock();
    double kernel1_time_spent = (double)(kernel1_end - kernel1_start) / CLOCKS_PER_SEC;
    cout<< "Time for kernel Stage 1 function: " << kernel1_time_spent << "msec" << endl;
 
    dim3 dimGrid2(1, 1, 1);
    dim3 dimBlock2(ceil((float)c/BLOCK_WIDTH), r, 1);
    clock_t kernel2_start = clock();
    Stage2<<<dimGrid2, dimBlock2>>>(d_Intr_S, d_Intr_S2, r, c/BLOCK_WIDTH);
    cudaDeviceSynchronize();
    clock_t kernel2_end = clock();
    double kernel2_time_spent = (double)(kernel2_end - kernel2_start) / CLOCKS_PER_SEC;
    cout<< "Time for kernel Stage 2 function: " << kernel2_time_spent << "msec" << endl;
 

    dim3 dimGrid3(ceil((float)c/BLOCK_WIDTH), ceil((float)r/BLOCK_HEIGHT), 1);
    dim3 dimBlock3(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    clock_t kernel3_start = clock();
    Stage3<<<dimGrid3, dimBlock3>>>(d_MatrixC, d_final_out, d_Intr_S2, r, c);
    cudaDeviceSynchronize();
    clock_t kernel3_end = clock();
    double kernel3_time_spent = (double)(kernel3_end - kernel3_start) / CLOCKS_PER_SEC;
    cout<< "Time for kernel Stage 3 function: " << kernel3_time_spent << "msec" << endl;
 
    
    // 3. Transfer MatrixC from device to host
 
    err = cudaMemcpy(MatrixC, d_MatrixC, sizeC, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
 
    // NEEDED FOR DEBUGGING ONLY
    err = cudaMemcpy(MatrixS, d_Intr_S, sizeS, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
 
    err = cudaMemcpy(MatrixS2, d_Intr_S2, sizeS, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
 
    err = cudaMemcpy(Final_out, d_final_out, sizeC, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
 
    
    // Free device memory for MatrixA, MatrixB, MatrixC
    err = cudaFree(d_MatrixA);
    check_cuda_error(err);
    err = cudaFree (d_MatrixC);
    check_cuda_error(err);
    
    err = cudaFree (d_Intr_S);
    check_cuda_error(err);
    err = cudaFree (d_Intr_S2);
    check_cuda_error(err);
    err = cudaFree (d_final_out);
    check_cuda_error(err);
}

// obtain a random value scaled to be within limits

inline int GetRandomValue( float minval, float maxval )
{
	return ( rand() * ( maxval - minval ) / RAND_MAX ) + minval;
}

void printMat(float* Matrix, const char *Name, int Rows, int Columns){
    printf("\nMatrix %s (%d X %d):\n\n", Name, Rows, Columns);
    for (int i = 0; i< Rows*Columns; i++){
        printf("%f ",Matrix[i]);
        if((i+1)%Columns==0 && i!=0) printf("\n");
    }
}

float * do_SAT(float* MatrixA, int M, int N){

    for(int i = 0; i < M; i++){
      for(int j = 0; j < N; j++){
         if(j != 0) MatrixA[j + i * N] = MatrixA[j + i * N - 1] + MatrixA[j + i * N];
      }
    }

    // columns prefix
    for(int j = 0; j < N; j++){
      for(int i = 0; i < M; i++){
          if(i != 0) MatrixA[j + i * N] = MatrixA[j + (i-1) * N ] + MatrixA[j + i * N];
      }
    }

    return MatrixA;
}

float * do_transpose(float* MatrixA, int M, int N){
    
    float * Transpose = (float*) malloc(sizeof(float) * M * N); // M x P 
    
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        Transpose[ i + j * M ] = MatrixA[ j + i * N ];
    
    return Transpose;
}


int main()
{
    // First, prepare the input --------------------------------------------------------------------------
    char file_name[200];
    cout<< "Please enter the greyscale image name: ";
    cin>>file_name;
    CImg<float> img(file_name);
    int N = img.width(), M = img.height(), depth = img.depth();
    if(depth != 1){
        cout<< "The image is not greyscale.";
        exit(1);
    }

    cout << "Image data:: width: " << N << ", height: " << M << ", depth: " << depth <<endl;
    cout<<endl<<endl;

    int Arow, Acol, Brow, Bcol, Crow, Ccol, Drow, Dcol;
    cout<< "Please input the points for the recatngular subarea in \"row col\" format\n";
    cin>> Arow >> Acol >> Brow >> Bcol >> Crow >> Ccol >> Drow >> Dcol;

    // Matrices of floats linearized
    float * MatrixA = (float*) malloc(sizeof(float) * M * N); // M x N
    float * MatrixC = (float*) malloc(sizeof(float) * M * N); // M x P
    float * MatrixS = (float*) malloc(sizeof(float) * M * (N/BLOCK_WIDTH)); // M x P
    float * MatrixS2 = (float*) malloc(sizeof(float) * M * (N/BLOCK_WIDTH)); // M x P
    float * Final_out1 = (float*) malloc(sizeof(float) * M * N); // M x P
    float * Final_out2 = (float*) malloc(sizeof(float) * M * N); // M x P
    float * Final_output = (float*) malloc(sizeof(float) * M * N); // M x P
    float * Seq_Sat = (float*) malloc(sizeof(float) * M * N); // M x P

    MatrixA = img.data();
 
    // seed the Pseudo Random Number Generato (PRNG) with srand
    // short seed = short( time(NULL) % RAND_MAX );
    // srand( seed );
 
    // for(int i=0;i<M*N;i++) MatrixA[i] = GetRandomValue(0.0f, 10.0f); // randomly generated

    clock_t wrapper_start = clock();
    ScanKernel(MatrixA, MatrixC, MatrixS, MatrixS2,Final_out1, M,N); // gpu
    clock_t wrapper_stop = clock();
    double wrapper_time_spent = (double)(wrapper_stop - wrapper_start) / CLOCKS_PER_SEC;

    float * MatrixA_T = do_transpose(Final_out1,M,N);
    ScanKernel(MatrixA_T , MatrixC, MatrixS, MatrixS2,Final_out2, N, M); // gpu
 
    Final_output = do_transpose(Final_out2,N,M);

    cout<< "Time for wrapper function: " << wrapper_time_spent << "msec"<< endl<<endl;

    clock_t seq_start = clock();
    Seq_Sat = do_SAT(MatrixA, M, N); // sequential
    clock_t seq_stop = clock();
    double seq_time_spent = (double)(seq_stop - seq_start) / CLOCKS_PER_SEC;
    cout<< "Time for sequential function: " << seq_time_spent << "msec" << endl;

    cout<< endl<<endl;

    cout<< "Output Validation:\n";

    cout<< "GPU total area = " << Final_output[Drow*N+Dcol] + Final_output[Arow*N+Acol] - Final_output[Brow*N+Bcol] - Final_output[Crow*N+Ccol] <<endl;
    cout<< "Seq total area = " << Seq_Sat[Drow*N+Dcol] + Seq_Sat[Arow*N+Acol] - Seq_Sat[Brow*N+Bcol] - Seq_Sat[Crow*N+Ccol] <<endl;

    cout<<endl<<endl;


    // printMat(Final_output, "Final_out", M, N);
    // printMat(Seq_Sat, "Seq_Sat", M, N);

    // comparing both methods ----------------------------------------------------------------------

    double gflop = (double)((double)M * (double)N * 2 - (double)M - (double)N)*1E-9;
    cout << "Gflops: "<< gflop <<endl;
    cout << "Sequential Gflop / msec : " << gflop /  (seq_time_spent) <<endl;
    cout << "Parallel Gflop / msec : " << gflop /  (wrapper_time_spent) <<endl;
    cout << "Speed Up = " << seq_time_spent / wrapper_time_spent << " times speed up\n";

    cout<<endl<<endl;
    return 0;
}