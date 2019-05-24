#include "stdio.h"
#include "cuda.h"
#include "stdlib.h"

const int N = 1024;

__global__ void addNN(int* a, int* b, int* c) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add11(int* a, int* b, int* c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int* host_a;
    int* host_b;
    int* host_c;

    int* dev_a;
    int* dev_b;
    int* dev_c;

    // allocate memory on host
    cudaHostAlloc((void**)&host_a, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, N * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, N * sizeof(int), cudaHostAllocDefault);

    // allocate memory on device
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++) {
        host_a[i] = -i + 1;
        host_b[i] = i * i;
    }

    // copy data to device
    cudaMemcpy((void*)dev_a, (void*)host_a, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dev_b, (void*)host_b, sizeof(int)*N, cudaMemcpyHostToDevice);
    
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    add11<<<1,1>>>(dev_a, dev_b, dev_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf("GPU compute time: %f\n", time);
    cudaEventRecord(stop);

    cudaMemcpy((void*)host_c, (void*)dev_c, sizeof(int)*N, cudaMemcpyDeviceToHost);

    // display the results
    for (int i=0; i<N; i++) {
        printf("%d ", host_c[i]);
    }

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
