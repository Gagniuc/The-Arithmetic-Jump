// index_mapping.cu
#include <iostream>
#include <chrono>

#define WIDTH  128
#define HEIGHT 128
#define DEPTH  128
#define TOTAL (WIDTH * HEIGHT * DEPTH)

__global__ void classic_kernel(int* out_i, int* out_j, int* out_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < TOTAL) {
        int count = 0;
        for (int d = 0; d < DEPTH; ++d)
            for (int i = 0; i < HEIGHT; ++i)
                for (int j = 0; j < WIDTH; ++j)
                    if (count++ == idx) {
                        out_i[idx] = i;
                        out_j[idx] = j;
                        out_k[idx] = d;
                    }
    }
}

__global__ void arithmetic_kernel(int* out_i, int* out_j, int* out_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < TOTAL) {
        int k = idx % (WIDTH * HEIGHT);
        int j = k % WIDTH;
        int i = (k - j) / WIDTH;
        int d = (idx - k) / (WIDTH * HEIGHT);
        out_i[idx] = i;
        out_j[idx] = j;
        out_k[idx] = d;
    }
}

bool check(int* a, int* b) {
    for (int i = 0; i < TOTAL; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

int main() {
    int* i_classic; int* j_classic; int* k_classic;
    int* i_arith;   int* j_arith;   int* k_arith;

    cudaMallocManaged(&i_classic, TOTAL * sizeof(int));
    cudaMallocManaged(&j_classic, TOTAL * sizeof(int));
    cudaMallocManaged(&k_classic, TOTAL * sizeof(int));
    cudaMallocManaged(&i_arith,   TOTAL * sizeof(int));
    cudaMallocManaged(&j_arith,   TOTAL * sizeof(int));
    cudaMallocManaged(&k_arith,   TOTAL * sizeof(int));

    auto start1 = std::chrono::high_resolution_clock::now();
    classic_kernel<<<(TOTAL+255)/256, 256>>>(i_classic, j_classic, k_classic);
    cudaDeviceSynchronize();
    auto end1 = std::chrono::high_resolution_clock::now();

    auto start2 = std::chrono::high_resolution_clock::now();
    arithmetic_kernel<<<(TOTAL+255)/256, 256>>>(i_arith, j_arith, k_arith);
    cudaDeviceSynchronize();
    auto end2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dur1 = end1 - start1;
    std::chrono::duration<double, std::milli> dur2 = end2 - start2;

    std::cout << "Classic kernel time:   " << dur1.count() << " ms\n";
    std::cout << "Arithmetic kernel time: " << dur2.count() << " ms\n";

    bool match = check(i_classic, i_arith) && check(j_classic, j_arith) && check(k_classic, k_arith);
    std::cout << "Result match: " << (match ? "YES" : "NO") << "\n";

    cudaFree(i_classic); cudaFree(j_classic); cudaFree(k_classic);
    cudaFree(i_arith);   cudaFree(j_arith);   cudaFree(k_arith);
    return 0;
}
