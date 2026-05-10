
#include <iostream>
#include <chrono>

__global__ void classic_kernel(int* out_i, int* out_j, int* out_k, int width, int height, int depth, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int count = 0;
        for (int d = 0; d < depth; ++d)
            for (int i = 0; i < height; ++i)
                for (int j = 0; j < width; ++j)
                    if (count++ == idx) {
                        out_i[idx] = i;
                        out_j[idx] = j;
                        out_k[idx] = d;
                    }
    }
}

__global__ void arithmetic_kernel(int* out_i, int* out_j, int* out_k, int width, int height, int depth, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int k = idx % (width * height);
        int j = k % width;
        int i = (k - j) / width;
        int d = (idx - k) / (width * height);
        out_i[idx] = i;
        out_j[idx] = j;
        out_k[idx] = d;
    }
}

bool check(int* a, int* b, int total) {
    for (int i = 0; i < total; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

void run_trial(int width, int height, int depth) {
    int total = width * height * depth;

    int* i_classic; int* j_classic; int* k_classic;
    int* i_arith;   int* j_arith;   int* k_arith;

    cudaMallocManaged(&i_classic, total * sizeof(int));
    cudaMallocManaged(&j_classic, total * sizeof(int));
    cudaMallocManaged(&k_classic, total * sizeof(int));
    cudaMallocManaged(&i_arith,   total * sizeof(int));
    cudaMallocManaged(&j_arith,   total * sizeof(int));
    cudaMallocManaged(&k_arith,   total * sizeof(int));

    auto start1 = std::chrono::high_resolution_clock::now();
    classic_kernel<<<(total+255)/256, 256>>>(i_classic, j_classic, k_classic, width, height, depth, total);
    cudaDeviceSynchronize();
    auto end1 = std::chrono::high_resolution_clock::now();

    auto start2 = std::chrono::high_resolution_clock::now();
    arithmetic_kernel<<<(total+255)/256, 256>>>(i_arith, j_arith, k_arith, width, height, depth, total);
    cudaDeviceSynchronize();
    auto end2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dur1 = end1 - start1;
    std::chrono::duration<double, std::milli> dur2 = end2 - start2;

    bool match = check(i_classic, i_arith, total) && check(j_classic, j_arith, total) && check(k_classic, k_arith, total);

    std::cout << "Test " << width << "×" << height << "×" << depth << ": \n";
    std::cout << "  Classic kernel time:   " << dur1.count() << " ms\n";
    std::cout << "  Arithmetic kernel time: " << dur2.count() << " ms\n";
    std::cout << "  Result match: " << (match ? "YES" : "NO") << "\n\n";

    cudaFree(i_classic); cudaFree(j_classic); cudaFree(k_classic);
    cudaFree(i_arith);   cudaFree(j_arith);   cudaFree(k_arith);
}

int main() {
    run_trial(16, 16, 16);
    run_trial(32, 32, 32);
    run_trial(64, 64, 64);
    run_trial(128, 128, 128);
    run_trial(256, 256, 64);
    run_trial(512, 128, 64);
    run_trial(512, 256, 64);
    return 0;
}
