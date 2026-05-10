
#include <iostream>
#include <chrono>
#include <vector>
#include <tuple>

__global__ void classic_kernel(int* out_i, int* out_j, int* out_k, int width, int height, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * depth;
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

__global__ void arithmetic_kernel(int* out_i, int* out_j, int* out_k, int width, int height, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * depth;
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
    classic_kernel<<<(total+255)/256, 256>>>(i_classic, j_classic, k_classic, width, height, depth);
    cudaDeviceSynchronize();
    auto end1 = std::chrono::high_resolution_clock::now();

    auto start2 = std::chrono::high_resolution_clock::now();
    arithmetic_kernel<<<(total+255)/256, 256>>>(i_arith, j_arith, k_arith, width, height, depth);
    cudaDeviceSynchronize();
    auto end2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> dur1 = end1 - start1;
    std::chrono::duration<double, std::milli> dur2 = end2 - start2;

    bool match = check(i_classic, i_arith, total) && check(j_classic, j_arith, total) && check(k_classic, k_arith, total);

	std::cout << "Test " << width << "x" << height << "x" << depth << ":\n";
    std::cout << "  Classic kernel time:   " << dur1.count() << " ms\n";
    std::cout << "  Arithmetic kernel time: " << dur2.count() << " ms\n";
    std::cout << "  Result match: " << (match ? "YES" : "NO") << "\n\n";

    cudaFree(i_classic); cudaFree(j_classic); cudaFree(k_classic);
    cudaFree(i_arith);   cudaFree(j_arith);   cudaFree(k_arith);
}

int main() {
    std::vector<std::tuple<int, int, int>> sizes = {
        {16, 16, 16},
        {32, 32, 32},
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 64},
        {512, 128, 64},
        {512, 256, 64},
        {1024, 256, 64},
        {1024, 512, 32},
        {1024, 1024, 16}
    };

	for (size_t i = 0; i < sizes.size(); ++i) {
		int w = std::get<0>(sizes[i]);
		int h = std::get<1>(sizes[i]);
		int d = std::get<2>(sizes[i]);
		run_trial(w, h, d);
	}


    return 0;
}
