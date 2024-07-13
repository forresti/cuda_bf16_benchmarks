#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include "helpers.h"

namespace cg = cooperative_groups;

void fill_array(float* arr, size_t N){
    for(int i=0; i<N; i++){
        // arr[i] = (float)i;
        arr[i] = (float)(i%100);  // TODO(fni): revert this. trying it because it might be easier for bf16
    }
}

void print_array(float* arr, size_t N){
    for(int i=0; i<N; i++){
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

void diff_arrays(float* arr1, float* arr2, size_t N){
    double diff = 0.0f;
    for(int i=0; i<N; i++){
        diff += arr2[i] - arr1[i];
    }
    double avg_diff = diff / N;
    std::cout << "    diff_arrays(), diff: " << diff << " avg_diff: " << avg_diff << std::endl;
}

__global__ void float_to_half_kernel(__nv_bfloat16* d_output, float* d_input, int N) {
    auto block = cg::this_thread_block();
    int32_t gid = block.group_index().x;
    int32_t tid = block.thread_index().x;
    int32_t idx = gid*256 + tid;
    if(idx < N){
        d_output[idx] = __float2bfloat16(d_input[idx]); // float to bf16
    }
}

void float_to_half(__nv_bfloat16* d_output, float* d_input, int N){
    dim3 threads = {256, 1, 1};
    dim3 blocks = {N/256, 1, 1};  // assume output_N is divisible by 256
    float_to_half_kernel<<<blocks, threads>>>(d_output, d_input, N);
}

__global__ void half_to_float_kernel(float* d_output, __nv_bfloat16* d_input, int N) {
    auto block = cg::this_thread_block();
    int32_t gid = block.group_index().x;
    int32_t tid = block.thread_index().x;
    int32_t idx = gid*256 + tid;
    if(idx < N){
        d_output[idx] = __bfloat162float(d_input[idx]); // bf16 to float
    }
}

void half_to_float(float* d_output, __nv_bfloat16* d_input, int N){
    dim3 threads = {256, 1, 1};
    dim3 blocks = {N/256, 1, 1};  // assume output_N is divisible by 256
    half_to_float_kernel<<<blocks, threads>>>(d_output, d_input, N);
}


void add_gold(float* input1, float* input2, float* output, size_t N){

    for(int32_t i=0; i<N; i++){
        output[i] = input1[i] + input2[i];
    }
}

__global__ void add_kernel_bf16_unroll(
    const float4* __restrict__ d_input1,
    const float4* __restrict__ d_input2,
    float4* __restrict__ d_output,
    const size_t N
) {
    auto block = cg::this_thread_block();
    int32_t gid = block.group_index().x;
    int32_t tid = block.thread_index().x;
    int32_t out_idx = (gid*block.size() + tid);

    float4 input1 = d_input1[out_idx];
    float4 input2 = d_input2[out_idx];
    float4 output;

    __nv_bfloat162* input1_bf16 = reinterpret_cast<__nv_bfloat162*>(&input1);
    __nv_bfloat162* input2_bf16 = reinterpret_cast<__nv_bfloat162*>(&input2);
    __nv_bfloat162* output_bf16 = reinterpret_cast<__nv_bfloat162*>(&output);

    // each of these does 2 elements at a time
    output_bf16[0] = input1_bf16[0] + input2_bf16[0];
    output_bf16[1] = input1_bf16[1] + input2_bf16[1];
    output_bf16[2] = input1_bf16[2] + input2_bf16[2];
    output_bf16[3] = input1_bf16[3] + input2_bf16[3];

    // write it all at once
    d_output[out_idx] = output;
}

void add_gpu_bf16_unroll(float4* d_input1, float4* d_input2, float4* d_output, size_t N){
    dim3 threads = {256, 1, 1};
    dim3 blocks = {N/(256*8), 1, 1};  // extra /8 because each thread makes 8 outputs
    add_kernel_bf16_unroll<<<blocks, threads>>>(d_input1, d_input2, d_output, N);
}

__global__ void add_kernel_bf16(
    const __nv_bfloat162* __restrict__ d_input1,
    const __nv_bfloat162* __restrict__ d_input2,
    __nv_bfloat162* __restrict__ d_output,
    const size_t N
) {
    auto block = cg::this_thread_block();
    int32_t gid = block.group_index().x;
    int32_t tid = block.thread_index().x;
    int32_t out_idx = (gid*block.size() + tid);

    d_output[out_idx] = d_input1[out_idx] + d_input2[out_idx];
}

void add_gpu_bf16(float4* d_input1, float4* d_input2, float4* d_output, size_t N){
    dim3 threads = {256, 1, 1};
    dim3 blocks = {N/(256*2), 1, 1};  // extra /2 because each thread makes 8 outputs
    add_kernel_bf16<<<blocks, threads>>>((__nv_bfloat162*)d_input1, (__nv_bfloat162*)d_input2, (__nv_bfloat162*)d_output, N);
}

__global__ void add_kernel_fp32_unroll(
    const float4* __restrict__ d_input1,
    const float4* __restrict__ d_input2,
    float4* __restrict__ d_output,
    const size_t N
) {
    auto block = cg::this_thread_block();
    int32_t gid = block.group_index().x;
    int32_t tid = block.thread_index().x;
    int32_t out_idx = 2*(gid*block.size() + tid);

    float4 input1[2] = {d_input1[out_idx], d_input1[out_idx+1]};
    float4 input2[2] = {d_input2[out_idx], d_input2[out_idx+1]};
    float4 output[2];

    output[0].w = input1[0].w + input2[0].w;
    output[0].x = input1[0].x + input2[0].x;
    output[0].y = input1[0].y + input2[0].y;
    output[0].z = input1[0].z + input2[0].z;
    output[1].w = input1[1].w + input2[1].w;
    output[1].x = input1[1].x + input2[1].x;
    output[1].y = input1[1].y + input2[1].y;
    output[1].z = input1[1].z + input2[1].z;

    d_output[out_idx] = output[0];
    d_output[out_idx+1] = output[1];
}

void add_gpu_fp32_unroll(float4* d_input1, float4* d_input2, float4* d_output, size_t N){
    dim3 threads = {256, 1, 1};
    dim3 blocks = {N/(256*8), 1, 1};  // extra /8 because each thread makes 8 outputs
    add_kernel_fp32_unroll<<<blocks, threads>>>(d_input1, d_input2, d_output, N);
}

__global__ void add_kernel_fp32(
    const float* __restrict__ d_input1,
    const float* __restrict__ d_input2,
    float* __restrict__ d_output,
    const size_t N
) {
    auto block = cg::this_thread_block();
    int32_t gid = block.group_index().x;
    int32_t tid = block.thread_index().x;
    int32_t out_idx = (gid*block.size() + tid);

    d_output[out_idx] = d_input1[out_idx] + d_input2[out_idx];
}

void add_gpu_fp32(float4* d_input1, float4* d_input2, float4* d_output, size_t N){
    dim3 threads = {256, 1, 1};
    dim3 blocks = {N/256, 1, 1};
    add_kernel_fp32<<<blocks, threads>>>((float*)d_input1, (float*)d_input2, (float*)d_output, N);
}

void run_add(size_t N, int repeat){

    size_t input_size = N * sizeof(float);
    size_t input_size_half = N * sizeof(__nv_bfloat16);

    float4* h_input1 = (float4*)malloc(input_size);
    float4* h_input2 = (float4*)malloc(input_size);
    float4* h_output = (float4*)malloc(input_size);
    float4* h_output_toCheck = (float4*)malloc(input_size);

    fill_array((float*)h_input1, N); // treat these as floats, not float4 here
    fill_array((float*)h_input2, N); // treat these as floats, not float4 here

    float4* d_input1;
    float4* d_input1_half;
    float4* d_input2;
    float4* d_input2_half;
    float4* d_output;
    float4* d_output_half;
    CHECK_CUDART(cudaMalloc(&d_input1, input_size));
    CHECK_CUDART(cudaMalloc(&d_input1_half, input_size_half));
    CHECK_CUDART(cudaMalloc(&d_input2, input_size));
    CHECK_CUDART(cudaMalloc(&d_input2_half, input_size_half));
    CHECK_CUDART(cudaMalloc(&d_output, input_size));
    CHECK_CUDART(cudaMalloc(&d_output_half, input_size_half));
    CHECK_CUDART(cudaMemcpy(d_input1, h_input1, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDART(cudaMemcpy(d_input2, h_input2, input_size, cudaMemcpyHostToDevice));
    float_to_half((__nv_bfloat16*)d_input1_half, (float*)d_input1, N);
    float_to_half((__nv_bfloat16*)d_input2_half, (float*)d_input2, N);

    // std::cout << "cpu input:" << std::endl;
    // print_array((float*)h_input, 400);

    // CPU
    double start = read_timer();
    add_gold((float*)h_input1, (float*)h_input2, (float*)h_output, N);
    double cpu_latency = read_timer() - start;
    std::cout << "run_add() cpu latency: " << cpu_latency << " sec" << std::endl;
    // std::cout << "cpu results:" << std::endl;
    // print_array((float*)h_output, 400);

    // GPU fp32
    double gpu_latency = 0.0f;
    int warmup_iter = 5;
    for(int i=0; i<(repeat+warmup_iter); i++){
        CHECK_CUDART(cudaMemset(d_output, 0, input_size));

        double start = read_timer();

        // add_gpu_fp32(d_input1, d_input2, d_output, N);
        add_gpu_fp32_unroll(d_input1, d_input2, d_output, N);

        CHECK_CUDART(cudaGetLastError());
        CHECK_CUDART(cudaDeviceSynchronize());
        if(i>=warmup_iter){
            // due to warmup overheads, don't time the zeroth iter.
            gpu_latency += read_timer() - start;
        }
    }
    gpu_latency = gpu_latency / repeat;
    double gb = (double)(2*input_size) / 1e9;
    double gb_per_sec = (double)gb / gpu_latency;

    std::cout << "run_add(): type=fp32, N: " << N << ", " << gb << " GB, latency: " << gpu_latency << " sec, " << gb_per_sec << " GB/s, " << std::endl;

    // check correctness
    CHECK_CUDART(cudaMemcpy(h_output_toCheck, d_output, input_size, cudaMemcpyDeviceToHost));
    diff_arrays((float*)h_output, (float*)h_output_toCheck, N);
    // std::cout << "gpu results:" << std::endl;
    // print_array((float*)h_output_toCheck, 400);
    CHECK_CUDART(cudaMemset(d_output, 0, input_size));

    // GPU bf16
    gpu_latency = 0.0f;
    for(int i=0; i<(repeat+warmup_iter); i++){
        CHECK_CUDART(cudaMemset(d_output_half, 0, input_size_half));

        double start = read_timer();

        // add_gpu_bf16(d_input1_half, d_input2_half, d_output_half, N);
        add_gpu_bf16_unroll(d_input1_half, d_input2_half, d_output_half, N);
        CHECK_CUDART(cudaGetLastError());
        CHECK_CUDART(cudaDeviceSynchronize());
        if(i>=warmup_iter){
            // due to warmup overheads, don't time the zeroth iter.
            gpu_latency += read_timer() - start;
        }
    }
    gpu_latency = gpu_latency / repeat;
    gb = (double)(2*input_size_half) / 1e9;
    gb_per_sec = (double)gb / gpu_latency;

    std::cout << "run_add(): type=bf16, N: " << N << ", " << gb << " GB, latency: " << gpu_latency << " sec, " << gb_per_sec << " GB/s, " << std::endl;

    // check correctness
    half_to_float((float*)d_output, (__nv_bfloat16*)d_output_half, N);
    CHECK_CUDART(cudaMemcpy(h_output_toCheck, d_output, input_size, cudaMemcpyDeviceToHost));
    diff_arrays((float*)h_output, (float*)h_output_toCheck, N);
    // std::cout << "gpu results:" << std::endl;
    // print_array((float*)h_output_toCheck, 400);
}

void memcpy_ubench(int elements, int repeat){
    float4* d_src;
    float4* d_dst;
    double latency = 0.0f;
    size_t size = elements * sizeof(float4);

    CHECK_CUDART(cudaMalloc(&d_src, size));
    CHECK_CUDART(cudaMalloc(&d_dst, size));

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CHECK_CURAND(curandGenerateUniform(gen, (float*)d_src, elements*4));

    for(int i=0; i<(repeat+1); i++){
        double start = read_timer();
        CHECK_CUDART(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice));
        CHECK_CUDART(cudaDeviceSynchronize());
        if(i>0){
            // due to warmup overheads, don't time the zeroth iter.
            latency += read_timer() - start;
        }
    }
    latency = latency / repeat;
    double size_gb = (double) size / 1e9;
    double gb_per_sec =  (double)size_gb / latency;
    std::cout << "memcpy_ubench(): " << latency << " sec, " << gb_per_sec << "GB/s, " << size_gb << " GB" << std::endl;
}
