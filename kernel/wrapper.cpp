// Author: Ruturaj
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <memory>
#include <vector>
#include <numeric>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <hip/hip_runtime.h>

// CK imports.
#include "ck/wrapper/layout.hpp"
#include "ck/wrapper/tensor.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/wrapper/operations/copy.hpp"
#include "ck/wrapper/operations/gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/wrapper/utils/kernel_utils.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"


// RV: TODO: fix the relative imports.
#include "../gpu_ops/gpu_ops.h"
#include "../gpu_ops/kernel_helpers.h"

namespace {
#define DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN,              \
                                                          NAME, ...)           \
  switch (TYPEIN) {                                                            \
  case gpu_ops::ElementType::F32: {                                            \
    using DataType = float;                                                    \
    __VA_ARGS__;                                                               \
    break;                                                                     \
  }                                                                            \
  case gpu_ops::ElementType::F16: {                                            \
    using DataType = ck::half_t;                                               \
    __VA_ARGS__;                                                               \
    break;                                                                     \
  }                                                                            \
  case gpu_ops::ElementType::BF16: {                                           \
    using DataType = ck::bhalf_t;                                              \
    __VA_ARGS__;                                                               \
    break;                                                                     \
  }                                                                            \
  default:                                                                     \
    break;                                                                     \
}

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    template<typename T>
    void CopyFromHost(const T* host_data, std::size_t size)
    {
        (void)hipMemcpy(p_mem_, host_data, size, hipMemcpyHostToDevice);
    }

    template<typename T>
    SimpleDeviceMem(const T* p_, std::size_t mem_size)
    {
        (void)hipMalloc(static_cast<void**>(&p_), mem_size);
    }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};


// Rv: Custom deleter for unique_ptr managing device memory.
struct HipDeleter {
    template<typename T>
    void operator()(T* ptr) {
        (void)hipFree(ptr);
    }
};


// Rv: Assuming M, K, N are defined and have valid values.
template<typename T>
void printMatrix(const T* mat, int N, const std::string& name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << mat[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}


template<typename T>
void printMatrix(const T* mat, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}


template <typename DataType,
          typename GemmTraits,
          ck::index_t scalar_per_vector,
          typename BlockShape,
          typename ThreadLayout>
__global__ void __CK_WRAPPER_LAUNCH_BOUNDS__ DeviceGemm(const void* p_a,
                                                        const void* p_b,
                                                        void* p_c,
                                                        const ck::index_t M,
                                                        const ck::index_t N,
                                                        const ck::index_t K,
                                                        const BlockShape tile_shape,
                                                        const ThreadLayout thread_layout)
{
    constexpr auto MPerBlock = ck::wrapper::size<0>(tile_shape);
    constexpr auto NPerBlock = ck::wrapper::size<1>(tile_shape);
    constexpr auto KPerBlock = ck::wrapper::size<2>(tile_shape);

    // Specify layouts for global memory.
    const auto a_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, K), ck::make_tuple(K, 1));
    const auto b_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(N, K), ck::make_tuple(K, 1));
    const auto c_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, N), ck::make_tuple(N, 1));
    // Specify layouts for tiles.
    constexpr auto a_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(MPerBlock, KPerBlock), ck::make_tuple(KPerBlock, ck::Number<1>{}));
    constexpr auto b_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(NPerBlock, KPerBlock), ck::make_tuple(KPerBlock, ck::Number<1>{}));
    constexpr auto c_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(MPerBlock, NPerBlock), ck::make_tuple(NPerBlock, ck::Number<1>{}));
    // Apply padding for global memory.
    auto a_global_layout_padded = ck::wrapper::pad(a_global_layout, shape(a_tile_layout));
    auto b_global_layout_padded = ck::wrapper::pad(b_global_layout, shape(b_tile_layout));
    auto c_global_layout_padded = ck::wrapper::pad(c_global_layout, shape(c_tile_layout));
    // Make tensors for global memory.
    auto a_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_a), a_global_layout_padded);
    auto b_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_b), b_global_layout_padded);
    auto c_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<DataType*>(p_c), c_global_layout_padded);
    // Allocate lds memory.
    __shared__ DataType lds_a[ck::wrapper::size(a_tile_layout)];
    __shared__ DataType lds_b[ck::wrapper::size(b_tile_layout)];
    // Make tensors for lds memory.
    auto a_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_a), a_tile_layout);
    auto b_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_b), b_tile_layout);
    // Specify block index as tuple.
    const auto block_idxs = ck::make_tuple(static_cast<ck::index_t>(blockIdx.x),
                                           static_cast<ck::index_t>(blockIdx.y),
                                           ck::wrapper::slice());
    // Specify access parameters for copy.
    using DimAccessOrder             = ck::Tuple<ck::Number<0>, ck::Number<1>>;
    constexpr ck::index_t vector_dim = 1;
    // Create tile and partition for C. Use specific function for blockwise_gemm to assign the
    // appropriate partitions.
    auto c_global_local_tile = ck::wrapper::make_local_tile(
        c_global_tensor,
        tile_shape,
        block_idxs,
        make_tuple(ck::Number<1>{}, ck::Number<1>{}, ck::wrapper::slice(KPerBlock)));
    auto c_global_local_partition =
        ck::wrapper::make_blockwise_gemm_xdl_c_local_partition<DataType,
                                                               decltype(a_tile_layout),
                                                               decltype(b_tile_layout),
                                                               ck::wrapper::size(thread_layout),
                                                               GemmTraits>(c_global_local_tile);
    // Create C vgpr to accumulate results.
    auto c_vgpr_reg = ck::wrapper::make_blockwise_gemm_xdl_c_vgpr<DataType,
                                                                  decltype(a_tile_layout),
                                                                  decltype(b_tile_layout),
                                                                  ck::wrapper::size(thread_layout),
                                                                  GemmTraits>();
    // Clear C vgpr.
    ck::wrapper::clear(c_vgpr_reg);

    // Iterate over K with KPerBlock step.
    const ck::index_t num_loop = ck::math::integer_divide_ceil(K, KPerBlock);
    ck::index_t i              = 0;
    do
    {
        // Get KPerBlock slice.
        const auto k_slice           = ck::wrapper::slice(i * KPerBlock, (i + 1) * KPerBlock);
        auto a_global_tensor_k_slice = a_global_tensor(ck::wrapper::slice(), k_slice);
        auto b_global_tensor_k_slice = b_global_tensor(ck::wrapper::slice(), k_slice);
        // Create local tiles for A and B.
        auto a_global_local_tile = ck::wrapper::make_local_tile(
            a_global_tensor_k_slice,
            tile_shape,
            block_idxs,
            make_tuple(ck::Number<1>{}, ck::wrapper::slice(N), ck::Number<1>{}));
        auto b_global_local_tile = ck::wrapper::make_local_tile(
            b_global_tensor_k_slice,
            tile_shape,
            block_idxs,
            make_tuple(ck::wrapper::slice(M), ck::Number<1>{}, ck::Number<1>{}));
        // Copy from global to lds.
        ck::wrapper::blockwise_copy<DimAccessOrder, vector_dim, scalar_per_vector>(
            a_global_local_tile, a_lds_tensor, thread_layout);
        ck::wrapper::blockwise_copy<DimAccessOrder, vector_dim, scalar_per_vector>(
            b_global_local_tile, b_lds_tensor, thread_layout);
        // Synchronize lds.
        ck::block_sync_lds();
        // Execute blockwise gemm.
        ck::wrapper::blockwise_gemm_xdl<DataType, ck::wrapper::size(thread_layout), GemmTraits>(
            a_lds_tensor, b_lds_tensor, c_vgpr_reg);

        ++i;
    } while(i < num_loop);
    // Copy vgpr results to C global memory.
    ck::wrapper::copy(c_vgpr_reg, c_global_local_partition);
}


template <typename DataType,
          typename GemmTraits,
          ck::index_t scalar_per_vector,
          typename BlockShape,
          typename ThreadLayout>
void PerformGemm(const ck::index_t M,
                 const ck::index_t N,
                 const ck::index_t K,
                 const BlockShape& tile_shape,
                 const ThreadLayout& thread_layout,
                 hipStream_t stream,
                 void** buffers)
{
    DataType* A = static_cast<DataType *>(buffers[0]);
    DataType* B = static_cast<DataType *>(buffers[1]);
    DataType* C = static_cast<DataType *>(buffers[2]);

    std::cout << "perform gemm!\n";

    // printMatrix(A, M, "A");
    // printMatrix(B, M, "B");

    size_t bytes = N * N * sizeof(DataType);
    std::unique_ptr<DataType*, HipDeleter> d_A, d_B, d_C;
    (void)hipMallocManaged(reinterpret_cast<void**>(&d_A), bytes);
    (void)hipMallocManaged(reinterpret_cast<void**>(&d_B), bytes);
    (void)hipMallocManaged(reinterpret_cast<void**>(&d_C), bytes);

    // Copy data from host to device.
    (void)hipMemcpyAsync(d_A.get(), A, bytes, hipMemcpyHostToDevice, nullptr);
    (void)hipMemcpyAsync(d_B.get(), B, bytes, hipMemcpyHostToDevice, nullptr);

    // printMatrix((DataType*)d_A.get(), M, K, "A");

    (void)hipDeviceSynchronize();

    const ck::index_t grid_size_x =
        ck::math::integer_divide_ceil(M, ck::wrapper::size<0>(tile_shape));
    const ck::index_t grid_size_y =
        ck::math::integer_divide_ceil(N, ck::wrapper::size<1>(tile_shape));

    const auto kernel =
        DeviceGemm<DataType, GemmTraits, scalar_per_vector, BlockShape, ThreadLayout>;
    const float avg_time = launch_and_time_kernel(StreamConfig{nullptr, true},
                                                  kernel,
                                                  dim3(grid_size_x, grid_size_y, 1),
                                                  dim3(ck::wrapper::size(thread_layout)),
                                                  0,
                                                  d_A.get(),
                                                  d_B.get(),
                                                  d_C.get(),
                                                  M,
                                                  N,
                                                  K,
                                                  tile_shape,
                                                  thread_layout);

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(DataType) * M * K + sizeof(DataType) * K * N + sizeof(DataType) * M * N;

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;

    (void)hipDeviceSynchronize();

    // Copy results back to host.
    (void)hipMemcpyAsync(C, d_C.get(), bytes, hipMemcpyDeviceToHost, nullptr);

    // printMatrix((DataType*)d_C.get(), M, N, "C");

    (void)hipFree(d_A.get());
    (void)hipFree(d_B.get());
    (void)hipFree(d_C.get());

    std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << std::endl;
}
} // namespace

namespace gpu_ops {
void gemm_runner(hipStream_t stream, void **buffers, const char *opaque,
                        std::size_t opaque_len) {
    const gemmConfig &d =
        *gpu_ops::UnpackDescriptor<gemmConfig>(opaque, opaque_len);

    std::cout << d.DataType << std::endl;
    std::cout << gpu_ops::ElementType::F32 << std::endl;

    const auto thread_layout =
        ck::wrapper::make_layout(ck::make_tuple(ck::Number<64>{}, ck::Number<4>{}),
                                    ck::make_tuple(ck::Number<4>{}, ck::Number<1>{}));
    const auto tile_shape = ck::make_tuple(ck::Number<256>{}, ck::Number<128>{}, ck::Number<32>{});

    // #Define hack is required to be able to support multiple types.
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        // First argument is type. We can pass multiple args to #define to support multiple such types.
        d.DataType, "gemm_kernel",
        PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_8K1, 8>(
        d.M, 
        d.N, 
        d.K,
        // static_cast<ck::index_t>(d.K), 
        tile_shape,
        thread_layout,
        stream,
        buffers);
)
}


// void gemm_runner_float(hipStream_t stream, void **buffers) {

// using DataType = float;

// // printMatrix(static_cast<DataType *>(buffers[0]), 16, "A in gr");
// const auto thread_layout =
//     ck::wrapper::make_layout(ck::make_tuple(ck::Number<64>{}, ck::Number<4>{}),
//                                 ck::make_tuple(ck::Number<4>{}, ck::Number<1>{}));
// const auto tile_shape = ck::make_tuple(ck::Number<256>{}, ck::Number<128>{}, ck::Number<32>{});
// PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_8K1, 8>(
//     16, 16, 16, tile_shape, thread_layout, stream, buffers);
// }

}


// int main(int argc, char* argv[])
// {
//     hipStream_t stream;

//     const int M = 16;
//     const int N = 16;
//     const int K = 16;

//     // Initialize matrices A and B in global memory, filled with 1s
//     float* A = new float[M * K];
//     float* B = new float[K * N];
//     float* C = new float[M * N];
//     std::fill_n(A, M * K, 1.0f); // Set all elements in A to 1
//     std::fill_n(B, K * N, 1.0f); // Set all elements in B to 1
//     std::fill_n(B, K * N, 1.0f); // Set all elements in B to 1

//     // Prepare the array of host buffer pointers
//     void* buffers[3] = {A, B, C};

//     gpu_ops::gemm_runner(stream, buffers);


//     delete [] A;
//     delete [] B;
//     delete [] C;
//     return 0;
// }