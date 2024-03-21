// Author: Ruturaj

#ifndef _GPU_OPS_H_
#define _GPU_OPS_H_

#include <iostream>
#include <hip/hip_runtime.h>

namespace gpu_ops {

enum ElementType { BF16, F16, F32 };

struct gemmConfig {
    int32_t M;
    int32_t N;
    int32_t K;
    ElementType DataType;
};

void gemm_runner(hipStream_t stream, void **buffers, const char *opaque,
  std::size_t opaque_len);

} // namespace gpu_ops

#endif