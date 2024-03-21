// Author: Ruturaj

#include "gpu_ops.h"
#include "pybind11_kernel_helpers.h"


void printMatrix(const float* mat, int N, const std::string& name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << mat[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}


namespace gpu_ops {
  void gemm_runner2(hipStream_t stream, void **buffers) {
    std::cout << "Kernel Code!\n";
    int N = 2;
    float* A = static_cast<float *>(buffers[0]);
    float* B = static_cast<float *>(buffers[1]);
    float* C = static_cast<float *>(buffers[2]);
    printMatrix(A, N, "A");
    printMatrix(B, N, "B");
    printMatrix(C, N, "C");
}
}


namespace {
pybind11::dict gemmRegistrations() {
  pybind11::dict dict;
  dict["gemm_runner"] =
      gpu_ops::EncapsulateFunction(gpu_ops::gemm_runner);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("get_gemm_registrations", &gemmRegistrations);

  m.def("create_gemm_config_descriptor",
    [](
            int32_t M,
            int32_t N,
            int32_t K,
            gpu_ops::ElementType d_type
       ) {
      return gpu_ops::PackDescriptor(gpu_ops::gemmConfig{
          M,
          N,
          K,
          d_type});
    });

  pybind11::enum_<gpu_ops::ElementType>(m, "ElementType")
      .value("BF16", gpu_ops::ElementType::BF16)
      .value("F16", gpu_ops::ElementType::F16)
      .value("F32", gpu_ops::ElementType::F32);
}
} // namespace