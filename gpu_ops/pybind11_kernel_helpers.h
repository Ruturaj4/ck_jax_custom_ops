#ifndef _GPU_OPS_PYBIND11_KERNEL_HELPERS_H_
#define _GPU_OPS_PYBIND11_KERNEL_HELPERS_H_

#include <pybind11/pybind11.h>

#include "kernel_helpers.h"

namespace gpu_ops {

template <typename T> pybind11::bytes PackDescriptor(const T &descriptor) {
  return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T> pybind11::capsule EncapsulateFunction(T *fn) {
  return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

} // namespace gpu_ops

#endif