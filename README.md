# CK JAX custom Ops
AMD Composable Kernel (CK) kernel Integration in Jax using Custom Ops.

https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html

## Necessity to create this repository.
The provided documentation (cutom ops) by JAX leverages CUDA kernel to show integration with JAX. Additionally, there is plenty of documentation online for CUDA integration elsewhere. However, there is not much documentation available on AMD's Composable Kernel (CK) kernel integration with JAX using custom ops.

How to run ->

```sh
> bash run.bash
```

Tested on ROCm ->
```
rocm-6.0.0/
```

Jax dependencies ->
```sh
> pip list | grep jax
jax                0.4.24
jaxlib             0.4.24+rocm600
```
